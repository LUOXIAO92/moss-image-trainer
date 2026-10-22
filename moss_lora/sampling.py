"""Sampling orchestration (Euler + CFG), mirroring ``MOSSImagePipeline.__call__``.

The model and the schedule are upstream code; only the loop lives here so that
training previews reuse the exact inference path.
"""

from __future__ import annotations

import torch

from .scheduler import MOSSImageScheduler

STRIDE = 16


@torch.no_grad()
def encode_prompts(text_encoder, tokenizer, prompts, max_length: int = 4096):
    """Chat-templated, unpadded forward per prompt -> (embeds [B,S,C], mask [B,S])."""
    if isinstance(prompts, str):
        prompts = [prompts]
    rows = []
    for prompt in prompts:
        formatted = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False, add_generation_prompt=True, enable_thinking=True,
        )
        inputs = tokenizer(formatted, return_tensors="pt", truncation=True, max_length=max_length)
        inputs = {k: v.to(text_encoder.device) for k, v in inputs.items()}
        out = text_encoder(**inputs, use_cache=False, return_dict=True)
        rows.append(out.last_hidden_state[0])
    length = max(r.shape[0] for r in rows)
    embeds = rows[0].new_zeros(len(rows), length, rows[0].shape[-1])
    mask = torch.zeros(len(rows), length, dtype=torch.bool, device=embeds.device)
    for i, row in enumerate(rows):
        embeds[i, : row.shape[0]] = row
        mask[i, : row.shape[0]] = True
    return embeds, mask


@torch.no_grad()
def sample(
    model,
    codec,
    embeds: torch.Tensor,
    mask: torch.Tensor,
    negative_embeds: torch.Tensor | None = None,
    negative_mask: torch.Tensor | None = None,
    height: int = 1024,
    width: int = 1024,
    steps: int = 50,
    guidance: float = 3.0,
    shift: float = 5.0,
    uncond_skip_layers: tuple[int, ...] = (),
    generator: torch.Generator | None = None,
    device: str | torch.device = "cuda",
    dtype=torch.bfloat16,
):
    """Euler integration with CFG; returns decoded images in [-1, 1]."""
    if height % STRIDE or width % STRIDE:
        raise ValueError(f"height/width must be multiples of {STRIDE}")
    if negative_embeds is not None and negative_embeds.shape[0] != embeds.shape[0]:
        repeat = embeds.shape[0] // negative_embeds.shape[0]
        negative_embeds = negative_embeds.repeat_interleave(repeat, 0)
        negative_mask = negative_mask.repeat_interleave(repeat, 0)
    scheduler = MOSSImageScheduler(shift=shift)
    scheduler.set_timesteps(steps, device=device)
    # the reference samples initial noise in fp32 on the generator's device
    gen_device = generator.device if generator is not None else "cpu"
    latents = torch.randn(
        embeds.shape[0], model.latent_channels, height // STRIDE, width // STRIDE,
        generator=generator, dtype=torch.float32, device=gen_device,
    ).to(device)
    for timestep in scheduler.timesteps:
        prediction = model(latents.to(dtype), timestep, embeds, mask).float()
        if guidance != 1.0 and negative_embeds is not None:
            # SLG "uncond skip": run the unconditional branch with those layers bypassed
            previous = getattr(model, "skip_layers", ())
            model.skip_layers = tuple(uncond_skip_layers)
            uncond = model(latents.to(dtype), timestep, negative_embeds, negative_mask).float()
            model.skip_layers = previous
            prediction = uncond + guidance * (prediction - uncond)
        latents = scheduler.step(prediction, timestep, latents, return_dict=False)[0]
    return codec.decode(latents)
