"""Weight-format adapters.

The architecture lives in ``moss_lora.model`` (upstream code). This module only maps
the on-disk layouts used by the shipped repack and the stock Qwen3.5 checkpoint
onto the modules we build, plus the latent packing the reference pipeline expects.
"""

from __future__ import annotations

import glob
import json
import os

import torch
from safetensors import safe_open
from safetensors.torch import load_file

from .model import MOSSImageTransformer

BN_EPS = 1e-4


# ---------------------------------------------------------------------------
# DiT
# ---------------------------------------------------------------------------

def split_fused_qkv(state: dict) -> dict:
    """The shipped repack merges q/k/v into one [3*dim, dim] weight; the reference keeps them apart."""
    if not any(k.endswith(".proj.qkv.weight") for k in state):
        return state
    out = {}
    for key, value in state.items():
        if key.startswith("layers.") and key.endswith(".proj.qkv.weight"):
            q, k, v = value.chunk(3, dim=0)
            base = key[: -len("qkv.weight")]
            out[base + "q.weight"] = q.contiguous()
            out[base + "k.weight"] = k.contiguous()
            out[base + "v.weight"] = v.contiguous()
        else:
            out[key] = value
    return out


# Which loader ``load_transformer`` uses. "streamed" builds the model block by
# block and never holds the whole checkpoint as bf16 parameters; "full" is the
# old read-everything-then-fill-it path, kept only so the two can be compared.
LOAD_MODE = "streamed"

# Where nf4 packing runs: "auto" uses the card when there is one, "cpu" packs in
# host memory. Packing a 5B model takes 45 seconds on the CPU and 0.2 on the
# card, and a block's bf16 copy on the card is 0.23 GiB, so "auto" means the card.
QUANTIZE_DEVICE = "auto"

TRANSFORMER_BLOCKS = 42


def load_transformer(
    path: str,
    dtype=torch.bfloat16,
    device: str | torch.device = "cpu",
    *,
    quantize_scope: str = "none",
    quantize_dtype: str = "nf4",
    blocks_to_swap: int = 0,
    quantize_conditioning: str = "none",
    log=None,
) -> MOSSImageTransformer:
    """Load the DiT, quantizing on the way in when a format is set.

    The streamed loader is the default: it builds the model on the meta device
    and fills it one block at a time, so the bf16 checkpoint never exists as a
    second copy in host memory and the random initialisation never runs. With
    ``quantize`` set, each block is packed before the next one is read, which is
    what makes a streamed nf4 run fit on a 16 GB host.
    """
    if LOAD_MODE == "full" or os.path.isdir(path):
        return _load_transformer_full(path, dtype=dtype, device=device,
                                      quantize_scope=quantize_scope, quantize_dtype=quantize_dtype,
                                      blocks_to_swap=blocks_to_swap,
                                      quantize_conditioning=quantize_conditioning, log=log)
    return _load_transformer_streamed(path, dtype=dtype, device=device,
                                      quantize_scope=quantize_scope, quantize_dtype=quantize_dtype,
                                      blocks_to_swap=blocks_to_swap,
                                      quantize_conditioning=quantize_conditioning, log=log)


def _note(log, message: str) -> None:
    if log is not None:
        log(message)


def _target(model, key: str):
    try:
        return model.get_parameter(key)
    except AttributeError:
        return model.get_buffer(key)


def _max_block_bytes(model) -> int:
    return max(sum(p.numel() * p.element_size() for p in block.parameters())
               for block in model.layers)


def _fill_cpu(model, state) -> None:
    """Copy one block's mmap views into parameters that live in host memory."""
    for key, tensor in state.items():
        _target(model, key).data.copy_(tensor)


def _fill_device(model, state, buffer) -> None:
    """Fill one block's parameters through a pinned staging buffer.

    A pageable host buffer reaches the card at about 4.7 GiB/s, a pinned one at
    22, and the difference is the whole load time for a bf16 model. Each tensor
    takes one copy into the pinned buffer and one asynchronous copy up; the
    caller syncs before reusing the buffer for the next block.
    """
    offset = 0
    plan = []
    for key, tensor in state.items():
        target = _target(model, key)
        nbytes = target.numel() * target.element_size()
        chunk = buffer[offset:offset + nbytes]
        view = chunk.view(target.dtype).reshape(target.shape)
        view.copy_(tensor)  # casts when the file and the run disagree on dtype
        plan.append((target, view))
        offset += nbytes
    for target, source in plan:
        target.data.copy_(source, non_blocking=True)
    torch.cuda.synchronize()


def _load_transformer_streamed(path: str, *, dtype, device, quantize_scope: str, quantize_dtype: str,
                               blocks_to_swap: int, quantize_conditioning: str, log) -> MOSSImageTransformer:
    from .quant import quantize_conditioning as quantize_conditioning_half
    from .quant import quantize_fp8, quantize_nf4

    scope = str(quantize_scope or "none").lower()
    fmt = str(quantize_dtype or "nf4").lower()
    pack = quantize_fp8 if fmt == "fp8" else quantize_nf4
    on_card = str(QUANTIZE_DEVICE).lower() != "cpu" and torch.cuda.is_available()

    with torch.device("meta"):
        model = MOSSImageTransformer(dtype=dtype)
    model.to_empty(device="cpu")
    model.eval()
    blocks = len(model.layers)
    swap = max(0, min(blocks, int(blocks_to_swap or 0)))
    buffer = torch.empty(_max_block_bytes(model), dtype=torch.uint8,
                         pin_memory=torch.cuda.is_available(), device="cpu")

    with safe_open(path, framework="pt") as handle:
        keys = list(handle.keys())
        block_keys = {index: [k for k in keys if k.startswith(f"layers.{index}.")]
                      for index in range(blocks)}
        head = {k: handle.get_tensor(k) for k in keys if not k.startswith("layers.")}
        head.pop("rope.bands", None)  # constant rebuilt by _RoPE
        model.load_state_dict(head, strict=False)
        del head
        if str(quantize_conditioning or "none").lower() != "none":
            replaced = quantize_conditioning_half(model, quantize_conditioning, compute_dtype=dtype)
            _note(log, f"conditioning: {replaced} Linears -> {quantize_conditioning}")
        # Only the conditioning half goes up now; the blocks follow one at a time
        # and a streamed tail never touches the card unless it is being packed.
        for name, module in model.named_children():
            if name != "layers":
                module.to(device)

        for index in range(blocks):
            state = split_fused_qkv({k: handle.get_tensor(k) for k in block_keys[index]})
            block = model.layers[index]
            offloaded = index >= blocks - swap
            if scope == "none":
                if offloaded:
                    _fill_cpu(model, state)
                else:
                    block.to_empty(device=device)
                    _fill_device(model, state, buffer)
            else:
                block.to_empty(device=device)
                _fill_device(model, state, buffer)
                kwargs = {"device": device} if (on_card and fmt == "nf4") else {}
                replaced = pack(block, scope, dtype=dtype, prefix="", **kwargs)
                if offloaded:
                    # The packed weight goes back to host memory; the scales stay
                    # on the card, where the swap expects them.
                    for parameter in block.parameters():
                        parameter.data = parameter.data.to("cpu")
                if index == 0:
                    _note(log, f"quantize: {replaced} frozen Linears per block -> {fmt} ({scope})")
            del state
    return model.eval()


def _load_transformer_full(path: str, *, dtype, device, quantize_scope: str, quantize_dtype: str,
                           blocks_to_swap: int, quantize_conditioning: str, log) -> MOSSImageTransformer:
    """The pre-streaming loader: read the file, build the model, fill it."""
    from .quant import quantize_conditioning as quantize_conditioning_half
    from .quant import quantize_fp8, quantize_nf4

    scope = str(quantize_scope or "none").lower()
    fmt = str(quantize_dtype or "nf4").lower()
    pack = quantize_fp8 if fmt == "fp8" else quantize_nf4
    on_host = scope != "none"

    if os.path.isdir(path):
        state = _load_dir_state(path, "diffusion_pytorch_model")
    else:
        state = load_file(path)
    state = split_fused_qkv(state)
    state.pop("rope.bands", None)
    with torch.device("meta"):
        model = MOSSImageTransformer(dtype=dtype)
    model.to_empty(device="cpu" if on_host else device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    missing = [k for k in missing if k != "rope.bands"]
    if unexpected or missing:
        raise RuntimeError(f"DiT key mismatch: missing={missing[:4]} unexpected={unexpected[:4]}")
    del state
    if scope != "none":
        replaced = pack(model, scope, dtype=dtype)
        _note(log, f"quantize: {replaced} frozen Linears -> {fmt} ({scope})")
        if str(quantize_conditioning or "none").lower() != "none":
            replaced = quantize_conditioning_half(model, quantize_conditioning, compute_dtype=dtype)
            _note(log, f"conditioning: {replaced} Linears -> {quantize_conditioning}")
        model.to(device)
    return model.eval()


def _load_dir_state(path: str, prefix: str) -> dict:
    index_path = os.path.join(path, f"{prefix}.safetensors.index.json")
    if os.path.isfile(index_path):
        with open(index_path) as f:
            index = json.load(f)
        files = sorted({os.path.join(path, name) for name in index["weight_map"].values()})
    else:
        files = sorted(glob.glob(os.path.join(path, "*.safetensors")))
    if not files:
        raise FileNotFoundError(f"no safetensors under {path}")
    state = {}
    for file in files:
        state.update(load_file(file))
    return state


# ---------------------------------------------------------------------------
# VAE
# ---------------------------------------------------------------------------

def pack_latents(latents: torch.Tensor) -> torch.Tensor:
    """(B,32,H,W) -> (B,128,H/2,W/2)."""
    b, c, h, w = latents.shape
    return latents.reshape(b, c, h // 2, 2, w // 2, 2).permute(0, 1, 3, 5, 2, 4).reshape(b, c * 4, h // 2, w // 2)


def unpack_latents(latents: torch.Tensor) -> torch.Tensor:
    """(B,128,H,W) -> (B,32,2H,2W)."""
    b, c, h, w = latents.shape
    return latents.reshape(b, c // 4, 2, 2, h, w).permute(0, 1, 4, 2, 5, 3).reshape(b, c // 4, h * 2, w * 2)


def legacy_vae_to_diffusers(state: dict) -> dict:
    """Legacy repacked VAE keys -> diffusers ``AutoencoderKLFlux2``."""
    out = {}
    for key, value in state.items():
        if ".mid.attn_1." in key:
            prefix, _, rest = key.partition(".mid.attn_1.")
            head, _, tail = rest.partition(".")
            name = {"q": "to_q", "k": "to_k", "v": "to_v", "proj_out": "to_out.0", "norm": "group_norm"}.get(head, head)
            if value.ndim == 4:  # legacy 1x1 conv == Linear
                value = value.reshape(value.shape[0], value.shape[1])
            out[f"{prefix}.mid_block.attentions.0.{name}.{tail}"] = value
            continue
        if key.startswith(("encoder.norm_out.", "decoder.norm_out.")):
            side = key.split(".")[0]
            out[f"{side}.conv_norm_out.{key[len(f'{side}.norm_out.'):]}"] = value
            continue
        if key.startswith("encoder.quant_conv."):
            out["quant_conv." + key[len("encoder.quant_conv."):]] = value
            continue
        if key.startswith("decoder.post_quant_conv."):
            out["post_quant_conv." + key[len("decoder.post_quant_conv."):]] = value
            continue
        mapped = False
        for side, blocks, store in (("encoder", "down", "down_blocks"), ("decoder", "up", "up_blocks")):
            prefix = f"{side}.{blocks}."
            if not key.startswith(prefix):
                continue
            index, _, tail = key[len(prefix):].partition(".")
            index = int(index)
            if side == "decoder":  # legacy order lists the highest resolution first
                index = 3 - index
            parts = tail.replace("block.", "").split(".")
            if parts[0] == "downsample":
                out[f"{side}.{store}.{index}.downsamplers.0.conv.{parts[-1]}"] = value
            elif parts[0] == "upsample":
                out[f"{side}.{store}.{index}.upsamplers.0.conv.{parts[-1]}"] = value
            else:
                name = parts[1].replace("nin_shortcut", "conv_shortcut")
                out[f"{side}.{store}.{index}.resnets.{parts[0]}.{name}.{parts[2]}"] = value
            mapped = True
            break
        if mapped:
            continue
        if ".mid.block_1." in key:
            out[key.replace(".mid.block_1.", ".mid_block.resnets.0.")] = value
            continue
        if ".mid.block_2." in key:
            out[key.replace(".mid.block_2.", ".mid_block.resnets.1.")] = value
            continue
        out[key] = value
    return out


def load_vae(path: str, dtype=torch.float16, device: str | torch.device = "cpu"):
    """Return a diffusers ``AutoencoderKLFlux2`` with the reference latent transform."""
    from diffusers import AutoencoderKLFlux2

    if os.path.isdir(path):
        vae = AutoencoderKLFlux2.from_pretrained(path, torch_dtype=dtype)
    else:
        config = {
            "act_fn": "silu", "batch_norm_eps": BN_EPS, "batch_norm_momentum": 0.1,
            "block_out_channels": [128, 256, 512, 512],
            "decoder_block_out_channels": [128, 256, 512, 512],
            "down_block_types": ["DownEncoderBlock2D"] * 4, "force_upcast": True,
            "in_channels": 3, "latent_channels": 32, "layers_per_block": 2,
            "mid_block_add_attention": True, "norm_num_groups": 32, "out_channels": 3,
            "patch_size": [2, 2], "sample_size": 1024,
            "up_block_types": ["UpDecoderBlock2D"] * 4,
            "use_post_quant_conv": True, "use_quant_conv": True,
        }
        vae = AutoencoderKLFlux2.from_config(config)
        missing, unexpected = vae.load_state_dict(legacy_vae_to_diffusers(load_file(path)), strict=False)
        if missing or unexpected:
            raise RuntimeError(f"VAE key mismatch: missing={missing[:4]} unexpected={unexpected[:4]}")
    return vae.eval().requires_grad_(False).to(dtype).to(device)


class LatentCodec:
    """Patchify + BatchNorm normalisation, matching MOSSImagePipeline.__call__."""

    def __init__(self, vae, dtype=torch.float16):
        self.vae = vae
        self.dtype = dtype
        # cuDNN runs float32 convolutions on TF32 tensor cores by default, which
        # rounds the mantissa to 10 bits: the same image then encodes differently
        # at batch 1 and batch 4 (measured 3.9e-3), and TF32 is slower here anyway
        # (268 ms/image against 167). Precision first, and batch stops mattering.
        torch.backends.cudnn.allow_tf32 = False
        bn = vae.bn
        self.mean = bn.running_mean.detach().float().view(1, -1, 1, 1)
        self.std = (bn.running_var.detach().float() + BN_EPS).sqrt().view(1, -1, 1, 1)

    @torch.no_grad()
    def encode(self, images: torch.Tensor, sample: bool = False) -> torch.Tensor:
        dist = self.vae.encode(images.to(self.vae.dtype)).latent_dist
        latents = dist.sample() if sample else dist.mode()
        latents = pack_latents(latents.float())
        mean = self.mean.to(latents.device, latents.dtype)
        std = self.std.to(latents.device, latents.dtype)
        return ((latents - mean) / std).to(self.dtype)

    @torch.no_grad()
    def decode(self, latents: torch.Tensor, chunk: int = 1) -> torch.Tensor:
        mean = self.mean.to(latents.device, latents.dtype)
        std = self.std.to(latents.device, latents.dtype)
        device = next(self.vae.parameters()).device
        z = unpack_latents(latents * std + mean)
        # The fp32 decoder holds several full resolution feature maps per sample,
        # so a three image preview costs ~9 GiB more than one image does. Decode
        # one sample at a time: same output, flat peak.
        if chunk <= 0 or z.shape[0] <= chunk:
            return self.vae.decode(z.to(device=device, dtype=self.vae.dtype)).sample
        parts = [
            self.vae.decode(z[i:i + chunk].to(device=device, dtype=self.vae.dtype)).sample
            for i in range(0, z.shape[0], chunk)
        ]
        return torch.cat(parts, dim=0)


# ---------------------------------------------------------------------------
# Text encoder (Qwen3.5-9B, frozen)
# ---------------------------------------------------------------------------
