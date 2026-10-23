"""Convert a weight-only checkpoint into a LoRA file without building the model.

`Trainer.export` needs the 5B model in memory only because it reads the PEFT module
alphas; the tensors themselves are already in the checkpoint under names like
``base_model.model.layers.0.proj.q.lora_A.default.weight``.  Renaming those keys is
enough, so a single-step export becomes a sub-second file write instead of a model
load.
"""

from __future__ import annotations

import os
import re

import torch
from safetensors.torch import load_file, save_file

DEFAULT_ALPHA = 16.0


_KOHYA_KEY = re.compile(
    r"^lora_unet_layers_(\d+)_proj_(q|k|v|o|ffn_up_gate|ffn_down)\.lora_(down|up)\.weight$"
)
# text_embedder 适配器只在 lora.include_text_embedder 打开时存在；键名同样是
# kohya 风格，但模块路径是 text_embedder.blocks.<n>.<proj>。
_KOHYA_ALPHA = re.compile(
    r"^lora_unet_layers_(\d+)_proj_(q|k|v|o|ffn_up_gate|ffn_down)\.alpha$"
)
KOHYA_TEXT_ALPHA = re.compile(
    r"^lora_unet_text_embedder_blocks_(\d+)_(qkv|proj|ffn_up|ffn_gate|ffn_down)\.alpha$"
)
_KOHYA_TEXT_KEY = re.compile(
    r"^lora_unet_text_embedder_blocks_(\d+)_(qkv|proj|ffn_up|ffn_gate|ffn_down)\."
    r"lora_(down|up)\.weight$"
)


def _kohya_module(name: str) -> tuple[str, str] | None:
    """``lora_unet_layers_0_proj_ffn_down.lora_down.weight`` -> (module, A|B).

    Underscores cannot be split back into dots reliably, so the layer/projection
    vocabulary of this architecture is matched explicitly.  Both the DiT layers and
    the optional text_embedder blocks are covered; anything else returns ``None``.
    """
    match = _KOHYA_KEY.match(name)
    if match:
        layer, projection, direction = match.groups()
        return f"layers.{layer}.proj.{projection}", ("A" if direction == "down" else "B")
    text = _KOHYA_TEXT_KEY.match(name)
    if text:
        block, projection, direction = text.groups()
        return (f"text_embedder.blocks.{block}.{projection}",
                "A" if direction == "down" else "B")
    return None


def lora_modules(checkpoint: dict) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    parts: dict[str, dict[str, torch.Tensor]] = {}
    for name, tensor in checkpoint.items():
        if name.startswith("lora_unet_"):
            parsed = _kohya_module(name)
            if parsed is None:
                continue
            module, kind = parsed
            parts.setdefault(module, {})[kind] = tensor
            continue
        if ".lora_" not in name:
            continue
        module, _, tail = name.partition(".lora_")
        # PEFT checkpoints prefix with base_model.model., ComfyUI/diffusers LoRA
        # files with diffusion_model.; both name the module the same way after it.
        module = module.replace("base_model.model.", "").replace("diffusion_model.", "", 1)
        kind = tail.split(".")[0]
        # PEFT names the pair A/B, every other layout calls them down/up.
        kind = {"down": "A", "up": "B"}.get(kind, kind)
        parts.setdefault(module, {})[kind] = tensor
    return {module: (pair["A"], pair["B"]) for module, pair in parts.items()
            if "A" in pair and "B" in pair}


def lora_alphas(checkpoint: dict) -> dict[str, float]:
    """Per-module alpha, whatever prefix the file uses.

    A missing alpha means "alpha = rank", which is what the caller assumes, so
    this only reports the ones the file actually carries.
    """
    out: dict[str, float] = {}
    for name, tensor in checkpoint.items():
        if not name.endswith(".alpha"):
            continue
        layer = _KOHYA_ALPHA.match(name)
        if layer:
            index, projection = layer.groups()
            out[f"layers.{index}.proj.{projection}"] = float(tensor)
            continue
        text = KOHYA_TEXT_ALPHA.match(name)
        if text:
            block, projection = text.groups()
            out[f"text_embedder.blocks.{block}.{projection}"] = float(tensor)
            continue
        module = name[:-6].replace("base_model.model.", "").replace("diffusion_model.", "", 1)
        out[module] = float(tensor)
    return out


def kohya_key(module: str) -> str:
    return "lora_unet_" + module.replace(".", "_")


def peft_modules(checkpoint: dict) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    return lora_modules(checkpoint)


def is_kohya(checkpoint: dict) -> bool:
    return any(key.startswith("lora_unet_") and key.endswith(".lora_down.weight") for key in checkpoint)


def to_kohya(checkpoint: dict, alpha: float) -> dict[str, torch.Tensor]:
    """PEFT-named LoRA tensors -> the shippable kohya layout."""
    out = {}
    for module, (a, b) in lora_modules(checkpoint).items():
        key = kohya_key(module)
        out[f"{key}.lora_down.weight"] = a
        out[f"{key}.lora_up.weight"] = b
        out[f"{key}.alpha"] = torch.tensor(float(alpha))
    return out


def convert(checkpoint_path: str, name: str, formats, suffix: str = "",
            out_dir: str = "", alpha: float = DEFAULT_ALPHA, step: int = 0,
            metadata_alpha: float | None = None) -> dict[str, str]:
    checkpoint = load_file(checkpoint_path)
    modules = lora_modules(checkpoint)
    if not modules:
        raise SystemExit(f"{checkpoint_path} has no LoRA tensors")
    alpha = float(metadata_alpha or alpha)
    formats = list(formats or ["kohya"])
    out_dir = out_dir or os.path.dirname(os.path.abspath(checkpoint_path))
    os.makedirs(out_dir, exist_ok=True)
    tag = (lambda fmt: "") if len(formats) == 1 else (lambda fmt: f"_{fmt}")
    written: dict[str, str] = {}
    for fmt in formats:
        tensors = {}
        for module, (a, b) in modules.items():
            if fmt == "kohya":
                key = "lora_unet_" + module.replace(".", "_")
                tensors[f"{key}.lora_down.weight"] = a.contiguous()
                tensors[f"{key}.lora_up.weight"] = b.contiguous()
                tensors[f"{key}.alpha"] = torch.tensor(alpha)
            elif fmt == "diffusers":
                tensors[f"transformer.{module}.lora_A.weight"] = a.contiguous()
                tensors[f"transformer.{module}.lora_B.weight"] = b.contiguous()
            else:
                raise SystemExit(f"unknown format: {fmt}")
        path = os.path.join(out_dir, f"{name}{suffix}{tag(fmt)}.safetensors")
        save_file(tensors, path, metadata={"step": str(step), "alpha": str(alpha)})
        written[fmt] = path
    return written
