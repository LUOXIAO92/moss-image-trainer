"""ConvRot int8 text-encoder weights, with no ComfyUI dependency.

The Qwen3.5-9B text encoder ships in two shapes: a plain HuggingFace directory, or a
single int8 ``.safetensors`` file whose linear weights were rotated with a regular
Hadamard matrix before quantization.  That rotation is ConvRot, from the QuaRot
family of rotation-based int8 quantization.  ConvRot is a numerical scheme, not a
vendor feature: ComfyUI loads files packaged this way, and so does the official
loader.

    H4[a, b] = -1 iff a ^ b == 3
    H       = kron(H4, ...) / sqrt(size)
    W_rot   = W @ blockdiag(H).T
    q, s    = int8 quantize(W_rot)      # per-row scale

so ``W = (q * s) @ blockdiag(H)``.  This module reproduces the dequantization exactly
(bit-identical to ``torch.ops.comfy_kitchen.dequantize_int8_convrot_weight_dtype``).

The container keeps two conventions inherited from that packaging: a
``.weight_scale`` tensor next to each ``.weight``, and a ``comfy_quant`` JSON marker
in the tensor metadata.  Both are file-format details; the quantization is ConvRot.

Plain "packed" checkpoints (a normal HF state dict in one file) and files whose keys
follow ``model.language_model.*`` are accepted too.
"""

from __future__ import annotations

import json

import torch

HADAMARD_CACHE: dict[int, torch.Tensor] = {}


def regular_hadamard(size: int) -> torch.Tensor:
    cached = HADAMARD_CACHE.get(size)
    if cached is not None:
        return cached
    h4 = torch.tensor([[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]], dtype=torch.float32)
    h = h4
    current = 4
    while current < size:
        h = torch.kron(h, h4)
        current *= 4
    h = h / size**0.5
    HADAMARD_CACHE[size] = h
    return h


def quant_marker(metadata_bytes: torch.Tensor | None) -> dict:
    if metadata_bytes is None:
        return {}
    try:
        raw = metadata_bytes.numpy().tobytes().decode("utf-8", "replace").rstrip("\x00")
        return json.loads(raw)
    except (ValueError, TypeError):
        return {}


def is_quantized(keys) -> bool:
    return any(key.endswith(".weight_scale") for key in keys)


def dequantize(int8_weight: torch.Tensor, scale: torch.Tensor, group: int, marker: dict) -> torch.Tensor:
    """int8 + per-row scale -> float32, undoing the ConvRot rotation."""
    weight = int8_weight.to(torch.float32) * scale.to(torch.float32)
    if not marker.get("convrot"):
        return weight
    group = int(marker.get("convrot_groupsize") or group)
    in_features = weight.shape[-1]
    if in_features % group:
        raise ValueError(f"in_features {in_features} not divisible by convrot group {group}")
    h = regular_hadamard(group).to(device=weight.device, dtype=weight.dtype)
    grouped = weight.reshape(weight.shape[0], in_features // group, group)
    return (grouped @ h).reshape(weight.shape)


_SKIP = ("visual.", "vision.", "mtp.", "lm_head.", "audio.", "projector.")
_BARE = ("embed_tokens.", "layers.", "norm.", "rotary_emb.")


def normalize_key(key: str) -> str | None:
    """Map a ComfyUI / packed key onto the HF text-tower name, or None to skip.

    Handles the three layouts seen in the wild: ``model.language_model.*``
    (HF full checkpoint and the ComfyUI repack), ``language_model.*`` /
    ``text_model.*`` (text-only exports), and bare ``layers.*`` keys (a packed
    text tower with the prefix already stripped).
    """
    for prefix in ("model.language_model.", "language_model.", "text_model.", "model."):
        if key.startswith(prefix):
            rest = key[len(prefix):]
            return None if rest.startswith(_SKIP) else rest
    if key.startswith(_BARE):
        return key
    return None
