"""NF4 weight-only quantization for QLoRA-style training.

Three levels, by memory saved:

    none  every Linear stays bf16                     9.5 GiB of block weights
    ffn   only ffn_up_gate / ffn_down become nf4      4.5 GiB  (saves ~5.0)
    all   every frozen Linear becomes nf4             2.5 GiB  (saves ~7.0)

LoRA adapters are never quantized. The base weights are frozen, so bitsandbytes'
4-bit linears only have to support the input gradient -- which they do.
"""

from __future__ import annotations

import torch
from torch import nn

SCOPES = ("none", "ffn", "all")
FP8_MAX = 448.0


class FP8Linear(nn.Module):
    """Frozen linear stored as fp8 e4m3 with per-row scales (inference only)."""

    def __init__(self, linear: nn.Linear):
        super().__init__()
        weight = linear.weight.detach().float()
        scale = weight.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / FP8_MAX
        self.register_buffer("weight", (weight / scale).to(torch.float8_e4m3fn), persistent=False)
        self.register_buffer("scale", scale.to(torch.bfloat16), persistent=False)
        self.bias = linear.bias
        self.in_features = linear.in_features
        self.out_features = linear.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype if x.dtype in (torch.bfloat16, torch.float16, torch.float32) else torch.bfloat16
        weight = self.weight.to(dtype) * self.scale.to(dtype)
        bias = self.bias.to(dtype) if self.bias is not None else None
        return torch.nn.functional.linear(x.to(dtype), weight, bias)
FFN_MARKERS = ("ffn_up_gate", "ffn_down")


def _to_linear4bit(linear: nn.Linear, dtype: torch.dtype = torch.bfloat16) -> nn.Module:
    import bitsandbytes as bnb

    quantized = bnb.nn.Linear4bit(
        linear.in_features,
        linear.out_features,
        bias=linear.bias is not None,
        compute_dtype=dtype,
        compress_statistics=True,
        quant_type="nf4",
    )
    quantized.weight = bnb.nn.Params4bit(
        linear.weight.data.detach().to("cpu", dtype),
        requires_grad=False,
        quant_type="nf4",
        compress_statistics=True,
    )
    quantized = quantized.to(linear.weight.device)
    if linear.bias is not None:
        quantized.bias = nn.Parameter(linear.bias.data.to(dtype), requires_grad=False)
    return quantized


def quantize_nf4(
    model: nn.Module,
    scope: str = "all",
    dtype: torch.dtype = torch.bfloat16,
    skip: tuple[str, ...] = ("lora_",),
    only: tuple[str, ...] | None = None,
) -> int:
    """Replace frozen Linears with nf4 ones. Returns how many were replaced."""
    if scope not in ("ffn", "all"):
        return 0
    targets = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear) or any(token in name for token in skip):
            continue
        module.weight.requires_grad = False  # base weights are frozen either way
        if only is not None and not any(marker in name for marker in only):
            continue
        if only is None and not name.startswith("layers."):
            continue  # skip the small non-block modules (326M params total)
        if scope == "ffn" and not any(marker in name for marker in FFN_MARKERS):
            continue
        targets.append((name, module))
    for name, module in targets:
        parent_name, _, leaf = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, leaf, _to_linear4bit(module, dtype))
    return len(targets)
