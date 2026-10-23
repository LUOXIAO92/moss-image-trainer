"""Linear layers that keep ConvRot int8 weights on the device.

The text encoder ships as int8 (6.44 GiB for its 200 linear layers). Turning
those into bf16 before inference doubles that to 12.9 GiB and buys nothing:
dequantisation is cheap enough to pay per forward pass. Measured on 725
captions, keeping int8 resident costs 1.2s over pre-dequantised weights.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from . import convrot

GROUP = 256


class QuantLinear(nn.Module):
    """``nn.Linear`` whose weight stays int8 and dequantises inside ``forward``."""

    def __init__(self, int8_weight: torch.Tensor, scale: torch.Tensor, marker: dict | None = None,
                 group: int = GROUP, bias: torch.Tensor | None = None,
                 dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.register_buffer("int8_weight", int8_weight, persistent=False)
        self.register_buffer("scale", scale, persistent=False)
        self.marker = dict(marker or {})
        self.group = int(group)
        self.weight_dtype = dtype
        if bias is None:
            self.register_buffer("bias", None, persistent=False)
        else:
            self.register_buffer("bias", bias, persistent=False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        weight = convrot.dequantize(self.int8_weight, self.scale, self.group, self.marker)
        # bf16, like the parameter this replaces. F.linear refuses mixed dtypes,
        # so matching the model dtype keeps the arithmetic identical to a
        # pre-dequantised layer instead of silently switching to float32.
        return F.linear(hidden, weight.to(self.weight_dtype), self.bias)

    def extra_repr(self) -> str:
        shape = tuple(self.int8_weight.shape)
        return f"in={shape[1]}, out={shape[0]}, int8"
