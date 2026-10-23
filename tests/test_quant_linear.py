"""QuantLinear 的数值与显存契约。

量化层只在设备上留 int8 + 每行 scale，前向时才反量化；这条路径必须和"加载时
一次性反量化成 bf16"给出完全相同的结果，否则缓存里的文本嵌入就无法复现。

    .venv/bin/python tests/test_quant_linear.py
"""

from __future__ import annotations

import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from moss_lora import convrot  # noqa: E402
from moss_lora.quant_linear import QuantLinear  # noqa: E402


def _weight(out_features: int = 64, in_features: int = 512, group: int = 256, seed: int = 0):
    generator = torch.Generator().manual_seed(seed)
    int8 = torch.randint(-127, 127, (out_features, in_features), generator=generator, dtype=torch.int8)
    scale = torch.rand(out_features, 1, generator=generator) / 127.0
    marker = {"convrot": True, "convrot_groupsize": group}
    return int8, scale, marker


def check_matches_static_dequant():
    int8, scale, marker = _weight()
    layer = QuantLinear(int8, scale, marker)
    hidden = torch.randn(3, 512, dtype=torch.bfloat16)

    live = layer(hidden)
    static_weight = convrot.dequantize(int8, scale, 256, marker).to(torch.bfloat16)
    static = F.linear(hidden, static_weight)
    assert torch.equal(live, static), (live - static).abs().max().item()


def check_keeps_int8_resident():
    int8, scale, marker = _weight()
    layer = QuantLinear(int8, scale, marker)
    assert layer.int8_weight.dtype == torch.int8
    assert layer.int8_weight.numel() * layer.int8_weight.element_size() == int8.numel(), \
        "权重必须按 int8 常驻，不能在构造时展开成浮点"
    tensorlike = [t for t in list(layer.parameters()) + list(layer.buffers()) if t is not None]
    assert all(t.dtype in (torch.int8, torch.float32) for t in tensorlike), [t.dtype for t in tensorlike]


def check_bias_is_carried():
    int8, scale, marker = _weight()
    bias = torch.randn(64, dtype=torch.bfloat16)
    layer = QuantLinear(int8, scale, marker, bias=bias)
    hidden = torch.randn(2, 512, dtype=torch.bfloat16)
    static_weight = convrot.dequantize(int8, scale, 256, marker).to(torch.bfloat16)
    assert torch.equal(layer(hidden), F.linear(hidden, static_weight, bias))


def main():
    check_matches_static_dequant()
    check_keeps_int8_resident()
    check_bias_is_carried()
    print("quant linear checks passed")


if __name__ == "__main__":
    main()
