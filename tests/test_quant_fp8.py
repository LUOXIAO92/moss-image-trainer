"""fp8 e4m3 weight storage must keep the forward and the input gradient.

    .venv/bin/python tests/test_quant_fp8.py

The point of the format is the base weights stay frozen and only the adapter
learns, so a replaced Linear has to do exactly two things: approximate the same
output, and pass a gradient back to its input.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch import nn

from moss_lora import quant


def check_buffer_and_precision():
    torch.manual_seed(0)
    linear = nn.Linear(64, 32, bias=True, dtype=torch.bfloat16)
    wrapped = quant.FP8Linear(linear)

    assert wrapped.weight.dtype == torch.float8_e4m3fn, wrapped.weight.dtype
    stored = wrapped.weight.numel() * wrapped.weight.element_size()
    original = linear.weight.numel() * linear.weight.element_size()
    assert stored * 2 == original, (stored, original)

    x = torch.randn(4, 64, dtype=torch.bfloat16)
    with torch.no_grad():
        expected = linear(x).float()
        got = wrapped(x).float()
    relative = (got - expected).norm() / expected.norm()
    assert relative < 0.03, f"fp8 forward drifted {relative:.4f}"
    print(f"forward relative error: {relative:.4f}")


def check_backward_to_input():
    torch.manual_seed(1)
    linear = nn.Linear(64, 32, bias=False, dtype=torch.bfloat16)
    wrapped = quant.FP8Linear(linear)
    x = torch.randn(4, 64, dtype=torch.bfloat16, requires_grad=True)
    wrapped(x).sum().backward()
    assert x.grad is not None and x.grad.shape == x.shape, x.grad
    assert all(not p.requires_grad for p in wrapped.parameters()), \
        "frozen weights must not collect gradients"
    print("input gradient ok")


def check_scope_rules():
    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.q = nn.Linear(8, 8)
            self.ffn_up_gate = nn.Linear(8, 16)
            self.ffn_down = nn.Linear(16, 8)

    class Toy(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([Block(), Block()])
            self.text_embedder = nn.Linear(8, 8)

    model = Toy()
    assert quant.quantize_fp8(model, "ffn") == 4, "ffn scope replaces 2 per block"
    assert isinstance(model.layers[0].ffn_up_gate, quant.FP8Linear)
    assert isinstance(model.layers[0].q, nn.Linear), "q is outside the ffn scope"

    model = Toy()
    assert quant.quantize_fp8(model, "all") == 6, "all replaces every block Linear"
    assert isinstance(model.layers[1].q, quant.FP8Linear)
    assert isinstance(model.text_embedder, nn.Linear), \
        "conditioning modules stay bf16 unless explicitly asked for"
    print("scope rules ok")


def check_unknown_scope_is_a_noop():
    model = nn.Sequential(nn.Linear(4, 4))
    assert quant.quantize_fp8(model, "none") == 0
    assert quant.quantize_fp8(model, "everything") == 0
    print("unknown scope ok")


def main() -> int:
    check_buffer_and_precision()
    check_backward_to_input()
    check_scope_rules()
    check_unknown_scope_is_a_noop()
    print("fp8 quantization checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
