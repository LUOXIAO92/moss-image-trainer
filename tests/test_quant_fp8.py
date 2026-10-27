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


def check_peft_attaches():
    """An fp8-replaced projection must still accept a LoRA adapter.

    PEFT decides with ``isinstance(target, torch.nn.Linear)``, and every scope
    fp8 replaces is also a LoRA target. When FP8Linear was a plain nn.Module the
    adapter injection raised, so ``quantize_dtype: fp8`` could not train at all.
    """
    from peft import LoraConfig, get_peft_model

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Module()
            self.proj.q = nn.Linear(16, 16)
            self.proj.ffn_down = nn.Linear(32, 16)

    class Toy(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([Block(), Block()])

    model = Toy()
    assert quant.quantize_fp8(model, "all") == 4
    config = LoraConfig(r=8, lora_alpha=8,
                        target_modules=r"layers\.\d+\.proj\.(q|ffn_down)$")
    model = get_peft_model(model, config)
    x = torch.randn(2, 16, requires_grad=True)
    model.layers[0].proj.q(x).sum().backward()
    assert x.grad is not None, "the frozen fp8 weight stopped the input gradient"
    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    # 2 blocks x 2 targeted projections x (lora_A, lora_B)
    assert len(trainable) == 8, trainable
    base = model.base_model.model.layers[0].proj.q.base_layer
    assert isinstance(base, quant.FP8Linear), type(base)
    print("peft adapter attaches to fp8 Linears")


def main() -> int:
    check_buffer_and_precision()
    check_backward_to_input()
    check_scope_rules()
    check_unknown_scope_is_a_noop()
    check_peft_attaches()
    print("fp8 quantization checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
