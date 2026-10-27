"""Activation offload: the boundaries checkpointing keeps move to host memory.

    .venv/bin/python tests/test_activation_offload.py

The trainer wraps one forward pass in ``torch.autograd.graph.save_on_cpu``.
Both checkpoint paths -- ``torch.utils.checkpoint`` for resident blocks and the
streaming checkpoint in ``offload.py`` -- save their block input through
autograd, so the hooks in that context see exactly those tensors: 42 boundaries
of tokens x hidden x 2 bytes, about 1 GiB at 1024x1024. This file locks down the
three things that are easy to break silently: the option needs checkpointing,
compile is refused while the hooks are active, and the offloaded run computes the
same gradients as the run that keeps the boundaries on the device.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402
from torch import nn  # noqa: E402
from torch.autograd.graph import saved_tensors_hooks  # noqa: E402
from torch.utils.checkpoint import checkpoint  # noqa: E402

from moss_lora.config import Config  # noqa: E402
from moss_lora.trainer import Trainer  # noqa: E402


def check_default_and_validation():
    config = Config()
    assert config.vram.offload_activations is False
    assert "err.config.offload_checkpointing" not in {
        problem.code for problem in config.validate()}

    config.vram.offload_activations = True
    assert "err.config.offload_checkpointing" not in {
        problem.code for problem in config.validate()}, "checkpointing is on by default"

    config.vram.gradient_checkpointing = False
    codes = {problem.code for problem in config.validate()}
    assert "err.config.offload_checkpointing" in codes, codes

    config.vram.gradient_checkpointing = True
    assert "err.config.offload_checkpointing" not in {
        problem.code for problem in config.validate()}


def check_presets_own_the_flag():
    from moss_lora.config import PRESET_TIERS, apply_preset

    config = Config()
    config.vram.offload_activations = True
    for tier in PRESET_TIERS:
        apply_preset(config, tier)
        assert config.vram.offload_activations is False, tier


def check_compile_is_refused():
    trainer = object.__new__(Trainer)
    trainer.config = SimpleNamespace(vram=SimpleNamespace(compile=True, blocks_to_swap=0))
    trainer.offload_activations = True
    trainer.device = "cuda"
    logs: list[str] = []
    trainer.log = logs.append
    original = importlib.util.find_spec
    # Triton exists on the machines that train with compile; pretend it does here
    # so the refusal is the offload rule and not the missing dependency.
    importlib.util.find_spec = lambda name: object()  # type: ignore[assignment]
    try:
        trainer._maybe_compile()
    finally:
        importlib.util.find_spec = original
    assert any("offload" in line for line in logs), logs


def check_hooks_carry_the_boundary():
    """A checkpointed block saves its input, and the hooks turn that into a copy."""
    layer = nn.Linear(32, 32)
    x = torch.randn(6, 32, requires_grad=True)
    packed: list[tuple[str, torch.dtype]] = []

    def pack(tensor):
        packed.append((tensor.device.type, tensor.dtype))
        return tensor.cpu() if tensor.is_cuda else tensor.clone()

    def unpack(tensor):
        return tensor

    with saved_tensors_hooks(pack, unpack):
        out = checkpoint(layer, x, use_reentrant=False)
    out.sum().backward()
    assert packed, "checkpoint saved no input, so there is nothing to offload"
    assert packed[0][1] == x.dtype, packed
    assert x.grad is not None and layer.weight.grad is not None


def check_gradients_match():
    """Offloading changes where a boundary lives, never what it holds."""
    torch.manual_seed(0)
    layer = nn.Linear(32, 32)
    weight = layer.weight.detach().clone()
    bias = layer.bias.detach().clone()
    x = torch.randn(6, 32)
    target = torch.randn(6, 32)

    def grads(offload: bool):
        layer.weight.data.copy_(weight)
        layer.bias.data.copy_(bias)
        layer.zero_grad(set_to_none=True)
        inputs = x.clone().requires_grad_(True)
        if offload:
            with saved_tensors_hooks(lambda t: t.cpu() if t.is_cuda else t.clone(),
                                     lambda t: t):
                out = checkpoint(layer, inputs, use_reentrant=False)
        else:
            out = checkpoint(layer, inputs, use_reentrant=False)
        nn.functional.mse_loss(out, target).backward()
        return layer.weight.grad.detach().clone(), layer.bias.grad.detach().clone()

    kept_w, kept_b = grads(False)
    moved_w, moved_b = grads(True)
    assert torch.equal(kept_w, moved_w), "weight gradient changed under offload"
    assert torch.equal(kept_b, moved_b), "bias gradient changed under offload"


def main() -> int:
    check_default_and_validation()
    check_presets_own_the_flag()
    check_compile_is_refused()
    check_hooks_carry_the_boundary()
    check_gradients_match()
    print("activation offload checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
