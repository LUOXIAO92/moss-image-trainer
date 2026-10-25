"""Previews must not go through torch.compile.

    .venv/bin/python tests/test_preview_compile.py

A sampling pass runs a batch and a shape the training loop never compiled, so the
compiled forward would spend minutes building a graph for it the first time --
and minutes of silence in the log reads as a hang. `Trainer._preview_forward`
hands the model its uncompiled forward for the duration of the sampling call and
puts the compiled wrapper back afterwards.

The check builds that arrangement directly: an `nn.Module` whose instance
`forward` is a fake compiled wrapper, exactly how `_maybe_compile` installs it.
No CUDA and no 5B model are involved.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402
from torch import nn  # noqa: E402

from moss_lora.trainer import Trainer  # noqa: E402


class Tiny(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 2)

    def forward(self, x):  # pragma: no cover - replaced in every case below
        return self.linear(x)


def check_compile_skips_without_triton() -> None:
    """A CUDA compile request without Triton must not install the wrapper."""
    net = Tiny()
    eager = net.forward
    logs: list[str] = []
    holder = types.SimpleNamespace(
        config=types.SimpleNamespace(
            vram=types.SimpleNamespace(compile=True, blocks_to_swap=0)
        ),
        device="cuda",
        model=net,
        log=logs.append,
        _eager_forward=None,
    )
    real_find_spec = importlib.util.find_spec
    importlib.util.find_spec = lambda name: None if name == "triton" else real_find_spec(name)
    try:
        Trainer._maybe_compile(holder)
    finally:
        importlib.util.find_spec = real_find_spec
    assert net.forward == eager, "the compiled wrapper was installed without Triton"
    assert logs == ["compile: skipped, Triton is not installed"], logs


def main() -> int:
    check_compile_skips_without_triton()
    net = Tiny()
    eager = net.forward
    calls = {"compiled": 0}

    def compiled(x):
        calls["compiled"] += 1
        return eager(x)

    # What _maybe_compile leaves behind: the compiled wrapper on the instance.
    net.forward = compiled
    holder = types.SimpleNamespace(model=net, _eager_forward=eager)
    x = torch.zeros(1, 2)

    before = net(x)
    assert calls["compiled"] == 1, calls

    with Trainer._preview_forward(holder):
        assert net.forward is eager, "the compiled wrapper is still installed during a preview"
        inside = net(x)
        assert calls["compiled"] == 1, f"the preview went through torch.compile: {calls}"
    assert net.forward is compiled, "the compiled forward was not handed back"
    assert torch.equal(before, inside), "eager and compiled output differ"

    # A run without compile: nothing to swap, and the call still works.
    plain = Tiny()
    plain_forward = plain.forward
    holder = types.SimpleNamespace(model=plain, _eager_forward=None)
    # A class method is bound fresh on every access, so identity does not hold here.
    with Trainer._preview_forward(holder):
        assert plain.forward == plain_forward
    assert plain.forward == plain_forward

    # Exceptions must not leave the model on the eager path.
    holder = types.SimpleNamespace(model=net, _eager_forward=eager)
    try:
        with Trainer._preview_forward(holder):
            raise RuntimeError("sampler blew up")
    except RuntimeError:
        pass
    assert net.forward is compiled, "the compiled forward stayed swapped out after an exception"

    print("preview compile checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
