"""Block swap: keep the last N transformer blocks in CPU memory while training.

Only useful together with gradient checkpointing, because the weights are needed
twice per step (forward and recompute). A custom checkpoint function swaps the
block in for each pass and out again right after, so at most one swapped block is
resident on the GPU at a time.
"""

from __future__ import annotations

import torch


class BlockSwap:
    def __init__(self, blocks, count: int, device="cuda"):
        self.blocks = blocks
        self.start = max(0, len(blocks) - int(count))
        self.device = device
        self.params: dict[int, list[torch.nn.Parameter]] = {}
        for index in range(self.start, len(blocks)):
            params = [p for p in blocks[index].parameters() if p.requires_grad is False]
            for param in params:
                param.data = param.data.to("cpu").pin_memory()
            self.params[index] = params
        self.loaded: set[int] = set()
        # a Function whose inputs never require grad produces outputs without a
        # grad_fn; this dummy keeps the graph alive (torch.utils.checkpoint does the same)
        self._dummy = torch.zeros(1, device=device, requires_grad=True)

    def is_swapped(self, index: int) -> bool:
        return index in self.params

    def load(self, index: int):
        if index not in self.params or index in self.loaded:
            return
        for param in self.params[index]:
            param.data = param.data.to(self.device, non_blocking=True)
        self.loaded.add(index)

    def release(self, index: int):
        if index not in self.params or index not in self.loaded:
            return
        for param in self.params[index]:
            param.data = param.data.to("cpu", non_blocking=True)
        self.loaded.discard(index)

    def __call__(self, block, index, x, rope, mod):
        return _SwapCheckpoint.apply(x, block, rope, mod, self, index, self._dummy)


class _SwapCheckpoint(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, block, rope, mod, swap, index, dummy):
        ctx.save_for_backward(x, rope, mod)
        ctx.block, ctx.swap, ctx.index = block, swap, index
        swap.load(index)
        with torch.no_grad():
            out = block(x, rope, mod)
        swap.release(index)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        x, rope, mod = ctx.saved_tensors
        block, swap, index = ctx.block, ctx.swap, ctx.index
        swap.load(index)
        with torch.enable_grad():
            x = x.detach().requires_grad_(True)
            out = block(x, rope, mod)
            out.backward(grad_out)
        swap.release(index)
        return (x.grad if ctx.needs_input_grad[0] else None), None, None, None, None, None, None
