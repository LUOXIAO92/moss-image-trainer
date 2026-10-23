"""Block swap: keep the last N transformer blocks in CPU memory while training.

Only useful together with gradient checkpointing, because the weights are needed
twice per step (forward and recompute). A custom checkpoint function swaps the
block in for each pass and out again right after.

Copies run on a dedicated CUDA stream one block ahead: while the compute stream
runs block k, the copy stream uploads block k+1 (block k-1 on the way back down
through the recompute) and then copies block k back into its pinned host
buffers. A copy is a small fraction of a block's compute, so this lookahead
hides essentially all of it. Two blocks may be resident at once, the one being
computed and the one being uploaded.

Every block in the stack has the same tensor shapes, so one released set of GPU
buffers is kept and handed to the next upload. Without that pool each pass
allocates and frees a block's worth of storage 84 times per step, and under the
caching allocator those transients never coalesce: the reserved footprint climbs
step after step and the run ends up holding the memory that streaming was meant
to give back.
"""

from __future__ import annotations

import torch

# Two resident slots: the block the compute stream is running and the block the
# copy stream is uploading. A third slot would only hold weights no kernel reads
# yet, and every slot costs a full block of VRAM, so this is fixed rather than a
# user-facing setting.
WINDOW = 2

# Released GPU buffer sets kept for the next upload. One is enough: during a walk
# two sets are live (one computing, one uploading) and the third is the one just
# released, which the next prefetch takes. Every kept set costs a block of VRAM,
# so this stays at one.
POOL_SETS = 1


class BlockSwap:
    def __init__(self, blocks, count: int, device="cuda"):
        self.blocks = blocks
        self.start = max(0, len(blocks) - int(count))
        self.device = device
        self._device = torch.device(device)
        self.copy_stream = torch.cuda.Stream(device=self._device)
        self.params: dict[int, list[torch.nn.Parameter]] = {}
        for index in range(self.start, len(blocks)):
            params = [p for p in blocks[index].parameters() if p.requires_grad is False]
            for param in params:
                # Pageable host copy for now: pinning every offloaded block up
                # front would take the whole host-side footprint before the first
                # block is ever streamed.
                param.data = param.data.to("cpu")
            self.params[index] = params
        # Pinned per-block staging, allocated on first use and reused afterwards.
        self.host: dict[int, list[torch.Tensor]] = {}
        # CUDA tensors currently bound to param.data, keyed by block.
        self.resident: dict[int, list[torch.Tensor]] = {}
        self.upload_done: dict[int, torch.cuda.Event] = {}
        self.download_done: dict[int, torch.cuda.Event] = {}
        # shape/dtype signature -> sets of GPU buffers released by earlier blocks
        self.pool: dict[tuple, list[list[torch.Tensor]]] = {}
        # a Function whose inputs never require grad produces outputs without a
        # grad_fn; this dummy keeps the graph alive (torch.utils.checkpoint does the same)
        self._dummy = torch.zeros(1, device=self._device, requires_grad=True)

    def is_swapped(self, index: int) -> bool:
        return index in self.params

    def _compute_stream(self) -> torch.cuda.Stream:
        return torch.cuda.current_stream(self._device)

    def _staging(self, index: int) -> list[torch.Tensor]:
        """Pinned host buffers for one block, allocated the first time it streams."""
        host = self.host.get(index)
        if host is None:
            host = [param.data.pin_memory() for param in self.params[index]]
            for param, buffer in zip(self.params[index], host):
                param.data = buffer
            self.host[index] = host
        return host

    @staticmethod
    def _key(host: list[torch.Tensor]) -> tuple:
        return tuple((tuple(buffer.shape), buffer.dtype) for buffer in host)

    def _borrow(self, host: list[torch.Tensor]) -> list[torch.Tensor] | None:
        """A released buffer set with the same shapes, or None to allocate one."""
        key = self._key(host)
        bucket = self.pool.get(key)
        return bucket.pop() if bucket else None

    def _give_back(self, host: list[torch.Tensor], tensors: list[torch.Tensor]) -> None:
        bucket = self.pool.setdefault(self._key(host), [])
        if len(bucket) < POOL_SETS:
            bucket.append(tensors)

    def _upload(self, index: int) -> None:
        host = self._staging(index)
        compute = self._compute_stream()
        tensors = self._borrow(host)
        if tensors is None:
            tensors = [torch.empty_like(buffer, device=self._device) for buffer in host]
        previous = self.download_done.get(index)
        # The allocator's promise is stream-ordered, and these tensors are written
        # by the copy stream. Their storage may be recycled from a tensor the
        # compute stream is still reading or producing, so the H2D has to wait for
        # the compute work enqueued so far. The lookahead keeps its block of slack:
        # this point is before the current block's kernels, so the copy still runs
        # during that block.
        self.copy_stream.wait_stream(compute)
        with torch.cuda.stream(self.copy_stream):
            if previous is not None:
                # The staging buffers still belong to the block's previous
                # download; reading them now would race that copy out.
                self.copy_stream.wait_event(previous)
            for tensor, buffer in zip(tensors, host):
                tensor.copy_(buffer, non_blocking=True)
            uploaded = torch.cuda.Event()
            uploaded.record()
        # The set is written by the copy stream and read by the compute stream.
        # Recording the copy stream keeps a set that gets evicted from the pool
        # from being recycled before that write lands; the compute stream is the
        # allocation stream, so it needs no record.
        for tensor in tensors:
            tensor.record_stream(self.copy_stream)
        self.upload_done[index] = uploaded
        self.resident[index] = tensors
        for param, tensor in zip(self.params[index], tensors):
            param.data = tensor

    def wait(self, index: int) -> None:
        """Make the compute stream wait for the block's upload before it runs."""
        if index not in self.resident:
            return
        uploaded = self.upload_done.get(index)
        if uploaded is not None:
            self._compute_stream().wait_event(uploaded)

    def prefetch(self, index: int) -> None:
        """Upload one block ahead without waiting for it.

        No-op when the block is not swapped, is already resident, or when both
        slots are taken: at that point a block has to be released before another
        one can be streamed in.
        """
        if index not in self.params or index in self.resident or len(self.resident) >= WINDOW:
            return
        self._upload(index)

    def load(self, index: int) -> None:
        if index not in self.params or index in self.resident:
            return
        if len(self.resident) >= WINDOW:
            # The lookahead normally keeps a slot free, so this only runs when a
            # walk skipped a block and left it prefetched but never released.
            # Evict the resident block farthest from this one; the neighbours are
            # the ones the walk is about to use.
            self.release(max(self.resident, key=lambda other: abs(other - index)))
        self._upload(index)

    def release(self, index: int) -> None:
        tensors = self.resident.pop(index, None)
        if tensors is None:
            return
        host = self._staging(index)
        compute = self._compute_stream()
        # The block's kernels are still queued on the compute stream; the copy
        # stream must not read the weights before they finish.
        self.copy_stream.wait_stream(compute)
        with torch.cuda.stream(self.copy_stream):
            for buffer, tensor in zip(host, tensors):
                buffer.copy_(tensor, non_blocking=True)
            downloaded = torch.cuda.Event()
            downloaded.record()
        # The host drops these references now. record_stream keeps the caching
        # allocator from handing the same memory to the next upload while this
        # copy is still queued on the side stream.
        for tensor in tensors:
            tensor.record_stream(self.copy_stream)
        self.download_done[index] = downloaded
        self._give_back(host, tensors)
        for param, buffer in zip(self.params[index], host):
            param.data = buffer

    def __call__(self, block, index, x, rope, mod, *extra):
        # ``extra`` carries whatever the block's forward takes beyond the common
        # three (mask, cu_seqlens, max_seqlens); dropping them silently changed
        # attention for every non-packed path.
        return _SwapCheckpoint.apply(x, block, rope, mod, self, index, self._dummy, *extra)


class _SwapCheckpoint(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, block, rope, mod, swap, index, dummy, *extra):
        ctx.save_for_backward(x, rope, mod)
        ctx.block, ctx.swap, ctx.index = block, swap, index
        ctx.extra = extra
        swap.load(index)
        swap.wait(index)
        # The forward walk is ascending, so the next block it needs is index + 1.
        swap.prefetch(index + 1)
        with torch.no_grad():
            out = block(x, rope, mod, *extra)
        swap.release(index)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        x, rope, mod = ctx.saved_tensors
        block, swap, index = ctx.block, ctx.swap, ctx.index
        swap.load(index)
        swap.wait(index)
        # The recompute walks the blocks downwards, so the next one is index - 1.
        swap.prefetch(index - 1)
        with torch.enable_grad():
            x = x.detach().requires_grad_(True)
            out = block(x, rope, mod, *ctx.extra)
            out.backward(grad_out)
        swap.release(index)
        # one None per non-tensor input: block, rope, mod, swap, index, dummy, extras
        return (x.grad if ctx.needs_input_grad[0] else None,) + (None,) * (6 + len(ctx.extra))
