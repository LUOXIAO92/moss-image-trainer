"""Streaming block swap must hide the copy without changing a single value.

The copies run on their own CUDA stream, so a host-side reference dropped while
a copy is still queued is a correctness bug: the caching allocator can hand the
same storage to the next block and the recompute then uses the wrong weights.
This walks a four-block model forward and backward with no swap, with all four
blocks streamed without lookahead, and with the window-two lookahead, and
compares the output and the gradient that reaches the input latents. It also
covers the preview path, which streams a block around a single forward.

    .venv/bin/python tests/test_offload_stream.py
"""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from moss_lora.model import MOSSImageTransformer  # noqa: E402
from moss_lora.offload import BlockSwap  # noqa: E402

BLOCKS = 4
DIM = 64
FFN_DIM = 128
HEADS = 8
TEXT_DIM = 32
FREQ_DIM = 8
CHANNELS = 16
TEXT_LEN = 5
STEPS = 3
TOLERANCE = 1e-5


def build_model():
    torch.manual_seed(0)
    model = MOSSImageTransformer(
        dim=DIM,
        ffn_dim=FFN_DIM,
        num_heads=HEADS,
        num_layers=BLOCKS,
        text_dim=TEXT_DIM,
        freq_dim=FREQ_DIM,
        latent_channels=CHANNELS,
        device="cuda",
        dtype=torch.float32,
    )
    model.train()
    model.gradient_checkpointing = True
    model.attention_backend = "sdpa"  # no FlashAttention dependency in a unit test
    # The trainer streams frozen base weights under a LoRA; freeze them here too,
    # otherwise BlockSwap finds no parameter it is allowed to move.
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def inputs():
    generator = torch.Generator().manual_seed(1)
    latent = torch.randn(1, CHANNELS, 2, 2, generator=generator)
    context = torch.randn(1, TEXT_LEN, TEXT_DIM, generator=generator).to("cuda")
    return latent, context


def forward_backward(model, latent, context):
    x = latent.to("cuda").requires_grad_(True)
    outputs = model.forward_packed(
        [x], torch.tensor(500.0, device="cuda"), [context], checkpoint_blocks=True
    )
    loss = sum(output.float().square().sum() for output in outputs)
    loss.backward()
    torch.cuda.synchronize()
    return [output.detach().clone() for output in outputs], x.grad.clone()


def preview(model, latent, context):
    with torch.no_grad():
        outputs = model.forward_packed([latent.to("cuda")], torch.tensor(500.0, device="cuda"),
                                       [context])
    torch.cuda.synchronize()
    return [output.detach().clone() for output in outputs]


def assert_close(want, got, label):
    assert len(want) == len(got), (label, len(want), len(got))
    for index, (expected, actual) in enumerate(zip(want, got)):
        diff = (expected.float() - actual.float()).abs().max().item()
        assert torch.allclose(expected.float(), actual.float(), atol=TOLERANCE, rtol=0), (
            f"{label}: tensor {index} differs by {diff} (tolerance {TOLERANCE})"
        )


def collect(model, latent, context):
    """Forward+backward STEPS times, then two previews; repeated passes reuse buffers."""
    outputs, grads = [], []
    for _ in range(STEPS):
        output, grad = forward_backward(model, latent, context)
        outputs.append(output)
        grads.append(grad)
    model.eval()
    previews = [preview(model, latent, context), preview(model, latent, context)]
    model.train()
    return outputs, grads, previews


def check_streaming(model, latent, context):
    resident = collect(model, latent, context)

    swap = BlockSwap(model.layers, BLOCKS, device="cuda")
    model.block_swap = swap
    try:
        streamed = collect(model, latent, context)
        for step, (want, got) in enumerate(zip(resident[0], streamed[0])):
            assert_close(want, got, f"lookahead output, step {step}")
        for step, (want, got) in enumerate(zip(resident[1], streamed[1])):
            assert_close([want], [got], f"lookahead input gradient, step {step}")
        for call, (want, got) in enumerate(zip(resident[2], streamed[2])):
            assert_close(want, got, f"preview output, call {call}")
        print(f"window-2 lookahead          : ok  {STEPS} forward+backward steps and "
              f"{len(resident[2])} previews match resident weights")

        # Same copies, no lookahead: load/wait/run/release with nothing uploaded
        # during the compute. This isolates the pipeline from the copy path.
        swap.prefetch = lambda index: None
        try:
            eager = collect(model, latent, context)
        finally:
            del swap.prefetch
        for step, (want, got) in enumerate(zip(resident[0], eager[0])):
            assert_close(want, got, f"no-lookahead output, step {step}")
        for step, (want, got) in enumerate(zip(resident[1], eager[1])):
            assert_close([want], [got], f"no-lookahead input gradient, step {step}")
        print(f"streamed, no lookahead      : ok  {STEPS} forward+backward steps match "
              f"resident weights")

        check_window(swap)
        print("window cap                  : ok  two blocks resident, prefetch is a no-op there")
        check_pool(swap)
        print("buffer pool                 : ok  a walk reuses one released set of GPU buffers")
    finally:
        model.block_swap = None


def check_window(swap):
    # Nothing is resident after a completed walk; fill both slots by hand and
    # confirm the third block waits for a release instead of taking VRAM.
    assert not swap.resident, list(swap.resident)
    swap.load(0)
    swap.load(1)
    assert set(swap.resident) == {0, 1}, list(swap.resident)
    swap.prefetch(2)
    assert 2 not in swap.resident, list(swap.resident)
    swap.release(0)
    swap.prefetch(2)
    assert set(swap.resident) == {1, 2}, list(swap.resident)
    swap.release(1)
    swap.release(2)
    assert not swap.resident, list(swap.resident)
    torch.cuda.synchronize()


def check_pool(swap):
    """A walk must stop allocating after the pool warms up.

    Without the pool, every load allocates a block's worth of GPU storage and
    every release frees it: 84 transients per step on the real model, which the
    caching allocator never coalesces, so the reserved footprint climbs until the
    card is full. Here the allocation count is the invariant.
    """
    created = []
    real_empty_like = torch.empty_like

    def counting_empty_like(*args, **kwargs):
        created.append(1)
        return real_empty_like(*args, **kwargs)

    torch.empty_like = counting_empty_like
    try:
        for index in range(BLOCKS):
            swap.prefetch(index)
            swap.wait(index)
            swap.release(index)
    finally:
        torch.empty_like = real_empty_like
    torch.cuda.synchronize()
    # One set per distinct block layout, plus the single pooled spare.
    assert len(created) <= BLOCKS, f"{len(created)} allocations for {BLOCKS} blocks: the pool is not used"
    assert len(created) <= 2, f"{len(created)} allocations for one repeated layout"


def main():
    if not torch.cuda.is_available():
        print("CUDA is not available; skipping the block-streaming test")
        return
    model = build_model()
    latent, context = inputs()
    check_streaming(model, latent, context)
    print("block streaming checks passed")


if __name__ == "__main__":
    main()
