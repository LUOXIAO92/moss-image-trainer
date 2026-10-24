"""The streamed loader must produce the same model as the full one.

    .venv/bin/python tests/test_load_streamed.py

``load_transformer`` builds on the meta device and fills one block at a time when
``LOAD_MODE`` is "streamed"; the old path reads everything and fills the model in
one go. Both must land the same bytes in the same parameters, including the fused
qkv split, and the streamed path must keep the tail it was told to offload in
host memory. Quantised loads must also pack identical bytes, which is what pins
the packing kernel: the CPU one is what the bytes were before streaming.

The real checkpoint is 9.5 GiB, so the test builds a two-block model and points
the loader's constructor at it.
"""

from __future__ import annotations

import os
import sys
import tempfile

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import moss_lora.weights as weights  # noqa: E402
from moss_lora.model import MOSSImageTransformer  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

DIM = 64
FFN = 96
HEADS = 8
TEXT = 32
FREQ = 8
CHANNELS = 16
BLOCKS = 2


def build_tiny(dtype=torch.bfloat16, device=None):
    return MOSSImageTransformer(dim=DIM, ffn_dim=FFN, num_heads=HEADS, num_layers=BLOCKS,
                                text_dim=TEXT, freq_dim=FREQ, latent_channels=CHANNELS,
                                device=device, dtype=dtype)


def write_checkpoint(path: str) -> dict:
    torch.manual_seed(7)
    model = build_tiny(device="cpu")
    state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    save_file(state, path)
    return state


def load_with(mode: str, path: str, device: str = "cpu", **kwargs):
    real = weights.MOSSImageTransformer
    weights.MOSSImageTransformer = build_tiny
    weights.LOAD_MODE = mode
    try:
        return weights.load_transformer(path, dtype=torch.bfloat16, device=device, **kwargs)
    finally:
        weights.MOSSImageTransformer = real
        weights.LOAD_MODE = "streamed"


def check_same_model(path: str) -> None:
    full = load_with("full", path)
    streamed = load_with("streamed", path)
    left, right = dict(full.state_dict()), dict(streamed.state_dict())
    assert set(left) == set(right), (set(left) ^ set(right))
    for key, want in left.items():
        if key == "rope.bands":
            continue  # derived on every forward, never loaded from the checkpoint
        got = right[key]
        assert got.shape == want.shape, (key, got.shape, want.shape)
        assert torch.equal(got, want), f"{key} differs between the loaders"
    print(f"same weights                : ok  {len(left)} tensors identical")


def check_tail_stays_in_host_memory(path: str) -> None:
    streamed = load_with("streamed", path, blocks_to_swap=BLOCKS, quantize_scope="none")
    devices = {p.device.type for block in streamed.layers for p in block.parameters()}
    assert devices == {"cpu"}, devices
    head = [p.device.type for name, p in streamed.named_parameters() if not name.startswith("layers.")]
    assert set(head) == {"cpu"}, head
    print("offloaded tail              : ok  every block parameter stays in host memory")


def check_quantized_blocks(path: str) -> None:
    if not torch.cuda.is_available():
        print("quantized load             : skip  no CUDA")
        return
    model = load_with("streamed", path, quantize_scope="all")
    # device="cpu" keeps everything on the host; the packing still happened.
    for block in model.layers:
        linears = [m for m in block.modules() if type(m).__name__ == "Linear4bit"]
        assert linears, "no Linear4bit in a quantized block"
    print("quantized load              : ok  every block packed to nf4")


def check_pack_device(path: str) -> None:
    """``QUANTIZE_DEVICE`` must choose the kernel, not only the destination.

    A block arrives here already on the card. Handing a plain ``Params4bit`` to
    ``.to("cuda")`` packs on the card and honours nothing, so the loader's
    request has to reach ``quantize_4bit`` itself.
    """
    if not torch.cuda.is_available():
        print("packing kernel             : skip  no CUDA")
        return
    import bitsandbytes as bnb

    seen: list[str] = []
    real_quantize = bnb.functional.quantize_4bit

    def spy(tensor, *args, **kwargs):
        seen.append(tensor.device.type)
        return real_quantize(tensor, *args, **kwargs)

    real = weights.QUANTIZE_DEVICE
    weights.QUANTIZE_DEVICE = "cpu"
    bnb.functional.quantize_4bit = spy
    try:
        load_with("streamed", path, device="cuda", quantize_scope="all")
    finally:
        bnb.functional.quantize_4bit = real_quantize
        weights.QUANTIZE_DEVICE = real
    assert seen, "nothing was packed"
    assert set(seen) == {"cpu"}, f"the CPU kernel was requested, packing ran on {seen}"
    print(f"packing kernel             : ok  {len(seen)} packs ran on cpu with the block on the card")


def check_quantized_matches(path: str, device: str) -> None:
    """Both loaders must pack identical bytes when they pack on the same kernel.

    The streamed loader hands each block to the packing function while the block
    already sits on the card. A plain ``Params4bit`` packs inside ``.to(device)``
    and would therefore pack on the card even when the CPU kernel was requested;
    the comparison pins the CPU pack and fails if that regresses.
    """
    real = weights.QUANTIZE_DEVICE
    weights.QUANTIZE_DEVICE = "cpu"  # the full path always packs on the host
    try:
        full = load_with("full", path, device=device, quantize_scope="all")
        streamed = load_with("streamed", path, device=device, quantize_scope="all")
    finally:
        weights.QUANTIZE_DEVICE = real
    pairs = [(a, b) for (_, a), (_, b) in zip(full.named_parameters(), streamed.named_parameters())
             if hasattr(a, "quant_state")]
    assert pairs, "no packed weight found"
    for want, got in pairs:
        assert want.quant_state.absmax.shape == got.quant_state.absmax.shape, "scale shape differs"
        assert torch.equal(want.data.cpu(), got.data.cpu()), "packed codes differ between the loaders"
        assert torch.equal(want.quant_state.absmax.cpu(),
                           got.quant_state.absmax.cpu()), "quantisation scales differ between the loaders"
    print(f"packed on {device:<19}: ok  {len(pairs)} packed weights identical, CPU kernel")


def main() -> int:
    with tempfile.TemporaryDirectory() as root:
        path = os.path.join(root, "tiny.safetensors")
        write_checkpoint(path)
        check_same_model(path)
        check_tail_stays_in_host_memory(path)
        check_quantized_blocks(path)
        check_pack_device(path)
        check_quantized_matches(path, "cpu")
        if torch.cuda.is_available():
            check_quantized_matches(path, "cuda")
    print("streamed loader checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
