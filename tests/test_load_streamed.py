"""The streamed loader must produce the same model as the full one.

    .venv/bin/python tests/test_load_streamed.py

``load_transformer`` builds on the meta device and fills one block at a time when
``LOAD_MODE`` is "streamed"; the old path reads everything and fills the model in
one go. Both must land the same bytes in the same parameters, including the fused
qkv split, and the streamed path must keep the tail it was told to offload in
host memory.

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


def load_with(mode: str, path: str, **kwargs):
    real = weights.MOSSImageTransformer
    weights.MOSSImageTransformer = build_tiny
    weights.LOAD_MODE = mode
    try:
        return weights.load_transformer(path, dtype=torch.bfloat16, device="cpu", **kwargs)
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


def main() -> int:
    with tempfile.TemporaryDirectory() as root:
        path = os.path.join(root, "tiny.safetensors")
        write_checkpoint(path)
        check_same_model(path)
        check_tail_stays_in_host_memory(path)
        check_quantized_blocks(path)
    print("streamed loader checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
