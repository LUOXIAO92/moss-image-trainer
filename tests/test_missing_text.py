"""A cache file without text embeddings must fail with the next step, not a KeyError.

    .venv/bin/python tests/test_missing_text.py

Building latents alone (or building a regularization cache by itself) leaves item
files that hold only ``latent``. Training used to die on `KeyError: 'text_mask'`,
which says nothing about the fix; the loader now names the missing keys and the
command that produces them.
"""

from __future__ import annotations

import os
import sys
import tempfile

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from moss_lora.trainer import Trainer  # noqa: E402
from safetensors.torch import save_file  # noqa: E402


def make_trainer(root: str) -> Trainer:
    trainer = Trainer.__new__(Trainer)
    trainer.cache_dir = root
    trainer.device = "cpu"
    trainer.dtype = torch.bfloat16
    return trainer


def main() -> int:
    with tempfile.TemporaryDirectory() as root:
        save_file({"latent": torch.zeros(4, 2, 2, dtype=torch.float16)},
                  os.path.join(root, "item.safetensors"))
        trainer = make_trainer(root)
        try:
            trainer._load_packed([{"file": "item.safetensors"}])
        except SystemExit as exc:
            message = str(exc)
            assert "text_embeds" in message and "text_mask" in message, message
            assert "encode-text" in message, message
            print("missing text               : ok  names the keys and the command")
        else:
            raise AssertionError("a cache item without text embeddings was accepted")
    print("missing-text checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
