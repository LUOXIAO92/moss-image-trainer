"""The preview bundle records the prompts it was encoded from.

    .venv/bin/python tests/test_preview_cache.py

A preview samples whatever vectors sit in ``preview.safetensors``. Editing the
prompt list without re-encoding would quietly draw the old text, so the bundle
carries its prompts in the safetensors metadata and the trainer refuses a bundle
that no longer matches the config.

No model and no GPU: the state function only reads the file's header.
"""

from __future__ import annotations

import os
import sys
import tempfile

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from moss_lora.cache import preview_cache_state, preview_metadata, preview_path  # noqa: E402
from moss_lora.encode import _write_preview_bundle  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

PROMPTS = ["a girl on a bed", "a girl in a forest"]
NEGATIVE = "worst quality"


def write_bundle(cache_dir: str, prompts=None, negative: str = NEGATIVE, metadata=True) -> None:
    tensors = {
        "embeds": torch.zeros(len(prompts or PROMPTS), 1, 4, dtype=torch.bfloat16),
        "mask": torch.ones(len(prompts or PROMPTS), 4, dtype=torch.long),
        "negative_embeds": torch.zeros(1, 1, 4, dtype=torch.bfloat16),
        "negative_mask": torch.ones(1, 4, dtype=torch.long),
    }
    kwargs = {"metadata": preview_metadata(prompts or PROMPTS, negative)} if metadata else {}
    save_file(tensors, preview_path(cache_dir), **kwargs)


def check_states() -> None:
    with tempfile.TemporaryDirectory() as root:
        assert preview_cache_state(root, PROMPTS, NEGATIVE) == "missing"

        write_bundle(root)
        assert preview_cache_state(root, PROMPTS, NEGATIVE) == "ok"

        # Same prompts, one character different: the vectors are for other text.
        edited = list(PROMPTS)
        edited[0] += "!"
        assert preview_cache_state(root, edited, NEGATIVE) == "stale"
        # Reordering is a different list too.
        assert preview_cache_state(root, list(reversed(PROMPTS)), NEGATIVE) == "stale"
        # The negative prompt is encoded into the same bundle.
        assert preview_cache_state(root, PROMPTS, NEGATIVE + " ") == "stale"
        # An empty prompt list never matches a bundle that has prompts.
        assert preview_cache_state(root, [], NEGATIVE) == "stale"

        write_bundle(root, negative="")
        assert preview_cache_state(root, PROMPTS, "") == "ok"


def check_legacy_bundle() -> None:
    """A bundle written before the metadata existed must read as stale, not ok."""
    with tempfile.TemporaryDirectory() as root:
        write_bundle(root, metadata=False)
        assert preview_cache_state(root, PROMPTS, NEGATIVE) == "stale"
        # A truncated file is stale too, never an exception in the training loop.
        with open(preview_path(root), "wb") as handle:
            handle.write(b"not a safetensors file")
        assert preview_cache_state(root, PROMPTS, NEGATIVE) == "stale"


def check_write_bundle() -> None:
    """The write helper uses preview_path() and keeps the prompt metadata."""
    with tempfile.TemporaryDirectory() as root:
        calls = []

        def encode(texts):
            calls.append(list(texts))
            count = len(texts)
            return (torch.zeros(count, 1, 4, dtype=torch.bfloat16),
                    torch.ones(count, 4, dtype=torch.long))

        _write_preview_bundle(root, encode, PROMPTS, NEGATIVE)
        assert calls == [PROMPTS, [NEGATIVE]], calls
        assert preview_cache_state(root, PROMPTS, NEGATIVE) == "ok"
        print("preview bundle write: ok  path helper stays callable")


def main() -> int:
    check_states()
    check_legacy_bundle()
    check_write_bundle()
    print("preview cache checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
