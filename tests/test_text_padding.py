"""Smoke checks for the local HuggingFace text encoder.

Optional: set MOSS_TE to a local Qwen3.5-9B directory to run it.  The product
does not ship a model or assume a path.

    MOSS_TE=/path/to/Qwen3.5-9B .venv/bin/python tests/test_text_padding.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main() -> int:
    model_dir = os.environ.get("MOSS_TE", "").strip()
    if not model_dir:
        print("skip: set MOSS_TE to a local Qwen3.5-9B directory")
        return 0
    from moss_lora.hf_text import HfTextEncoder

    encoder = HfTextEncoder(model_dir)

    prompts = ["mychar, girl", "a much longer caption about a girl standing in a garden at golden hour"]
    embeds, mask = encoder.encode(prompts, batch_size=2)
    assert embeds.shape[0] == len(prompts), embeds.shape
    assert mask.shape == embeds.shape[:2], (mask.shape, embeds.shape)

    # The shorter prompt must not be affected by padding in a mixed-length batch.
    short = "mychar, girl, portrait"
    long = "a red fox sitting in a snowy forest at golden hour, photorealistic, cinematic lighting, snow falling"
    alone, _ = encoder.encode([short], batch_size=1)
    batched, _ = encoder.encode([short, long], batch_size=2)
    import torch

    cos = torch.nn.functional.cosine_similarity(alone[0].flatten(), batched[0].flatten(), dim=0).item()
    assert cos > 0.999, f"padding leaked into a valid prompt: cosine {cos}"
    print(f"hf text encoder passed (batch-vs-single cosine {cos:.6f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
