"""The estimate endpoint must answer from the config, and never with a CUDA context.

    .venv/bin/python tests/test_estimate_router.py

Three things are checked. A valid YAML comes back as training and preview peaks
whose total is the larger of the two plus the context allowance. The handler
reads the five VRAM fields the YAML carries instead of resolving a preset from
the card, so the number it prints is the one the run would produce. And the
handler never asks the driver anything: `detect_vram_gb()` calls
`torch.cuda.is_available()` and would pin a CUDA context in the web process --
VRAM the run being predicted for needs. The probe is patched to explode so a
regression fails here instead of quietly costing half a gigabyte of the card.
"""

from __future__ import annotations

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import moss_lora.config as config_module  # noqa: E402
from moss_lora.web.routers.estimate import api_estimate  # noqa: E402


class FakeRequest:
    def __init__(self, body: dict):
        self._body = body

    async def json(self) -> dict:
        return self._body


GOOD = """name: estimate
model:
  dit: /nope/dit.safetensors
dataset:
  path: /nope/dataset
cache:
  dir: /nope/cache
vram:
  blocks_to_swap: 12
  quantize_dit: ffn
preview:
  prompts: []
"""

# A config written before presets became manual still loads: "auto" and -1 fold
# into the documented defaults instead of asking the card.
LEGACY = """name: legacy
model:
  dit: /nope/dit.safetensors
vram:
  preset: auto
  gradient_checkpointing: auto
  blocks_to_swap: -1
"""


def main() -> int:
    def explode() -> float | None:
        raise AssertionError("the estimate route asked the driver, which creates a CUDA context")

    real_detect = config_module.detect_vram_gb
    config_module.detect_vram_gb = explode
    try:
        payload = asyncio.run(api_estimate(FakeRequest({"yaml": GOOD, "workspace": ""})))
        blind = asyncio.run(api_estimate(FakeRequest({"yaml": LEGACY, "workspace": ""})))
    finally:
        config_module.detect_vram_gb = real_detect

    for key in ("train", "preview", "context", "total", "resident_blocks", "streamed_blocks"):
        assert key in payload, f"{key} missing from {sorted(payload)}"
    assert payload["train"] > 0 and payload["preview"] >= 0, payload
    assert abs(payload["total"] - (max(payload["train"], payload["preview"]) + payload["context"])) < 0.02, payload
    # The YAML says 12 of 42 blocks stream. The three-block lookahead and pool
    # stay resident, so the estimator counts 33 resident and 9 streamed.
    assert payload["resident_blocks"] == 33, payload.get("notes")
    assert payload["streamed_blocks"] == 9, payload.get("notes")

    # The retired sentinels resolve to defaults: everything resident, nothing
    # quantized, and no driver call in the path.
    assert blind["resident_blocks"] == 42, blind.get("notes")
    assert blind["streamed_blocks"] == 0, blind.get("notes")
    assert blind["total"] > 0, blind
    print("estimate router checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
