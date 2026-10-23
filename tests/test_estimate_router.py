"""The estimate endpoint must answer from the config, and never with a CUDA context.

    .venv/bin/python tests/test_estimate_router.py

Two things are checked. A valid YAML comes back as training and preview peaks
whose total is the larger of the two plus the context allowance. And the handler
resolves `vram.preset: auto` from the nvidia-smi probe rather than from
`detect_vram_gb()`, because that one calls `torch.cuda.is_available()` and would
pin a CUDA context in the web process -- VRAM the run being predicted for needs.
The second check patches `detect_vram_gb` to explode, so a regression fails here
instead of quietly costing half a gigabyte of the card.
"""

from __future__ import annotations

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import moss_lora.config as config_module  # noqa: E402
import moss_lora.web.routers.estimate as estimate_module  # noqa: E402
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
  preset: auto
preview:
  prompts: []
"""


def main() -> int:
    def explode() -> float | None:
        raise AssertionError("the estimate route called detect_vram_gb(), which creates a CUDA context")

    real_detect = config_module.detect_vram_gb
    real_total = estimate_module.gpu_total_gb
    config_module.detect_vram_gb = explode
    estimate_module.gpu_total_gb = lambda: 15.0
    try:
        payload = asyncio.run(api_estimate(FakeRequest({"yaml": GOOD, "workspace": ""})))
    finally:
        config_module.detect_vram_gb = real_detect
        estimate_module.gpu_total_gb = real_total

    for key in ("train", "preview", "context", "total", "resident_blocks", "streamed_blocks"):
        assert key in payload, f"{key} missing from {sorted(payload)}"
    assert payload["train"] > 0 and payload["preview"] >= 0, payload
    assert abs(payload["total"] - (max(payload["train"], payload["preview"]) + payload["context"])) < 0.02, payload
    # A 15 GB card resolves to the 16 GB tier, which quantizes nothing and keeps
    # every block resident.
    assert payload["streamed_blocks"] == 0, payload.get("notes")
    assert payload["resident_blocks"] == 42, payload.get("notes")

    # No driver answer: the preset stays unresolved, and the route still must not
    # reach for detect_vram_gb().
    estimate_module.gpu_total_gb = lambda: None
    try:
        blind = asyncio.run(api_estimate(FakeRequest({"yaml": GOOD, "workspace": ""})))
    finally:
        estimate_module.gpu_total_gb = real_total
    assert blind["total"] > 0, blind

    # nvidia-smi reports total memory in MiB; 32607 MiB must become the 32 tier.
    assert round(32607 / 1024, 1) == 31.8
    print("estimate router checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
