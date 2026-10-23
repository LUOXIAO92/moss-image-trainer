"""The preflight endpoint must answer a clean YAML without blowing up.

    .venv/bin/python tests/test_validate_router.py

Guards a shadowing bug: the handler assigns a local `problem` while parsing a
broken YAML, which shadowed the imported `problem()` helper for the whole
function and turned every *valid* preflight into an UnboundLocalError.
"""

from __future__ import annotations

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from moss_lora.web.routers.validate import api_validate  # noqa: E402


class FakeRequest:
    def __init__(self, body: dict):
        self._body = body

    async def json(self) -> dict:
        return self._body


GOOD = """name: probe
model:
  dit: /nope/dit.safetensors
dataset:
  path: /nope/dataset
  reg_path: /nope/reg
cache:
  dir: /nope/cache
preview:
  prompts: []
"""

BAD = "name: probe\nmodel: [unclosed\n"


def main() -> int:
    payload = asyncio.run(api_validate(FakeRequest({"yaml": GOOD, "workspace": ""})))
    assert payload["ok"] is False
    assert payload["yaml_error"] is None, payload["yaml_error"]
    codes = [item["code"] for item in payload["problem_codes"] if item]
    assert "err.dataset.reg_path_missing" in codes, codes
    assert "err.cache.dir_empty" not in codes, "cache.dir is set"
    assert payload["problems"] and len(payload["problems"]) == len(payload["problem_codes"])

    broken = asyncio.run(api_validate(FakeRequest({"yaml": BAD, "workspace": ""})))
    assert broken["yaml_error"]["code"] == "err.yaml.syntax", broken["yaml_error"]
    assert "问题" in broken["yaml_error"]["message"] or "YAML" in broken["yaml_error"]["message"]

    empty = asyncio.run(api_validate(FakeRequest({"yaml": "   ", "workspace": ""})))
    assert empty["yaml_error"]["code"] == "err.yaml.empty", empty["yaml_error"]
    print("validate router checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
