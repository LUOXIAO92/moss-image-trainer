"""Caption dropout uses a cached empty prompt and stays off at 0.

    .venv\\Scripts\\python.exe tests/test_caption_dropout.py

The training process never loads the 9B text encoder.  The unconditional
prompt is therefore encoded once by ``encode-text`` and read from
``empty_text.safetensors``; this test covers that handoff plus the per-sample
replacement in ``_load_packed``.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from moss_lora.cache import empty_text_path  # noqa: E402
from moss_lora.config import Config  # noqa: E402
from moss_lora.data import ALIGN, sizing_spec  # noqa: E402
from moss_lora.encode import write_empty_text  # noqa: E402
from moss_lora.trainer import Trainer  # noqa: E402
from moss_lora.web.services.cache_cmd import _cache_summary  # noqa: E402


def check_empty_bundle(root: str) -> None:
    calls: list[list[str]] = []

    def encode(texts):
        calls.append(texts)
        return torch.ones(1, 2, 4), torch.tensor([[True, True]])

    write_empty_text(root, encode)
    path = empty_text_path(root)
    data = load_file(path)
    assert calls == [[""]], calls
    assert set(data) == {"embeds", "mask"}, data.keys()
    assert data["embeds"].shape == (1, 2, 4), data["embeds"].shape
    assert data["mask"].tolist() == [[True, True]], data["mask"]
    with safe_open(path, framework="pt") as handle:
        assert (handle.metadata() or {}).get("prompt") == "", handle.metadata()


def make_trainer(root: str, dropout: float) -> Trainer:
    trainer = Trainer.__new__(Trainer)
    trainer.cache_dir = root
    trainer.device = "cpu"
    trainer.dtype = torch.float32
    trainer.caption_dropout = dropout
    trainer._empty_context = None
    return trainer


def write_item(root: str) -> None:
    save_file(
        {
            "latent": torch.zeros(4, 2, 2, dtype=torch.float16),
            "text_embeds": torch.zeros(3, 4, dtype=torch.float32),
            "text_mask": torch.tensor([True, True, False]),
        },
        os.path.join(root, "item.safetensors"),
    )


def check_replacement(root: str) -> None:
    write_item(root)
    trainer = make_trainer(root, 1.0)
    trainer._empty_context = trainer._load_empty_context()
    latents, contexts = trainer._load_packed(
        [{"file": "item.safetensors"}],
        drop_rng=torch.Generator().manual_seed(0),
    )
    assert len(latents) == 1
    assert contexts[0].shape == (1, 2, 4), contexts[0].shape
    assert torch.equal(contexts[0], torch.ones(1, 2, 4)), contexts[0]

    # The same cached unconditional branch is ignored when the configured
    # probability is 0: a user who sets it to 0 keeps the caption.
    trainer.caption_dropout = 0.0
    _, contexts = trainer._load_packed(
        [{"file": "item.safetensors"}],
        drop_rng=torch.Generator().manual_seed(0),
    )
    assert torch.equal(contexts[0], torch.zeros(1, 2, 4)), contexts[0]

    os.remove(empty_text_path(root))
    try:
        trainer._load_empty_context()
    except SystemExit as exc:
        message = str(exc)
        assert "empty prompt" in message and "encode-text" in message, message
    else:
        raise AssertionError("missing empty prompt was accepted")


def check_defaults_and_validation() -> None:
    config = Config()
    assert config.train.caption_dropout == 0.0, config.train.caption_dropout
    config.train.caption_dropout = 1.5
    codes = [problem.code for problem in config.validate()]
    assert "err.config.caption_dropout" in codes, codes


def check_cache_readiness(root: str) -> None:
    os.makedirs(root, exist_ok=True)
    with open(os.path.join(root, "index.json"), "w", encoding="utf-8") as handle:
        json.dump({
            "schema": 2,
            "items": [],
            "sizing": sizing_spec(1024 * 1024, ALIGN),
        }, handle)

    config = Config()
    config.cache.dir = root
    config.preview.prompts = []
    config.preview.negative_prompt = ""
    summary = _cache_summary(config)
    assert summary["empty_text_ready"] is False, summary
    assert summary["ready"] is True, summary

    config.train.caption_dropout = 0.1
    summary = _cache_summary(config)
    assert summary["empty_text_ready"] is False, summary
    assert summary["missing_empty_text"] is True, summary
    assert summary["ready"] is False, summary

    write_empty_text(root, lambda texts: (
        torch.ones(1, 2, 4), torch.tensor([[True, True]])))
    summary = _cache_summary(config)
    assert summary["empty_text_ready"] is True, summary
    assert summary["missing_empty_text"] is False, summary
    assert summary["ready"] is True, summary


def main() -> int:
    root = tempfile.mkdtemp(prefix="moss-caption-dropout-")
    try:
        check_empty_bundle(root)
        check_replacement(root)
        check_defaults_and_validation()
        cache_root = tempfile.mkdtemp(prefix="moss-caption-cache-")
        try:
            check_cache_readiness(cache_root)
        finally:
            shutil.rmtree(cache_root, ignore_errors=True)
        print("caption-dropout checks passed")
    finally:
        shutil.rmtree(root, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
