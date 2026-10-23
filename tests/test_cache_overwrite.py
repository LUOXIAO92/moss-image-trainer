"""`cache.overwrite` must rebuild existing sample files, and only then.

The builder decides per sample file: with the flag off an existing file with the
requested tensor is skipped, with it on every item is encoded again. The check
uses a temp cache directory and a fake sample file, so no dataset and no GPU are
involved.

    .venv/bin/python tests/test_cache_overwrite.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import types

import torch
import yaml
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from moss_lora.cache import CacheBuilder, item_uid  # noqa: E402
from moss_lora.cli import build_parser  # noqa: E402
from moss_lora.config import Config  # noqa: E402
from moss_lora.data import ALIGN, sizing_spec  # noqa: E402
from moss_lora.web.services.cache_cmd import _cache_summary  # noqa: E402
from moss_lora.web.services.launcher import _command  # noqa: E402


def _item() -> types.SimpleNamespace:
    return types.SimpleNamespace(
        key="fake-sample", image="fake.png", caption="a fake caption",
        size=(64, 64), repeats=1, reg=False,
    )


def _builder(directory: str, item) -> CacheBuilder:
    """A builder with the constructor's dataset scan and GPU setup removed."""
    builder = CacheBuilder.__new__(CacheBuilder)
    builder.dir = directory
    builder.reg_dir = os.path.join(directory, "reg_cache")
    builder.sizing = sizing_spec(Config().dataset.max_area, ALIGN)
    builder.sizing_changed = False
    builder.items = [item]
    builder.batch = {}
    builder.log = lambda *_: None
    builder.config = Config.from_dict({"cache": {"dir": directory}})
    return builder


def check_removed_fields_still_load() -> None:
    """Old YAML may still name the four removed keys; loading must ignore them."""
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "config.yaml")
        with open(path, "w", encoding="utf-8") as handle:
            yaml.safe_dump({"cache": {
                "dir": "cache",
                "text_max_length": 2048,
                "text_dtype": "float16",
                "latent_dtype": "bfloat16",
                "latent_sample": True,
                "text_batch": 8,
            }}, handle)
        config = Config.load(path)
    for removed in ("text_max_length", "text_dtype", "latent_dtype", "latent_sample"):
        assert not hasattr(config.cache, removed), removed
    assert config.cache.text_batch == 8, config.cache.text_batch
    assert config.cache.overwrite is False, config.cache.overwrite


def check_existing_file_skipped_until_overwrite() -> None:
    with tempfile.TemporaryDirectory() as directory:
        item = _item()
        builder = _builder(directory, item)
        builder.config.cache.overwrite = False
        path = builder.path_for(item)
        # One fake sample file holding every tensor the builder looks for.
        save_file({
            "latent": torch.zeros(1, 1, dtype=torch.float16),
            "text_embeds": torch.zeros(1, 1, dtype=torch.bfloat16),
            "text_mask": torch.ones(1, dtype=torch.bool),
        }, path, metadata=builder.item_metadata(item))

        # Index and summary see the existing sample and call the cache ready.
        builder.write_index()
        with open(os.path.join(directory, "index.json"), encoding="utf-8") as handle:
            index = json.load(handle)
        assert [entry["file"] for entry in index["items"]] == [f"{item_uid(item)}.safetensors"]
        assert index["sizing"] == builder.sizing, index["sizing"]
        summary = _cache_summary(builder.config)
        assert summary["ready"] and summary["index_ready"], summary
        assert summary["items"] == 1 and summary["text_items"] == 1, summary

        assert builder._pending("latent") == [], "an existing latent should be skipped"
        assert builder._pending("text_embeds") == [], "an existing text embedding should be skipped"

        builder.config.cache.overwrite = True
        assert builder._pending("latent") == [item], "overwrite must rebuild the latent"
        assert builder._pending("text_embeds") == [item], "overwrite must rebuild the text embedding"


def check_missing_tensor_still_rebuilds() -> None:
    """Without overwrite, a sample file that lacks the tensor is pending."""
    with tempfile.TemporaryDirectory() as directory:
        item = _item()
        builder = _builder(directory, item)
        save_file({"latent": torch.zeros(1, 1)}, builder.path_for(item))
        assert builder._pending("latent") == [], "a present tensor must be kept"
        assert builder._pending("text_embeds") == [item], "a missing tensor must be rebuilt"


def check_cli_and_job_command() -> None:
    parser = build_parser()
    assert parser.parse_args(["cache", "--config", "run.yaml"]).overwrite is False
    assert parser.parse_args(["cache", "--config", "run.yaml", "--overwrite"]).overwrite is True

    latent = _command("cache", "/cache/run.yaml", ["latent"], overwrite=True)
    assert latent[:2] == ["bash", "-c"], latent
    assert "moss_lora.cli cache" in latent[2], latent
    assert latent[2].endswith("--overwrite"), latent
    assert "encode-text" not in latent[2], "a latents-only rebuild must not touch captions"

    both = _command("cache", "/cache/run.yaml", ["latent", "text"], overwrite=True)
    assert both[2].count("--overwrite") == 2, both

    plain = _command("cache", "/cache/run.yaml", ["latent", "text"], overwrite=False)
    assert "--overwrite" not in plain[2], plain


def main() -> None:
    check_removed_fields_still_load()
    check_existing_file_skipped_until_overwrite()
    check_missing_tensor_still_rebuilds()
    check_cli_and_job_command()
    print("cache overwrite checks passed")


if __name__ == "__main__":
    main()
