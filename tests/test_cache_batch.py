"""缓存索引必须记下每个阶段用过的 batch 大小。

同一段提示词在 batch=1 和 batch=64 下编码出的向量不同（不同的 kernel、不同的
归约顺序），所以一份缓存不写 batch 就无法复现，也无法解释两份缓存的差异。
latent（VAE）同理。

    .venv/bin/python tests/test_cache_batch.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from moss_lora.cache import CacheBuilder  # noqa: E402
from moss_lora.config import Config  # noqa: E402


def _builder(directory: str):
    """A builder stripped to the parts write_index touches (no images involved)."""
    builder = CacheBuilder.__new__(CacheBuilder)
    builder.dir = directory
    builder.reg_dir = ""
    builder.sizing = {"mode": "native", "area": 1048576, "align": 16}
    builder.items = []
    builder.batch = {}
    builder.log = lambda *_: None
    builder.config = types.SimpleNamespace(
        dataset=types.SimpleNamespace(moss_caption_source="tag_caption"))
    return builder


def _read(directory: str) -> dict:
    with open(os.path.join(directory, "index.json"), encoding="utf-8") as handle:
        return json.load(handle)


def check_records_and_merges():
    with tempfile.TemporaryDirectory() as directory:
        latent = _builder(directory)
        latent.record_batch("latent", 4)
        latent.write_index()
        assert _read(directory)["batch"] == {"latent": 4}, _read(directory)["batch"]

        # 单独跑 encode-text 只能加自己的那一项，不能抹掉 latent 的记录
        text = _builder(directory)
        text.record_batch("text", 64)
        text.write_index()
        assert _read(directory)["batch"] == {"latent": 4, "text": 64}, _read(directory)["batch"]


def check_empty_batch_is_still_explicit():
    with tempfile.TemporaryDirectory() as directory:
        _builder(directory).write_index()
        assert _read(directory)["batch"] == {}, _read(directory)["batch"]


def check_config_fields():
    """UI 上填的固定 batch 必须能读进配置，负数要被校验拦住。"""
    config = Config.from_dict({"cache": {"text_batch": 32, "vae_batch": 2}})
    assert config.cache.text_batch == 32, config.cache.text_batch
    assert config.cache.vae_batch == 2, config.cache.vae_batch
    assert not [p for p in config.validate() if "cache." in p], config.validate()

    bad = Config.from_dict({"cache": {"text_batch": -1}})
    assert any("cache.text_batch" in p for p in bad.validate()), bad.validate()


def main():
    check_config_fields()
    check_records_and_merges()
    check_empty_batch_is_still_explicit()
    print("cache batch checks passed")


if __name__ == "__main__":
    main()
