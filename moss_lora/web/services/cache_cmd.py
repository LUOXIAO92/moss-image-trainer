# moved from app.py: L706-718, L962-1025
"""Cache job parts and the cache-readiness summary."""

from __future__ import annotations

import json
import os

from fastapi import HTTPException

from ...config import Config, expand
from ..deps import _rel


CACHE_PARTS = ("latent", "text")
CACHE_PART_LABELS = {"latent": "latent 缓存", "text": "文本编码"}


def _cache_parts(body: dict) -> list[str]:
    """缓存任务要生成哪几样：latent / 文本，勾几个跑几个，顺序固定。"""
    raw = body.get("parts")
    if raw is None:
        raw = ["latent", "text"]
    parts = [part for part in CACHE_PARTS if part in (raw or [])]
    if not parts:
        raise HTTPException(400, "至少勾一个：latent 缓存 或 文本编码")
    return parts

def _cache_summary(config: Config) -> dict:
    """Cache readiness without changing the historical ``items``/``ready`` keys."""
    cache_dir = os.path.abspath(os.path.expanduser(expand(config.cache.dir)))
    index_path = os.path.join(cache_dir, "index.json")
    preview_path = os.path.join(cache_dir, "preview.safetensors")
    index_ready = os.path.isfile(index_path)
    preview_ready = os.path.isfile(preview_path)
    from ...cache import reg_cache_dir

    reg_dir = reg_cache_dir(config)
    info = {
        "items": 0,
        "ready": index_ready,
        "dir": cache_dir,
        "dir_rel": _rel(cache_dir),
        "reg_dir": reg_dir,
        "reg_dir_rel": _rel(reg_dir),
        "index_ready": index_ready,
        # preview.safetensors 只用于预览图，缺了不影响训练，前端不要当成阻塞项
        "preview_ready": preview_ready,
        "index_mtime": os.path.getmtime(index_path) if index_ready else None,
    }
    if not index_ready:
        return info
    with open(index_path, encoding="utf-8") as f:
        index = json.load(f)
    items = index.get("items", [])
    info["items"] = len(items)
    reg_items = [item for item in items if item.get("reg")]
    info["reg_items"] = len(reg_items)
    # 索引格式只认 2（训练集/正则集分目录 + 逐条 file 相对路径）；其它一律当不可用
    try:
        schema = int(index.get("schema", 0))
    except (TypeError, ValueError):
        schema = 0
    info["legacy_index"] = schema != 2
    if info["legacy_index"]:
        info["ready"] = False
    if index.get("reg_dir"):
        info["reg_dir"] = index["reg_dir"]
        info["reg_dir_rel"] = _rel(index["reg_dir"])
    info["train_items"] = info["items"] - info["reg_items"]
    # 逐条文件的实际落盘数量（训练集与正则集分目录）
    info["train_files"] = sum(
        1 for item in items
        if not item.get("reg") and os.path.isfile(os.path.join(cache_dir, item.get("file", "")))
    )
    info["reg_files"] = sum(
        1 for item in items
        if item.get("reg") and os.path.isfile(os.path.join(cache_dir, item.get("file", "")))
    )
    # 文本嵌入是第二条流水线（encode-text）；只有 latent 的时候训练会在读缓存时炸
    text_items = 0
    for item in items:
        try:
            if int(item.get("text_len") or 0) > 0:
                text_items += 1
        except (TypeError, ValueError):
            continue
    info["text_items"] = text_items
    info["missing_text"] = info["items"] - text_items
    if not info["legacy_index"] and info["missing_text"] > 0:
        info["ready"] = False
    # 尺寸规格对不上：latent 的宽高来自另一套 max_area/对齐，训练会拒绝加载
    from ...data import ALIGN, sizing_spec

    expected = sizing_spec(config.dataset.max_area, ALIGN)
    cached = index.get("sizing")
    info["sizing"] = cached or None
    info["stale_sizing"] = cached != expected
    if info["stale_sizing"]:
        info["ready"] = False
    return info
