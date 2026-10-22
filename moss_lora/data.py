"""Dataset scanning and resolution bucketing."""

from __future__ import annotations

import math

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
_REPEAT_DIR = re.compile(r"^(\d+)[_-](.+)$")

# 提示词生成 worker 写在图片旁边的专用 sidecar：foo.jpg -> foo_moss-image-trainer.json
MOSS_SIDECAR_SUFFIX = "_moss-image-trainer.json"
MOSS_STYLE_TAG = "tag_caption_style"
MOSS_STYLE_OFFICIAL = "official_style"
# 训练文本可以取的字段：tag+caption 那段，或官方描述的三档
MOSS_CAPTION_SOURCES = ("tag_caption", "official_long", "official_medium", "official_short")
MOSS_CAPTION_SOURCE_LABELS = {
    "tag_caption": "tag + caption",
    "official_long": "官方描述 · 长",
    "official_medium": "官方描述 · 中",
    "official_short": "官方描述 · 短",
}


@dataclass
class Item:
    image: str
    caption: str
    repeats: int = 1
    width: int = 0
    height: int = 0
    bucket: tuple[int, int] = (0, 0)
    tags: list[str] = field(default_factory=list)
    reg: bool = False           # from dataset.reg_path (regularisation images)
    caption_source: str = ""    # moss | file | metadata | default

    @property
    def key(self) -> str:
        return os.path.abspath(self.image)


def _sidecar_data(stem: Path) -> dict:
    path = Path(str(stem) + MOSS_SIDECAR_SUFFIX)
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def sidecar_styles(stem: Path) -> set[str]:
    """Which caption styles this image already has (``tag_caption_style`` etc.)."""
    data = _sidecar_data(stem)
    return {key for key in (MOSS_STYLE_TAG, MOSS_STYLE_OFFICIAL) if isinstance(data.get(key), dict)}


def read_moss_caption(stem: Path, source: str = "tag_caption") -> str:
    """Caption from ``<stem>_moss-image-trainer.json`` for the chosen source.

    ``source`` picks one style/field: ``tag_caption`` -> ``tag_caption_style.caption``,
    ``official_long|medium|short`` -> the matching tier of ``official_style``.
    Empty when the image does not have that style.
    """
    data = _sidecar_data(stem)
    if not data:
        return ""
    if source == "tag_caption":
        style = data.get(MOSS_STYLE_TAG)
        value = style.get("caption") if isinstance(style, dict) else None
        return value.strip() if isinstance(value, str) and value.strip() else ""
    tier = {"official_long": "caption_long", "official_medium": "caption_medium",
            "official_short": "caption_short"}.get(source)
    style = data.get(MOSS_STYLE_OFFICIAL)
    if not tier or not isinstance(style, dict):
        return ""
    value = style.get(tier)
    return value.strip() if isinstance(value, str) and value.strip() else ""


def _read_caption(stem: Path, extensions) -> str | None:
    for ext in extensions:
        path = stem.with_suffix(ext)
        if path.is_file():
            return path.read_text(encoding="utf-8").strip()
    return None


def scan(root: str, caption_ext=(".txt", ".caption"), metadata: str = "", default_caption: str = "",
         moss_caption_source: str = "tag_caption"):
    """Walk ``root`` and pair images with captions.

    Captions come from a sidecar file, a ``metadata.jsonl`` entry, or ``default_caption``.
    Directory names like ``3_myconcept`` add a repeat count (kohya convention).
    """
    root = os.path.abspath(os.path.expanduser(root))
    if not os.path.isdir(root):
        raise FileNotFoundError(root)
    meta = {}
    if metadata:
        with open(metadata, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                name = row.get("file_name") or row.get("image") or row.get("path")
                text = row.get("caption") or row.get("prompt") or row.get("text") or ""
                if name:
                    meta[os.path.abspath(os.path.join(root, name))] = text

    items: list[Item] = []
    for dirpath, _, filenames in os.walk(root):
        repeats = 1
        match = _REPEAT_DIR.match(os.path.basename(dirpath))
        if match:
            repeats = int(match.group(1))
        for name in sorted(filenames):
            path = Path(dirpath, name)
            if path.suffix.lower() not in IMAGE_EXTS:
                continue
            key = os.path.abspath(path)
            stem = path.with_suffix("")
            source = "metadata" if meta.get(key) else ""
            caption = meta.get(key) or ""
            if not caption:
                caption = read_moss_caption(stem, moss_caption_source)
                source = "moss" if caption else ""
            if not caption:
                caption = _read_caption(stem, caption_ext) or ""
                source = "file" if caption else ""
            if not caption and default_caption:
                caption, source = default_caption, "default"
            items.append(Item(image=str(path), caption=caption, repeats=repeats, caption_source=source))
    if not items:
        raise RuntimeError(f"no images with supported extensions under {root}")
    return items


def read_size(path: str) -> tuple[int, int]:
    with Image.open(path) as img:
        return img.size


# a deliberately coarse grid: 7 aspect ratios, each keeping the pixel budget
BUCKET_ASPECTS = (1.0, 4 / 3, 3 / 4, 3 / 2, 2 / 3, 16 / 9, 9 / 16)


def bucket_resolutions(area: int, step: int = 64):
    """One bucket per aspect ratio, both sides rounded to ``step``."""
    buckets = []
    for aspect in BUCKET_ASPECTS:
        width = int(round(math.sqrt(area * aspect) / step)) * step
        height = int(round(math.sqrt(area / aspect) / step)) * step
        bucket = (max(step, width), max(step, height))
        if bucket not in buckets:
            buckets.append(bucket)
    return buckets


def target_size(width: int, height: int, area: int, step: int = 16):
    """保持宽高比、面积封顶、宽高对齐到 step，不再放大。

    官方训练配方（技术报告 §3.3）就是这个做法：preserve aspect ratio、对齐 16、
    面积封顶。用它而不是固定桶时，每张图保留自己的分辨率。
    """
    if width <= 0 or height <= 0:
        return step, step
    if width * height <= area:
        return max(step, width - width % step), max(step, height - height % step)
    scale = (area / (width * height)) ** 0.5
    tw = max(step, int(round(width * scale / step)) * step)
    th = max(step, int(round(height * scale / step)) * step)
    return tw, th


def assign_buckets(items, area: int, step: int = 64, min_side: int = 256, max_side: int = 2048,
                   native: bool = False):
    """Snap every image to the closed aspect-ratio list (coarse on purpose).

    ``native=True`` 时不做桶对齐，改成每张图按自己的宽高比缩到面积上限 —— 用于需要
    保留随机分辨率的实验。
    """
    if native:
        for item in items:
            if not item.width or not item.height:
                item.width, item.height = read_size(item.image)
            item.bucket = target_size(item.width, item.height, area, step)
        return items
    buckets = bucket_resolutions(area, step)
    ratios = [(w / h, (w, h)) for w, h in buckets]
    for item in items:
        if not item.width or not item.height:
            item.width, item.height = read_size(item.image)
        aspect = item.width / item.height
        item.bucket = min(ratios, key=lambda pair: abs(pair[0] - aspect))[1]
    return items


def summarise(items) -> dict:
    buckets: dict[tuple[int, int], int] = {}
    for item in items:
        buckets[item.bucket] = buckets.get(item.bucket, 0) + item.repeats
    styles = {"tag_caption": 0, "official": 0}
    for item in items:
        present = sidecar_styles(Path(item.image).with_suffix(""))
        if MOSS_STYLE_TAG in present:
            styles["tag_caption"] += 1
        if MOSS_STYLE_OFFICIAL in present:
            styles["official"] += 1
    return {
        "images": len(items),
        "with_caption": sum(1 for i in items if i.caption),
        "with_moss": sum(1 for i in items if i.caption_source == "moss"),
        "styles": styles,
        "buckets": {f"{w}x{h}": n for (w, h), n in sorted(buckets.items())},
    }
