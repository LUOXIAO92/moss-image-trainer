"""尺寸规则的回归测试：每张图保留自己的宽高比，面积封顶，不再有固定桶。

    .venv/bin/python tests/test_sizing.py
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image  # noqa: E402

from moss_lora.data import (ALIGN, Item, assign_sizes, sizing_spec, summarise,  # noqa: E402
                            target_size)


def check_cap():
    cases = [
        (2000, 1000, 1024 * 1024),
        (4000, 3000, 4 * 1024 * 1024),      # 面积上限本来就不止 1M
        (999, 1333, 1024 * 1024),
        (3000, 400, 1024 * 1024),
    ]
    for width, height, area in cases:
        tw, th = target_size(width, height, area, ALIGN)
        assert tw * th <= area, f"{width}x{height}: {tw}x{th} 超出面积上限 {area}"
        assert tw % ALIGN == 0 and th % ALIGN == 0, (tw, th)
        # 相对误差：对齐到 16 对细长图的比例影响最大
        assert abs(tw / th - width / height) / (width / height) < 0.02, (tw / th, width / height)
        assert tw >= ALIGN and th >= ALIGN, (tw, th)
    print(f"area cap            : ok  {len(cases)} cases, never above the budget")


def check_no_upscale():
    assert target_size(512, 512, 1024 * 1024, ALIGN) == (512, 512)
    assert target_size(1000, 700, 1024 * 1024, ALIGN) == (992, 688)  # 只向下对齐
    assert target_size(0, 0, 1024 * 1024, ALIGN) == (ALIGN, ALIGN)
    print("no upscale          : ok  small images keep their size")


def check_per_image(tmp):
    wide = os.path.join(tmp, "wide.png")
    tall = os.path.join(tmp, "tall.png")
    Image.new("RGB", (1200, 800)).save(wide)
    Image.new("RGB", (800, 1200)).save(tall)
    items = [Item(image=wide, caption=""), Item(image=tall, caption="")]
    assign_sizes(items, 1024 * 1024)
    assert items[0].size == (1200, 800) and items[1].size == (800, 1200), [i.size for i in items]
    # 两张图的宽高比不同，尺寸就必须不同：没有吸附到同一个桶
    assert items[0].size != items[1].size
    print("per-image sizes     : ok  no shared canvas, no aspect snapping")


def check_summary():
    items = [Item(image="a.jpg", caption="x", width=100, height=100)]
    assign_sizes(items, 1024 * 1024)
    info = summarise(items)
    assert "buckets" not in info, info
    assert info["images"] == 1 and info["with_caption"] == 1
    print("summarise           : ok  counts only, no bucket table")


def check_spec():
    assert sizing_spec(1024 * 1024) != sizing_spec(4 * 1024 * 1024)
    assert sizing_spec(1024 * 1024)["mode"] == "native"
    print("sizing spec         : ok  area changes the fingerprint")


def check_guard(tmp):
    """A cache built by another geometry must be refused, not silently used."""
    from moss_lora.config import Config
    from moss_lora.trainer import Trainer

    config = Config()
    config.dataset.max_area = 4 * 1024 * 1024
    guard = Trainer._check_sizing
    expected = sizing_spec(config.dataset.max_area, ALIGN)

    class Fake:
        pass

    def run(index):
        fake = Fake()
        fake.config = config
        fake.index = index
        try:
            guard(fake)
        except SystemExit as exc:
            return str(exc)
        return ""

    assert run({"sizing": expected}) == "", "matching sizing was refused"
    stale = run({"sizing": {"mode": "grid", "area": 1024 * 1024, "align": 64}})
    assert "尺寸规格" in stale, stale
    missing = run({})
    assert "尺寸规格" in missing, "an index without the field must be refused too"
    print("sizing guard        : ok  mismatched and missing fingerprints both stop the run")


def check_builder_marks_stale(tmp):
    """The cache builder re-encodes latents when the geometry changed."""
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from moss_lora.cache import CacheBuilder

    data_dir = os.path.join(tmp, "imgs")
    os.makedirs(data_dir)
    Image.new("RGB", (600, 400)).save(os.path.join(data_dir, "one.png"))
    cache_dir = os.path.join(tmp, "cache")
    os.makedirs(cache_dir)
    with open(os.path.join(cache_dir, "index.json"), "w", encoding="utf-8") as handle:
        json.dump({"schema": 2, "items": [], "sizing": {"mode": "grid", "area": 1048576, "align": 64}}, handle)

    from moss_lora.config import Config

    config = Config()
    config.dataset.path = data_dir
    config.cache.dir = cache_dir
    builder = CacheBuilder(config, log=lambda message: None)
    assert builder.sizing_changed is True
    assert builder._pending("latent"), "latents were not scheduled for re-encoding"
    print("builder invalidation: ok  old geometry re-encodes latents only")


def main():
    tmp = tempfile.mkdtemp(prefix="moss-sizing-")
    try:
        check_cap()
        check_no_upscale()
        check_per_image(tmp)
        check_summary()
        check_spec()
        check_guard(tmp)
        check_builder_marks_stale(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("\nsizing checks passed")


if __name__ == "__main__":
    main()
