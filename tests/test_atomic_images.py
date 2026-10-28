"""Sample images appear whole or not at all.

    .venv/bin/python tests/test_atomic_images.py

``Image.save(path)`` writes in place: the name exists while the bytes are still
arriving, and a reader that opens it gets ``image file is truncated``. The
dashboard lists ``samples/`` and asks for thumbnails as soon as a name appears,
so the writer now saves through a temporary sibling and renames it into place.
"""

from __future__ import annotations

import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image  # noqa: E402

from moss_lora import runtime  # noqa: E402

TMP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".tmp")


def scratch(name: str) -> str:
    path = os.path.join(TMP, name)
    shutil.rmtree(path, ignore_errors=True)
    os.makedirs(path, exist_ok=True)
    return path


def leftovers(directory: str) -> list[str]:
    return [name for name in os.listdir(directory) if name.endswith(".tmp")]


def check_publishes_with_a_rename():
    tmp = scratch("atomic-publish")
    target = os.path.join(tmp, "samples", "step003000_0.png")
    seen = {}

    class Probe:
        def save(self, path, format=None):
            seen["path"] = path
            seen["format"] = format
            Image.new("RGB", (4, 4), (10, 20, 30)).save(path, format)

    runtime.save_image_atomic(target, Probe())
    assert seen["path"] != target and seen["path"].endswith(".tmp"), "the writer must not touch the final name"
    assert seen["format"] == "PNG"
    assert os.path.isfile(target)
    assert not leftovers(os.path.dirname(target))
    with Image.open(target) as image:
        assert image.size == (4, 4)
    print("publish: temp file renamed, no leftovers")


def check_failure_leaves_nothing():
    tmp = scratch("atomic-failure")
    target = os.path.join(tmp, "samples", "step003000_0.png")

    class Broken:
        def save(self, path, format=None):
            with open(path, "wb") as handle:
                handle.write(b"partial")
            raise OSError("disk full")

    try:
        runtime.save_image_atomic(target, Broken())
    except OSError:
        pass
    else:
        raise AssertionError("the write should have raised")
    assert not os.path.exists(target), "a failed write must not publish a name"
    assert not leftovers(os.path.dirname(target))
    print("failure: no target, no temp file")


def main() -> int:
    check_publishes_with_a_rename()
    check_failure_leaves_nothing()
    print("atomic image checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
