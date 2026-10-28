"""A thumbnail of an unfinished image answers 404, not an ASGI failure.

    .venv/bin/python tests/test_thumbnail_race.py

Two contracts meet here. The encoder publishes sample files with a rename, so a
listed name always holds complete bytes; and when a file is still changing (a
dataset image being copied in), the reader retries while the source moves and
then reports a plain 404 whose body is JSON -- never an exception object that
breaks response serialization.
"""

from __future__ import annotations

import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import HTTPException  # noqa: E402
from PIL import Image  # noqa: E402

from moss_lora.web.services import dataset  # noqa: E402

TMP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".tmp")


def scratch(name: str) -> str:
    path = os.path.join(TMP, name)
    shutil.rmtree(path, ignore_errors=True)
    os.makedirs(path, exist_ok=True)
    return path


def write_png(path: str, size: tuple = (8, 8)) -> None:
    Image.new("RGB", size, (200, 30, 30)).save(path, "PNG")


def truncate(path: str) -> None:
    with open(path, "rb") as handle:
        data = handle.read()
    with open(path, "wb") as handle:
        handle.write(data[: len(data) // 2])


def with_cache_dir(fn):
    """Point thumbnails at a scratch cache so no workspace is needed."""
    original = dataset._thumb_cache_dir
    cache = scratch("thumbs-cache")
    dataset._thumb_cache_dir = lambda full, workspace="": cache
    try:
        return fn(cache)
    finally:
        dataset._thumb_cache_dir = original


def check_truncated_file_is_a_404():
    tmp = scratch("thumb-truncated")
    path = os.path.join(tmp, "step003000_0.png")
    write_png(path)
    truncate(path)

    def run(_cache):
        try:
            dataset.thumb_response(path, 220)
        except HTTPException as exc:
            return exc
        raise AssertionError("a truncated image must not produce a thumbnail")

    exc = with_cache_dir(run)
    assert exc.status_code == 404
    body = json.dumps({"detail": exc.detail})
    assert "err.thumbnail.failed" in body
    print("truncated image: 404 with a serializable body")


def check_complete_file_is_cached():
    tmp = scratch("thumb-complete")
    path = os.path.join(tmp, "step003000_0.png")
    write_png(path)

    def run(cache):
        response = dataset.thumb_response(path, 220)
        assert os.path.basename(response.path).endswith(".webp")
        assert os.path.isdir(cache) and os.listdir(cache)
        return response

    response = with_cache_dir(run)
    assert response.headers["cache-control"] == "no-cache", "a bare URL stays honest"
    print("complete image: webp cached, no-cache without a stamp")


def check_page_stamp_allows_caching():
    tmp = scratch("thumb-stamped")
    path = os.path.join(tmp, "step003000_0.png")
    write_png(path)

    response = with_cache_dir(lambda _cache: dataset.thumb_response(path, 220, "", "1793167030518"))
    assert "immutable" in response.headers["cache-control"]
    assert "max-age" in response.headers["cache-control"]
    print("page stamp: the browser may keep the thumbnail")


def check_retry_follows_a_moving_file():
    tmp = scratch("thumb-moving")
    path = os.path.join(tmp, "copy.png")
    write_png(path)
    truncate(path)
    calls = {"n": 0}
    original = dataset._stat_key

    def moving(_target):
        calls["n"] += 1
        return (calls["n"], 1)

    dataset._stat_key = moving
    try:
        try:
            dataset._render_thumbnail(path, os.path.join(tmp, "out.webp"), 64)
        except Exception:
            pass
        else:
            raise AssertionError("a truncated moving source should not produce a thumbnail")
    finally:
        dataset._stat_key = original
    # attempt one: stat before, stat after it failed and the file had moved;
    # attempt two: stat before, then it gives up without a third comparison.
    assert calls["n"] == 3, f"one retry then give up, got {calls['n']} stat calls"
    print("moving source: retried once, then reported")


def main() -> int:
    check_truncated_file_is_a_404()
    check_complete_file_is_cached()
    check_page_stamp_allows_caching()
    check_retry_follows_a_moving_file()
    print("thumbnail checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
