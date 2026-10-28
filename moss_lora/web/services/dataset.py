# moved from app.py: L1380-1401, L2584-2699
"""Dataset scans, cached thumbnails and the sample-image listing."""

from __future__ import annotations

import hashlib
import os
import re
import uuid

from fastapi import HTTPException
from fastapi.responses import FileResponse

from ..deps import ROOT, _rel
from .workspace import (_resolve_workspace, _workspace_dir, _workspace_root,
                        _workspace_runs_dir)
from ..errors import fail


def dataset_info(path: str, source: str = "tag_caption"):
    """Caption coverage and thumbnails for a dataset directory.

    Sizing happens when the cache is built (``max_area`` caps the pixel area),
    so this scan no longer reads every image header: nothing here needs a size.
    """
    from ...data import MOSS_CAPTION_SOURCES, scan, summarise

    if source not in MOSS_CAPTION_SOURCES:
        raise fail(400, "err.dataset.source_unknown", source=source)
    try:
        items = scan(path, moss_caption_source=source)
    except Exception as exc:
        raise HTTPException(400, str(exc))
    info = summarise(items)
    info["path"] = os.path.abspath(os.path.expanduser(path))
    info["source"] = source
    info["thumbnails"] = [i.image for i in items[:24]]
    return info

def _thumb_cache_dir(full: str, workspace: str = "") -> str:
    """Return the cache directory owned by the image's workspace/run.

    Sample images live under ``<workspace>/runs/<run>/...``; their thumbnails
    belong next to the run so deleting a run also deletes its cache.  Dataset
    previews and baseline images use the workspace-level ``.thumbs``.
    """
    requested = _resolve_workspace(workspace)
    root = os.path.realpath(_workspace_root())
    real = os.path.realpath(full)
    try:
        rel_root = os.path.relpath(real, root)
    except ValueError:
        rel_root = ".."
    if rel_root != ".." and not rel_root.startswith(".." + os.sep):
        parts = rel_root.split(os.sep)
        if parts and os.path.isdir(os.path.join(root, parts[0])):
            requested = parts[0]
    ws_dir = os.path.realpath(_workspace_dir(requested))
    runs_dir = os.path.realpath(os.path.join(ws_dir, "runs"))
    if real == runs_dir or real.startswith(runs_dir + os.sep):
        # Mirror /api/samples: the run is derived from the image's directory,
        # not the filename, so direct sample files and samples/ subdirs agree.
        rel = os.path.relpath(os.path.dirname(real), runs_dir)
        parts = [] if rel == "." else rel.split(os.sep)
        if "samples" in parts:
            run_parts = parts[:parts.index("samples")]
        elif len(parts) > 1:
            run_parts = parts[:-1]
        else:
            run_parts = parts
        run_parts = [part for part in run_parts if part not in ("", ".", "..")]
        if run_parts:
            return os.path.join(runs_dir, *run_parts, ".thumbs")
    return os.path.join(ws_dir, ".thumbs")

def _stat_key(path: str) -> tuple:
    stat = os.stat(path)
    return stat.st_mtime_ns, stat.st_size


def _render_thumbnail(full: str, out: str, width: int) -> None:
    """Decode, shrink and store one thumbnail, retrying if the source moved.

    A sample used to be written in place, so a poll could read it mid-write and
    PIL answered "image file is truncated". The trainer now publishes sample
    files with a rename, but a retry still covers a dataset image being copied
    in, and costs nothing when the file is stable.
    """
    from PIL import Image, ImageOps

    for attempt in (1, 2):
        before = _stat_key(full)
        temp = f"{out}.{uuid.uuid4().hex[:6]}.tmp"
        try:
            with Image.open(full) as image:
                image = ImageOps.exif_transpose(image)
                image.thumbnail((width, width), Image.Resampling.LANCZOS)
                if image.mode not in ("RGB", "RGBA"):
                    image = image.convert("RGB")
                image.save(temp, "WEBP", quality=82, method=4)
                os.replace(temp, out)
                return
        except Exception:
            if os.path.exists(temp):
                try:
                    os.remove(temp)
                except OSError:
                    pass
            if attempt == 2 or _stat_key(full) == before:
                raise


def thumb_response(path: str, w: int = 384, workspace: str = "", v: str = ""):
    """Small cached WebP thumbnail; the original is only loaded in the modal."""
    full = path if os.path.isabs(path) else os.path.abspath(os.path.join(ROOT, path))
    image_ext = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
    if not os.path.isfile(full) or os.path.splitext(full)[1].lower() not in image_ext:
        raise fail(404, "err.file.not_found")
    width = max(64, min(1024, int(w)))
    try:
        from PIL import Image  # noqa: F401  (fail fast when Pillow is missing)

        stat = os.stat(full)
        key = hashlib.sha1(
            f"{full}:{stat.st_mtime_ns}:{stat.st_size}:{width}".encode()
        ).hexdigest()
        thumb_dir = _thumb_cache_dir(full, workspace)
        os.makedirs(thumb_dir, exist_ok=True)
        out = os.path.join(thumb_dir, f"{key}.webp")
        if not os.path.exists(out):
            _render_thumbnail(full, out, width)
        # ``v`` is the page-load stamp the browser sends: within one page view
        # a thumbnail cannot change, so it may live in the cache instead of
        # being revalidated on every dashboard poll. Without it (a direct URL,
        # another client) keep the honest no-cache answer.
        cache = "private, max-age=86400, immutable" if v else "no-cache"
        return FileResponse(out, media_type="image/webp",
                            headers={"Cache-Control": cache})
    except Exception as exc:
        raise fail(404, "err.thumbnail.failed", error=str(exc))

def samples_list(dir: str = "", workspace: str = ""):
    scope = _resolve_workspace(workspace)
    base_default = _workspace_runs_dir(scope)
    if dir:
        if os.path.isabs(dir):
            roots = [dir]
        else:
            roots = [os.path.abspath(os.path.join(_workspace_dir(scope), dir))]
    else:
        roots = [base_default]
    out = []
    seen: set[str] = set()
    for base in roots:
        if not os.path.isdir(base):
            continue
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = [name for name in dirnames if not name.startswith(".")]
            for filename in sorted(filenames):
                if not filename.endswith(".png"):
                    continue
                path = os.path.join(dirpath, filename)
                real = os.path.realpath(path)
                if real in seen:
                    continue
                seen.add(real)
                rel = os.path.relpath(dirpath, base)
                parts = [] if rel == "." else rel.split(os.sep)
                if "samples" in parts:
                    run_parts = parts[:parts.index("samples")]
                else:
                    run_parts = parts[:-1] if len(parts) > 1 else parts
                run = "/".join(run_parts) if run_parts else os.path.basename(base)
                match = re.search(r"step[_-]?(\d+)", filename)
                out.append({
                    "path": _rel(path),
                    "run": f"{scope}/{run}",
                    "workspace": scope,
                    "step": int(match.group(1)) if match else None,
                    "mtime": os.path.getmtime(path),
                })
    # no cap: with several runs the grid needs every sample (was [:60], which
    # silently hid whole runs once the newest one filled the budget)
    return sorted(out, key=lambda x: -x["mtime"])
