# moved from app.py: L2337-2379
"""Directory listing and the read-only file server."""

from __future__ import annotations

import os

from fastapi import APIRouter
from fastapi.responses import FileResponse

from ..deps import ROOT
from ..services.workspace import _workspace_root
from ..errors import fail

router = APIRouter()


@router.get("/api/fs/list")
def api_fs_list(path: str = ""):
    """Directory listing for the export target picker (directories only)."""
    base = os.path.abspath(os.path.expanduser(path or _workspace_root()))
    if not os.path.isdir(base):
        # allow typing a not-yet-created directory: list its parent instead
        parent = os.path.dirname(base)
        if os.path.isdir(parent):
            return {"path": base, "parent": parent, "exists": False,
                    "entries": _fs_entries(parent)}
        raise fail(404, "err.dir.not_found", path=path)
    return {"path": base, "parent": os.path.dirname(base), "exists": True,
            "entries": _fs_entries(base)}

def _fs_entries(directory: str) -> list[dict]:
    entries = []
    try:
        names = sorted(os.listdir(directory))
    except OSError:
        return entries
    for name in names:
        if name.startswith("."):
            continue
        full = os.path.join(directory, name)
        try:
            is_dir = os.path.isdir(full)
            size = 0 if is_dir else os.path.getsize(full)
        except OSError:
            continue
        entries.append({"name": name, "path": full, "is_dir": is_dir, "size_mb": round(size / 2**20, 1)})
    return entries

@router.get("/api/lora/inspect")
def api_lora_inspect(path: str = ""):
    """Rank and alpha summary of a LoRA file, for the warm-start confirmation."""
    full = os.path.abspath(os.path.expanduser(path or ""))
    if not path or not os.path.isfile(full):
        raise fail(404, "err.lora.file_missing", path=path)
    try:
        from safetensors.torch import load_file

        from ...lora_file import lora_alphas, lora_modules

        checkpoint = load_file(full)
        modules = lora_modules(checkpoint)
    except Exception as exc:  # unreadable, not safetensors, corrupt header
        raise fail(400, "err.lora.unreadable", path=path, error=str(exc)) from None
    if not modules:
        raise fail(400, "err.lora.no_modules", path=path)
    alphas = lora_alphas(checkpoint)
    ranks = sorted({int(down.shape[0]) for down, _up in modules.values()})
    alpha_values = sorted({round(float(alphas.get(name, down.shape[0])), 6)
                           for name, (down, _up) in modules.items()})
    return {"path": full, "modules": len(modules), "ranks": ranks, "alphas": alpha_values}


@router.get("/api/files")
def api_files(path: str):
    """Serve images and LoRAs from anywhere on disk (local tool, read-only)."""
    full = path if os.path.isabs(path) else os.path.abspath(os.path.join(ROOT, path))
    allowed = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".safetensors", ".yaml", ".txt",
               ".log", ".json", ".zip")
    if not os.path.isfile(full) or os.path.splitext(full)[1].lower() not in allowed:
        raise fail(404, "err.file.not_found")
    return FileResponse(full, filename=os.path.basename(full))
