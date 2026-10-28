# moved from app.py: L2621-2699
"""Cached thumbnails and the sample-image grid."""

from __future__ import annotations

from fastapi import APIRouter

from ..services.dataset import samples_list, thumb_response

router = APIRouter()


@router.get("/api/thumb")
def api_thumb(path: str, w: int = 384, workspace: str = "", v: str = ""):
    """Small cached WebP thumbnail; the original is only loaded in the modal."""
    return thumb_response(path, w, workspace, v)


@router.get("/api/samples")
def api_samples(dir: str = "", workspace: str = ""):
    return samples_list(dir, workspace)
