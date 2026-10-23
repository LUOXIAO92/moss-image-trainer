# moved from app.py: L1380-1401
"""``GET /api/dataset``: dataset scan, caption coverage and preview thumbnails."""

from __future__ import annotations

from fastapi import APIRouter

from ..services.dataset import dataset_info

router = APIRouter()


@router.get("/api/dataset")
def api_dataset(path: str, source: str = "tag_caption"):
    return dataset_info(path, source)
