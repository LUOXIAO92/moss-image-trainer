# moved from app.py: L1380-1401
"""``GET /api/dataset``: dataset scan, buckets and preview thumbnails."""

from __future__ import annotations

from fastapi import APIRouter

from ..services.dataset import dataset_info

router = APIRouter()


@router.get("/api/dataset")
def api_dataset(path: str, max_area: int = 1024 * 1024, bucket_step: int = 64,
                source: str = "tag_caption"):
    return dataset_info(path, max_area, bucket_step, source)
