# moved from app.py: L1156-1158
"""``GET /api/system``: interpreter, GPU and model-path snapshot."""

from __future__ import annotations

from fastapi import APIRouter

from ..services.system import _system

router = APIRouter()


@router.get("/api/system")
def api_system():
    return _system()
