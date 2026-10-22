# moved from app.py: L1754-1809, L2149-2210, L2296-2334
"""LoRA export endpoints and the exported-file inventory."""

from __future__ import annotations

from fastapi import APIRouter, Request

from ..services.export import export_batch, export_one, loras_list

router = APIRouter()


@router.post("/api/workspaces/{name}/export-batch")
async def api_export_batch(name: str, request: Request):
    """Export a range of checkpoints at once, either into a directory or a zip."""
    return await export_batch(name, request)


@router.post("/api/workspaces/{name}/export")
async def api_export(name: str, request: Request):
    """Export one checkpoint of a finished run (kohya or diffusers layout)."""
    return await export_one(name, request)


@router.get("/api/loras")
def api_loras(workspace: str = "", run: str = "", include_state: bool = False):
    return loras_list(workspace, run, include_state)
