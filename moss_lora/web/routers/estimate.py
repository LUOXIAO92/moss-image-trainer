"""``POST /api/estimate``: peak VRAM for the config currently in the form.

Read-only and side-effect free: the YAML is parsed, the preset is filled in the
way a job would fill it, and the arithmetic in :mod:`moss_lora.memory` answers.
Nothing is written and no model is loaded, so the training page can refresh the
number while the user edits.
"""

from __future__ import annotations

import yaml
from fastapi import APIRouter, Request

from ...config import apply_preset
from ...memory import estimate_vram
from ..errors import fail
from ..services.system import gpu_total_gb
from ..services.workspace import _config_from_data, _resolve_workspace

router = APIRouter()


@router.post("/api/estimate")
async def api_estimate(request: Request):
    body = await request.json()
    text = body.get("yaml") or ""
    if not text.strip():
        raise fail(400, "err.yaml.empty")
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        mark = getattr(exc, "problem_mark", None)
        problem = str(getattr(exc, "problem", exc))
        if mark:
            raise fail(400, "err.yaml.syntax", line=mark.line + 1, column=mark.column + 1,
                       problem=problem)
        raise fail(400, "err.yaml.syntax_plain", problem=problem)
    if not isinstance(data, dict):
        raise fail(400, "err.yaml.not_mapping")

    workspace = _resolve_workspace(body.get("workspace"))
    try:
        config = _config_from_data(workspace, data)
    except Exception as exc:  # malformed types are a bad request, not a server fault
        raise fail(400, "err.config.structure", error=exc)
    # Same resolution a job gets, so a preset looks identical in the estimate and
    # in the run it would start. The capacity comes from nvidia-smi rather than
    # detect_vram_gb(), because the latter would create a CUDA context in the web
    # process and hold VRAM that the job it is predicting for needs.
    apply_preset(config, gpu_total_gb(), detect=False)
    return estimate_vram(config)
