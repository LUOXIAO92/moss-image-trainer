# moved from app.py: L1161-1172, L1237-1288, L1291-1335
"""Settings read/write and model-path probing."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from .. import deps
from ... import settings as runtime_settings
from ..deps import SETTINGS_REL, _settings_snapshot
from ..services.settings_svc import (_blank_default_prompts, _caption_prompt_defaults,
                                     _caption_service_view, _plan_api_key, probe_model)
from ..errors import error_detail, fail

router = APIRouter()


@router.get("/api/settings")
def api_settings_get():
    settings, resolved = _settings_snapshot()
    return {
        "path": SETTINGS_REL,
        "settings": settings,
        "resolved": resolved,
        "status": runtime_settings.settings_status(settings),
        "caption_prompts": _caption_prompt_defaults(),
        "caption_service": _caption_service_view(settings),
        "first_run": deps.FIRST_RUN,
    }


@router.post("/api/settings")
async def api_settings_post(request: Request):
    body = await request.json()
    payload = body.get("settings") if isinstance(body, dict) and isinstance(body.get("settings"), dict) else body
    if not isinstance(payload, dict):
        raise fail(400, "err.settings.not_mapping")
    submitted_caption = payload.get("caption") if isinstance(payload.get("caption"), dict) else {}
    # 只提交一部分（例如设置页只改路径）时，保留其它段（提示词生成、运行时）不动
    settings_snapshot, _ = _settings_snapshot()
    payload = runtime_settings.merge_settings(settings_snapshot, payload)
    _blank_default_prompts(payload)
    key_error = None
    key_notes: list = []
    stored_key = None
    literal_key = ""
    if "api_key" in submitted_caption:
        # 探测服务是阻塞 IO，丢到线程里，别卡住事件循环
        key_error, key_notes, stored_key, literal_key = await asyncio.to_thread(
            _plan_api_key, payload.get("caption") or {}, str(submitted_caption.get("api_key") or ""))
        if key_error:
            code, params = key_error
            raise fail(400, code, **params)
        payload["caption"]["api_key"] = stored_key
    errors, warnings, status = runtime_settings.validate_settings(payload)
    if errors:
        return JSONResponse(
            {
                # Field-keyed detail so the browser can show each problem next to
                # the input it belongs to, in the reader's language.
                "detail": error_detail("err.settings.invalid"),
                "errors": {field: item.detail() for field, item in errors.items()},
                "warnings": warnings,
                "status": runtime_settings.settings_status(payload),
            },
            status_code=400,
        )
    if key_notes:
        warnings = list(warnings) + [item.detail() for item in key_notes]
    with deps._SETTINGS_LOCK:
        if stored_key is not None:
            deps._LITERAL_API_KEY = literal_key
        saved = runtime_settings.save_settings(payload)
        applied = runtime_settings.apply_environment(saved)
        deps.SETTINGS = saved
        deps.RESOLVED = applied
        deps.FIRST_RUN = False
    return {
        "ok": True,
        "path": SETTINGS_REL,
        "settings": saved,
        "resolved": applied,
        "status": runtime_settings.settings_status(saved),
        "caption_service": _caption_service_view(saved),
        "warnings": warnings,
    }


@router.post("/api/models/probe")
async def api_model_probe(request: Request):
    """Check one model path without loading it (files or a HF model directory)."""
    return await probe_model(request)
