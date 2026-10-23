# moved from app.py: L1064-1142
"""Language packs and the UI language switch."""

from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response

from ... import i18n as i18n_pack
from ..deps import _settings_snapshot
from ..services.settings_svc import _save_language
from ..errors import fail

router = APIRouter()


@router.get("/api/i18n")
def api_i18n_languages():
    settings, _ = _settings_snapshot()
    current = i18n_pack.resolve((settings.get("ui") or {}).get("language"))
    return {
        "languages": list(i18n_pack.pack_languages().values()),
        "current": current,
        "defaults": list(i18n_pack.BUILTIN_CODES),
        "default": i18n_pack.DEFAULT_CODE,
    }


@router.get("/api/i18n/template")
def api_i18n_template(base: str = i18n_pack.DEFAULT_CODE):
    base = base if base in i18n_pack.pack_languages() else i18n_pack.DEFAULT_CODE
    payload = i18n_pack.template(base)
    body = json.dumps(payload, ensure_ascii=False, indent=1)
    return Response(body, media_type="application/json; charset=utf-8",
                    headers={"Content-Disposition": f'attachment; filename="i18n-template-{base}.json"'})


@router.get("/api/i18n/{code}")
def api_i18n_pack(code: str):
    try:
        wanted = i18n_pack.normalize_code(code)
    except i18n_pack.LanguageError as exc:
        raise fail(400, exc.code, **exc.params) from None
    users = i18n_pack.entries(i18n_pack.load_user(wanted))
    values = i18n_pack.merged(wanted)
    return {
        "code": wanted,
        "name": i18n_pack.pack_languages().get(wanted, {}).get("name", wanted),
        "entries": values,
        "user_keys": len(users),
        "builtin": wanted in i18n_pack.BUILTIN_CODES,
    }


@router.put("/api/i18n/{code}")
async def api_i18n_save(code: str, request: Request):
    try:
        wanted = i18n_pack.normalize_code(code)
    except i18n_pack.LanguageError as exc:
        raise fail(400, exc.code, **exc.params) from None
    body = await request.json()
    payload = body.get("pack") if isinstance(body, dict) and isinstance(body.get("pack"), dict) else body
    try:
        saved = i18n_pack.save_pack(wanted, payload)
    except i18n_pack.LanguageError as exc:
        raise fail(400, exc.code, **exc.params) from None
    return {"ok": True, **saved}


@router.delete("/api/i18n/{code}")
def api_i18n_delete(code: str):
    try:
        wanted = i18n_pack.normalize_code(code)
    except i18n_pack.LanguageError as exc:
        raise fail(400, exc.code, **exc.params) from None
    result = i18n_pack.delete_pack(wanted)
    settings, _ = _settings_snapshot()
    current = i18n_pack.resolve((settings.get("ui") or {}).get("language"))
    if current != (settings.get("ui") or {}).get("language"):
        _save_language(current)
    return {"ok": True, **result, "current": current}


@router.post("/api/i18n/current")
async def api_i18n_current(request: Request):
    body = await request.json()
    try:
        code = i18n_pack.normalize_code(body.get("code"))
    except (i18n_pack.LanguageError, AttributeError) as exc:
        if isinstance(exc, i18n_pack.LanguageError):
            raise fail(400, exc.code, **exc.params) from None
        raise fail(400, "err.lang.code_invalid") from None
    known = i18n_pack.pack_languages()
    if code not in known:
        raise fail(404, "err.lang.unknown", code=code)
    _save_language(code)
    return {"ok": True, "current": code, "name": known[code]["name"]}
