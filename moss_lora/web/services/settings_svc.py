# moved from app.py: L1145-1154, L1175-1228, L1291-1335
"""Settings persistence, the settings-page views and model-path probing."""

from __future__ import annotations

import copy
import os

from fastapi import Request

from .. import deps
from ... import settings as runtime_settings
from ...config import expand
from ..deps import ROOT
from .caption import _caption_prompt_defaults, _probe_service, resolve_api_key
from ..errors import fail
from ...problems import Problem, problem


def _save_language(code: str) -> None:
    with deps._SETTINGS_LOCK:
        payload = copy.deepcopy(deps.SETTINGS)
        payload.setdefault("ui", {})["language"] = code
        saved = runtime_settings.save_settings(payload)
        applied = runtime_settings.apply_environment(saved)
        deps.SETTINGS = saved
        deps.RESOLVED = applied


def _blank_default_prompts(payload: dict) -> None:
    """设置页会把默认模板预填进输入框；和内置模板一字不差就存空串，别把 3 KB 提示词写进配置。"""
    caption = payload.get("caption")
    if not isinstance(caption, dict):
        return
    defaults = _caption_prompt_defaults()
    for field, key in (("prompt_tags", "tags"), ("prompt_natural", "natural")):
        value = str(caption.get(field) or "").strip()
        if value and value == defaults[key].strip():
            caption[field] = ""


def _caption_service_view(settings: dict) -> dict:
    caption = settings.get("caption") or {}
    key, source = resolve_api_key(caption)
    return {
        "base_url": str(caption.get("base_url") or ""),
        "model": str(caption.get("model") or ""),
        "api_key": str(caption.get("api_key") or ""),   # 只有环境变量名会出现在这里
        "has_key": bool(key),
        "key_source": source,
        "key_saved": source == "env",
    }


def _plan_api_key(caption: dict, raw: str) -> tuple[tuple[str, dict] | None, list[Problem], str, str]:
    """填环境变量名就存名字；填密钥就只记在内存里，不落盘；密钥被拒直接报错。

    The notes come back as problems, not sentences: the settings page shows them
    under the key field, so they have to follow the reader's language.
    """
    base_url = str(caption.get("base_url") or "").strip()
    raw = raw.strip()
    if not raw:
        return None, [], "", ""
    model = str(caption.get("model") or "").strip()

    def mismatch(ids: list[str]) -> list[Problem]:
        if not ids or not model or model in ids:
            return []
        return [problem("err.settings.model_missing", model=model, ids=", ".join(ids[:4]))]

    env_value = os.environ.get(raw, "").strip()
    if env_value:
        kind, detail, ids = _probe_service(base_url, env_value)
        if kind == "unauthorized":
            return ("err.settings.key_invalid", {"reason": detail}), [], "", ""
        if kind == "ok":
            notes = []
        elif kind == "no_base_url":
            notes = [problem("err.caption.no_endpoint")]
        else:
            notes = [problem("err.settings.key_env_unreachable", name=raw, reason=detail)]
        return None, notes + mismatch(ids), raw, ""
    kind, detail, ids = _probe_service(base_url, raw)
    if kind == "unauthorized":
        return ("err.settings.key_rejected", {}), [], "", ""
    notes = [problem("err.settings.key_literal_note")]
    if kind == "no_base_url":
        notes.append(problem("err.caption.no_endpoint"))
    elif kind == "unreachable":
        notes.append(problem("err.settings.key_unreachable", reason=detail))
    return None, notes + mismatch(ids), "", raw

def _reject(result: dict, code: str, **params) -> dict:
    """Mark a probe result as failed with a code the settings page can translate.

    ``reason`` stays as the rendered Chinese so a client that ignores the code
    still shows a sentence; ``reason_detail`` is what the UI actually renders.
    """
    item = problem(code, **params)
    result.update({"ok": False, "reason": item.render(), "reason_detail": item.detail()})
    return result


async def probe_model(request: Request):
    """Check one model path without loading it (files or a HF model directory)."""
    body = await request.json()
    kind = str(body.get("kind") or "").strip()
    raw = str(body.get("path") or "").strip()
    if kind not in ("dit", "text_encoder", "vae"):
        raise fail(400, "err.settings.model_kind")
    if not raw:
        return _reject({"kind": kind, "path": ""}, "err.settings.path_empty")
    expanded = expand(raw)
    if "${" in expanded:
        return _reject({"kind": kind, "path": expanded}, "err.settings.path_variable")
    full = os.path.abspath(os.path.join(ROOT, os.path.expanduser(expanded))) if not os.path.isabs(expanded) else os.path.abspath(os.path.expanduser(expanded))
    exists = os.path.exists(full)
    result = {
        "ok": False, "kind": kind, "path": full, "exists": exists,
        "is_file": os.path.isfile(full), "is_dir": os.path.isdir(full),
        "size_mb": round(os.path.getsize(full) / 2**20, 1) if os.path.isfile(full) else None,
        "reason": "",
    }
    if not exists:
        return _reject(result, "err.settings.path_missing", path=full)
    if kind == "text_encoder":
        if result["is_file"]:
            if not full.lower().endswith(".safetensors"):
                _reject(result, "err.settings.te_not_safetensors")
            else:
                result["ok"] = True
        elif result["is_dir"]:
            if not os.path.isfile(os.path.join(full, "config.json")):
                _reject(result, "err.settings.te_no_config")
            else:
                result["ok"] = True
        else:
            _reject(result, "err.settings.te_kind")
        return result
    if not result["is_file"]:
        _reject(result, "err.settings.file_expected")
    elif not full.lower().endswith(".safetensors"):
        _reject(result, "err.settings.not_safetensors")
    else:
        result["ok"] = True
    return result
