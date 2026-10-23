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


def _plan_api_key(caption: dict, raw: str) -> tuple[tuple[str, dict] | None, str, str, str]:
    """填环境变量名就存名字；填密钥就只记在内存里，不落盘；密钥被拒直接报错。"""
    base_url = str(caption.get("base_url") or "").strip()
    raw = raw.strip()
    if not raw:
        return None, "", "", ""
    model = str(caption.get("model") or "").strip()

    def mismatch(ids: list[str]) -> str:
        if not ids or not model or model in ids:
            return ""
        return f"服务里没有模型 {model}（现有：{', '.join(ids[:4])}），跑的时候会失败"

    env_value = os.environ.get(raw, "").strip()
    if env_value:
        kind, detail, ids = _probe_service(base_url, env_value)
        if kind == "unauthorized":
            return ("err.settings.key_invalid", {"reason": detail}), "", "", ""
        note = "" if kind == "ok" else f"环境变量 {raw} 已保存，但服务暂时没连上（{detail}）"
        warn = "；".join(part for part in (note, mismatch(ids)) if part)
        return None, warn, raw, ""
    kind, detail, ids = _probe_service(base_url, raw)
    if kind == "unauthorized":
        return ("err.settings.key_rejected", {}), "", "", ""
    note = "填的是密钥本身，只在本次运行有效、不写进设置文件；想要长期生效请填环境变量名"
    if kind == "unreachable":
        note += f"（服务当前连不上：{detail}）"
    warn = "；".join(part for part in (note, mismatch(ids)) if part)
    return None, warn, "", raw

async def probe_model(request: Request):
    """Check one model path without loading it (files or a HF model directory)."""
    body = await request.json()
    kind = str(body.get("kind") or "").strip()
    raw = str(body.get("path") or "").strip()
    if kind not in ("dit", "text_encoder", "vae"):
        raise fail(400, "err.settings.model_kind")
    if not raw:
        return {"ok": False, "kind": kind, "path": "", "reason": "路径为空"}
    expanded = expand(raw)
    if "${" in expanded:
        return {"ok": False, "kind": kind, "path": expanded, "reason": "路径包含未展开的 ${MODELS}；请设置 MODELS 环境变量，或直接填绝对路径"}
    full = os.path.abspath(os.path.join(ROOT, os.path.expanduser(expanded))) if not os.path.isabs(expanded) else os.path.abspath(os.path.expanduser(expanded))
    exists = os.path.exists(full)
    result = {
        "ok": False, "kind": kind, "path": full, "exists": exists,
        "is_file": os.path.isfile(full), "is_dir": os.path.isdir(full),
        "size_mb": round(os.path.getsize(full) / 2**20, 1) if os.path.isfile(full) else None,
        "reason": "",
    }
    if not exists:
        result["reason"] = f"路径不存在：{full}"
        return result
    if kind == "text_encoder":
        if result["is_file"]:
            if not full.lower().endswith(".safetensors"):
                result["reason"] = "文本编码器文件必须是 .safetensors"
            else:
                result["ok"] = True
        elif result["is_dir"]:
            if not os.path.isfile(os.path.join(full, "config.json")):
                result["reason"] = "目录里没有 config.json；这不是一个 HuggingFace 模型目录"
            else:
                result["ok"] = True
        else:
            result["reason"] = "文本编码器必须是 HF 目录或 .safetensors 文件"
        return result
    if not result["is_file"]:
        result["reason"] = "这是一个目录；这里需要 .safetensors 文件"
    elif not full.lower().endswith(".safetensors"):
        result["reason"] = "模型文件必须是 .safetensors"
    else:
        result["ok"] = True
    return result
