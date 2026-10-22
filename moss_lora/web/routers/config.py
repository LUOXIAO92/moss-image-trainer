# moved from app.py: L1404-1444, L2213-2275, L2278-2293
"""Workspace YAML config, VRAM presets, saved configs and cache readiness."""

from __future__ import annotations

import os

import yaml
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from ...config import VRAM_PRESETS
from ..deps import _rel, _safe_name, _write_text
from ..services.cache_cmd import _cache_summary
from ..services.workspace import (_config_entry, _ensure_workspace_dirs, _resolve_workspace,
                                  _workspace_config, _workspace_config_path,
                                  _workspace_configs_dir, _workspace_dir, _workspace_exists,
                                  _workspace_name)

router = APIRouter()


@router.get("/api/config")
def api_config(workspace: str = ""):
    name = _resolve_workspace(workspace)
    full = _workspace_config_path(name)
    if not os.path.isfile(full):
        raise HTTPException(404, f"工作区配置不存在：{name}")
    with open(full, encoding="utf-8") as handle:
        return {"path": _rel(full), "workspace": name, "yaml": handle.read()}


@router.get("/api/presets")
def api_presets():
    return {"presets": VRAM_PRESETS}


@router.post("/api/config")
async def api_save_config(request: Request):
    body = await request.json()
    text = body.get("yaml", "")
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        mark = getattr(exc, "problem_mark", None)
        where = f"第 {mark.line + 1} 行第 {mark.column + 1} 列" if mark else ""
        return JSONResponse(
            {"detail": f"YAML 语法错误 {where}: {getattr(exc, 'problem', exc)}".strip(),
             "line": mark.line + 1 if mark else None,
             "column": mark.column + 1 if mark else None},
            status_code=400,
        )
    workspace = _resolve_workspace(body.get("workspace"))
    try:
        config = _workspace_config(workspace, data or {})
        problems = config.validate()
    except Exception as exc:
        raise HTTPException(400, f"配置结构错误：{exc}")
    if problems and body.get("strict"):
        return JSONResponse({"problems": problems}, status_code=400)
    path = _workspace_config_path(workspace)
    _write_text(path, text)
    return {"path": _rel(path), "workspace": workspace, "problems": problems}

@router.get("/api/workspaces/{name}/configs")
def api_configs_list(name: str):
    name = _workspace_name(name)
    if not _workspace_exists(name):
        raise HTTPException(404, f"工作区不存在：{name}")
    directory = _workspace_configs_dir(name)
    entries = []
    if os.path.isdir(directory):
        entries = [_config_entry(os.path.join(directory, f))
                   for f in sorted(os.listdir(directory)) if f.endswith((".yaml", ".yml"))]
    return {"configs": entries, "workspace": name,
            "default": _workspace_config_path(name)}


@router.get("/api/workspaces/{name}/configs/{config}")
def api_configs_get(name: str, config: str):
    name = _workspace_name(name)
    config = _safe_name(config, "配置名称")
    path = os.path.join(_workspace_configs_dir(name), f"{config}.yaml")
    if not os.path.isfile(path):
        raise HTTPException(404, f"配置不存在：{config}")
    with open(path, encoding="utf-8") as handle:
        return {"name": config, "yaml": handle.read(), "entry": _config_entry(path)}


@router.post("/api/workspaces/{name}/configs")
async def api_configs_save(name: str, request: Request):
    """Create or overwrite ``<ws>/configs/<config>.yaml``."""
    name = _workspace_name(name)
    if not _workspace_exists(name):
        raise HTTPException(404, f"工作区不存在：{name}")
    body = await request.json()
    config = _safe_name(body.get("name"), "配置名称")
    text = body.get("yaml") or ""
    if not text.strip():
        raise HTTPException(400, "yaml required")
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise HTTPException(400, f"YAML 语法错误: {exc}")
    if not isinstance(data, dict):
        raise HTTPException(400, "配置必须是一个 YAML 映射")
    data.setdefault("name", config)
    directory = _workspace_configs_dir(name)
    os.makedirs(directory, exist_ok=True)
    _ensure_workspace_dirs(name)
    path = os.path.join(directory, f"{config}.yaml")
    if os.path.exists(path) and not body.get("overwrite", True):
        raise HTTPException(409, f"配置已存在：{config}")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(yaml.safe_dump(data, allow_unicode=True, sort_keys=False))
    return {"ok": True, "name": config, "entry": _config_entry(path)}


@router.delete("/api/workspaces/{name}/configs/{config}")
def api_configs_delete(name: str, config: str):
    name = _workspace_name(name)
    config = _safe_name(config, "配置名称")
    path = os.path.join(_workspace_configs_dir(name), f"{config}.yaml")
    if not os.path.isfile(path):
        raise HTTPException(404, f"配置不存在：{config}")
    os.remove(path)
    return {"ok": True}

@router.get("/api/cache")
def api_cache(path: str = "", workspace: str = "", reg_path: str | None = None):
    name = _resolve_workspace(workspace)
    config = _workspace_config(name)
    if path:
        config.cache.dir = (
            path if os.path.isabs(path)
            else os.path.abspath(os.path.join(_workspace_dir(name), path))
        )
    if reg_path is not None:
        raw_reg = reg_path.strip()
        config.cache.reg_dir = (
            raw_reg if os.path.isabs(raw_reg)
            else os.path.abspath(os.path.join(_workspace_dir(name), raw_reg))
        ) if raw_reg else ""
    return _cache_summary(config)
