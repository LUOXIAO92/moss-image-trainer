# moved from app.py: L1404-1444, L2213-2275, L2278-2293
"""Workspace YAML config, run configs, VRAM presets and cache readiness."""

from __future__ import annotations

import os

import yaml
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from ...config import VRAM_PRESETS
from ..deps import _rel, _safe_name, _write_text
from ..services.cache_cmd import _cache_job, _cache_stop, _cache_summary
from ..services.workspace import (_config_from_data, _resolve_workspace, _run_config_default,
                                  _run_config_path, _run_config_snapshot, _run_is_live,
                                  _workspace_dir, _workspace_exists, _workspace_name,
                                  _workspace_runs_dir, _write_run_config)

router = APIRouter()


@router.get("/api/presets")
def api_presets():
    return {"presets": VRAM_PRESETS}


def _yaml_error(exc: yaml.YAMLError) -> JSONResponse:
    mark = getattr(exc, "problem_mark", None)
    where = f"第 {mark.line + 1} 行第 {mark.column + 1} 列" if mark else ""
    return JSONResponse(
        {"detail": f"YAML 语法错误 {where}: {getattr(exc, 'problem', exc)}".strip(),
         "line": mark.line + 1 if mark else None,
         "column": mark.column + 1 if mark else None},
        status_code=400,
    )


@router.post("/api/workspaces/{name}/runs")
async def api_runs_create(name: str, request: Request):
    """Create a run: its own complete ``config.yaml``, seeded once."""
    name = _workspace_name(name)
    if not _workspace_exists(name):
        raise HTTPException(404, f"工作区不存在：{name}")
    body = await request.json()
    run = _safe_name(body.get("name"), "run 名称")
    from_run = str(body.get("from") or "").strip()
    if from_run:
        from_run = _safe_name(from_run, "来源 run")
    target = os.path.join(_workspace_runs_dir(name), run)
    if os.path.exists(target):
        raise HTTPException(409, f"运行名已存在：{run}")
    text = _run_config_default(name, run, from_run)
    path = _write_run_config(name, run, text)
    return {"ok": True, "run": run, "path": _rel(path)}


@router.get("/api/workspaces/{name}/runs/{run}/config")
def api_run_config(name: str, run: str):
    """The config one run started with, from the run directory and nowhere else."""
    name = _workspace_name(name)
    run = _safe_name(run, "run 名称")
    text, path, _source = _run_config_snapshot(name, run)
    if not text:
        raise HTTPException(404, f"运行没有配置存档：{run}")
    return {"yaml": text, "path": path, "run": run}


@router.put("/api/workspaces/{name}/runs/{run}/config")
async def api_run_config_save(name: str, run: str, request: Request):
    """Edit a run's config. Frozen while the run trains, free afterwards."""
    name = _workspace_name(name)
    run = _safe_name(run, "run 名称")
    if not os.path.isfile(_run_config_path(name, run)):
        raise HTTPException(404, f"运行没有配置存档：{run}")
    if _run_is_live(name, run):
        raise HTTPException(409, "本次运行进行中，配置已冻结")
    body = await request.json()
    text = body.get("yaml") or ""
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        return _yaml_error(exc)
    if not isinstance(data, dict):
        raise HTTPException(400, "配置必须是一个 YAML 映射")
    # The run directory names the config; a stale name field would drift.
    data["name"] = run
    path = _run_config_path(name, run)
    _write_text(path, yaml.safe_dump(data, allow_unicode=True, sort_keys=False, width=10**9))
    return {"ok": True, "run": run, "path": _rel(path)}


@router.post("/api/cache/stop")
def api_cache_stop(workspace: str = ""):
    name = _resolve_workspace(workspace)
    return _cache_stop(_workspace_dir(name))


@router.get("/api/cache")
def api_cache(path: str = "", workspace: str = "", reg_path: str | None = None, run: str = ""):
    name = _resolve_workspace(workspace)
    data: dict = {}
    if run:
        text, _path, _source = _run_config_snapshot(name, _safe_name(run, "run 名称"))
        if text:
            data = yaml.safe_load(text) or {}
    config = _config_from_data(name, data)
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
    summary = _cache_summary(config)
    summary["job"] = _cache_job(_workspace_dir(name))
    return summary
