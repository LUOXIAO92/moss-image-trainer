# moved from app.py: L1404-1444, L2213-2275, L2278-2293
"""Workspace YAML config, run configs, VRAM presets and cache readiness."""

from __future__ import annotations

import os

import yaml
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from ...config import VRAM_PRESETS
from ..deps import _rel, _safe_name, _write_text
from ..services.cache_cmd import _cache_job, _cache_stop, _cache_summary
from ..services.workspace import (_config_from_data, _resolve_workspace, _run_config_default,
                                  _run_config_path, _run_config_snapshot, _run_is_live,
                                  _workspace_dir, _workspace_exists, _workspace_name,
                                  _workspace_runs_dir, _write_run_config)
from ..errors import error_detail, fail

router = APIRouter()


@router.get("/api/presets")
def api_presets():
    return {"presets": VRAM_PRESETS}


def _yaml_error(exc: yaml.YAMLError) -> JSONResponse:
    mark = getattr(exc, "problem_mark", None)
    problem = str(getattr(exc, "problem", exc))
    if mark:
        detail = error_detail("err.yaml.syntax", line=mark.line + 1, column=mark.column + 1,
                              problem=problem)
    else:
        detail = error_detail("err.yaml.syntax_plain", problem=problem)
    return JSONResponse(
        {"detail": detail,
         "line": mark.line + 1 if mark else None,
         "column": mark.column + 1 if mark else None},
        status_code=400,
    )


@router.post("/api/workspaces/{name}/runs")
async def api_runs_create(name: str, request: Request):
    """Create a run: its own complete ``config.yaml``, seeded once."""
    name = _workspace_name(name)
    if not _workspace_exists(name):
        raise fail(404, "err.workspace.not_found", name=name)
    body = await request.json()
    run = _safe_name(body.get("name"), "run")
    from_run = str(body.get("from") or "").strip()
    if from_run:
        from_run = _safe_name(from_run, "run")
    target = os.path.join(_workspace_runs_dir(name), run)
    if os.path.exists(target):
        raise fail(409, "err.run.exists", run=run)
    text = _run_config_default(name, run, from_run)
    path = _write_run_config(name, run, text)
    return {"ok": True, "run": run, "path": _rel(path)}


@router.get("/api/workspaces/{name}/runs/{run}/config")
def api_run_config(name: str, run: str):
    """The config one run started with, from the run directory and nowhere else."""
    name = _workspace_name(name)
    run = _safe_name(run, "run")
    text, path, _source = _run_config_snapshot(name, run)
    if not text:
        raise fail(404, "err.run.no_config", run=run)
    return {"yaml": text, "path": path, "run": run}


@router.put("/api/workspaces/{name}/runs/{run}/config")
async def api_run_config_save(name: str, run: str, request: Request):
    """Edit a run's config. Frozen while the run trains, free afterwards."""
    name = _workspace_name(name)
    run = _safe_name(run, "run")
    if not os.path.isfile(_run_config_path(name, run)):
        raise fail(404, "err.run.no_config", run=run)
    if _run_is_live(name, run):
        raise fail(409, "err.run.frozen")
    body = await request.json()
    text = body.get("yaml") or ""
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        return _yaml_error(exc)
    if not isinstance(data, dict):
        raise fail(400, "err.yaml.not_mapping")
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
        text, _path, _source = _run_config_snapshot(name, _safe_name(run, "run"))
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
