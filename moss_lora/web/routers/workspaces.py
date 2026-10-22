# moved from app.py: L1338-1377, L1826-1999, L2067-2146, L2382-2449
"""Workspace CRUD, disk usage, run/process control, layout and dashboard."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time

import yaml
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from ..deps import RESERVED_WORKSPACES, _rel, _safe_name, _write_text
from ..jobs import JOBS
from ..services.system import (_is_cli_train_process, _parse_metrics, _run_samples)
from ..services.workspace import (_dir_usage, _has_live_cli_training, _resolve_workspace,
                                  _workspace_config_path, _workspace_config_text,
                                  _workspace_dir, _workspace_exists, _workspace_info,
                                  _workspace_logs_dir, _workspace_name, _workspace_root,
                                  _workspace_run_dirs, _workspace_run_list,
                                  _workspace_runs_dir, _workspaces_payload)

router = APIRouter()


@router.get("/api/workspaces")
def api_workspaces():
    return _workspaces_payload()


@router.post("/api/workspaces")
async def api_workspaces_create(request: Request):
    body = await request.json()
    name = _workspace_name(body.get("name"))
    if _workspace_exists(name):
        return JSONResponse({"detail": f"工作区已存在：{name}"}, status_code=409)
    supplied = body.get("yaml")
    if supplied is not None:
        if not isinstance(supplied, str) or not supplied.strip():
            raise HTTPException(400, "yaml 不能为空")
        try:
            data = yaml.safe_load(supplied)
        except yaml.YAMLError as exc:
            raise HTTPException(400, f"YAML 语法错误：{getattr(exc, 'problem', exc)}")
        if not isinstance(data, dict):
            raise HTTPException(400, "配置必须是一个 YAML 映射")
        text = supplied
    else:
        text = _workspace_config_text()
    root = _workspace_root()
    ws = _workspace_dir(name)
    os.makedirs(os.path.join(ws, "train_cache"), exist_ok=True)
    os.makedirs(os.path.join(ws, "reg_cache"), exist_ok=True)
    os.makedirs(os.path.join(ws, "runs"), exist_ok=True)
    os.makedirs(os.path.join(ws, "logs"), exist_ok=True)
    _write_text(_workspace_config_path(name), text)
    return {"ok": True, "workspace": _workspace_info(name), **_workspaces_payload()}


@router.get("/api/workspaces/{name}/runs")
def api_workspace_runs(name: str):
    name = _workspace_name(name)
    if not _workspace_exists(name):
        raise HTTPException(404, f"工作区不存在：{name}")
    return {"workspace": name, "runs": _workspace_run_list(name)}

@router.get("/api/workspaces/{name}/usage")
def api_workspace_usage(name: str):
    """Size breakdown for the workspace manager."""
    name = _workspace_name(name)
    if not _workspace_exists(name):
        raise HTTPException(404, f"工作区不存在：{name}")
    base = _workspace_dir(name)
    parts = {
        "configs": _dir_usage(os.path.join(base, "configs")),
        "train_cache": _dir_usage(os.path.join(base, "train_cache")),
        "reg_cache": _dir_usage(os.path.join(base, "reg_cache")),
        "runs": _dir_usage(os.path.join(base, "runs")),
        "logs": _dir_usage(os.path.join(base, "logs")),
        "baseline": _dir_usage(os.path.join(base, "baseline")),
        "exports": _dir_usage(os.path.join(base, "exports")),
    }
    runs = []
    runs_dir = os.path.join(base, "runs")
    if os.path.isdir(runs_dir):
        for entry in sorted(os.listdir(runs_dir)):
            run_dir = os.path.join(runs_dir, entry)
            if os.path.isdir(run_dir):
                runs.append({"run": entry, **_dir_usage(run_dir)})
    total_mb = round(_dir_usage(base)["mb"], 1)
    return {"workspace": name, "path": _rel(base), "absolute": base,
            "parts": parts, "runs": runs, "total_mb": total_mb}

@router.delete("/api/workspaces/{name}/runs/{run}")
def api_delete_run(name: str, run: str):
    name = _workspace_name(name)
    run = _safe_name(run, "run 名称")
    target = os.path.join(_workspace_runs_dir(name), run)
    if not os.path.isdir(target):
        raise HTTPException(404, f"run 不存在：{run}")
    shutil.rmtree(target)
    return {"ok": True}


@router.delete("/api/workspaces/{name}/cache")
def api_delete_cache(name: str):
    """清空工作区缓存：训练集与正则集两个目录。"""
    name = _workspace_name(name)
    base = _workspace_dir(name)
    for sub in ("train_cache", "reg_cache"):
        target = os.path.join(base, sub)
        if os.path.isdir(target):
            shutil.rmtree(target)
        os.makedirs(target, exist_ok=True)
    return {"ok": True}


@router.delete("/api/workspaces/{name}/baseline")
def api_delete_baseline(name: str):
    name = _workspace_name(name)
    target = os.path.join(_workspace_dir(name), "baseline")
    if os.path.isdir(target):
        shutil.rmtree(target)
    return {"ok": True}


@router.delete("/api/workspaces/{name}")
def api_delete_workspace(name: str):
    name = _workspace_name(name)
    if name in RESERVED_WORKSPACES:
        raise HTTPException(400, "保留名称，不能删除")
    target = _workspace_dir(name)
    if not os.path.isdir(target):
        raise HTTPException(404, f"工作区不存在：{name}")
    shutil.rmtree(target)
    return {"ok": True}


@router.get("/api/workspaces/{name}/log")
def api_run_log(name: str, run: str = "", tail: int = 200):
    """Tail the log file of a run (CLI-started runs have no in-memory job)."""
    name = _workspace_name(name)
    directory = _workspace_logs_dir(name)
    if not run:
        raise HTTPException(400, "run required")
    run = _safe_name(run, "run 名称")
    path = os.path.join(directory, f"{run}.log")
    if not os.path.isfile(path):
        return {"lines": [], "path": _rel(path), "exists": False}
    with open(path, encoding="utf-8", errors="replace") as handle:
        lines = handle.readlines()
    return {"lines": [line.rstrip("\n") for line in lines[-max(1, min(tail, 2000)):]],
            "path": _rel(path), "exists": True}


def _has_live_cli_training(name: str) -> bool:
    workspace = os.path.abspath(_workspace_dir(name))
    for item in os.listdir("/proc"):
        if not item.isdigit():
            continue
        if _is_cli_train_process(int(item), workspace):
            return True
    return False


@router.get("/api/workspaces/{name}/active")
def api_active_run(name: str):
    """Which run is training right now, judged by the heartbeat file."""
    name = _workspace_name(name)
    runs_dir = _workspace_runs_dir(name)
    best = None
    if os.path.isdir(runs_dir):
        for run in os.listdir(runs_dir):
            beat = os.path.join(runs_dir, run, "training.json")
            if not os.path.isfile(beat):
                continue
            try:
                data = json.load(open(beat, encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            age = time.time() - float(data.get("time") or 0)
            pid = data.get("pid")
            if pid:
                try:
                    alive = int(pid) > 0 and os.path.exists(f"/proc/{int(pid)}")
                except (TypeError, ValueError):
                    alive = False
                if not alive:
                    continue
            if data.get("status") != "done" and age < 120 and (best is None or age < best["age"]):
                best = {"run": run, "step": data.get("step"), "total": data.get("total"),
                        "status": data.get("status"), "age": round(age, 1)}
    if best is None and _has_live_cli_training(name):
        # fall back to the log mtime only while a matching python CLI process lives
        logs = _workspace_logs_dir(name)
        if os.path.isdir(logs):
            for entry in os.scandir(logs):
                if not entry.name.endswith(".log"):
                    continue
                age = time.time() - entry.stat().st_mtime
                if age < 120 and (best is None or age < best["age"]):
                    best = {"run": os.path.splitext(entry.name)[0], "step": None,
                            "total": None, "status": "running", "age": round(age, 1)}
    return {"active": best}


@router.post("/api/workspaces/{name}/stop")
def api_stop_training(name: str):
    """Stop CLI-started training for one workspace.

    Prefer the pid stored in training.json; if there is no usable heartbeat,
    scan /proc for the python process (never the bash wrapper).
    """
    name = _workspace_name(name)
    if not _workspace_exists(name):
        raise HTTPException(404, f"工作区不存在：{name}")
    workspace = os.path.abspath(_workspace_dir(name))
    pids: list[int] = []
    seen: set[int] = set()

    runs_dir = _workspace_runs_dir(name)
    if os.path.isdir(runs_dir):
        for entry in os.scandir(runs_dir):
            if not entry.is_dir():
                continue
            beat = os.path.join(entry.path, "training.json")
            try:
                data = json.load(open(beat, encoding="utf-8"))
                pid = int(data.get("pid"))
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                continue
            if pid > 0 and pid not in seen and os.path.exists(f"/proc/{pid}") and _is_cli_train_process(pid, workspace):
                pids.append(pid)
                seen.add(pid)

    if not pids:
        for item in os.listdir("/proc"):
            if not item.isdigit():
                continue
            pid = int(item)
            if pid in seen or pid == os.getpid():
                continue
            if _is_cli_train_process(pid, workspace):
                pids.append(pid)
                seen.add(pid)

    import signal
    stopped: list[int] = []
    for pid in pids:
        try:
            os.kill(pid, signal.SIGINT)
            stopped.append(pid)
        except ProcessLookupError:
            continue
        except PermissionError:
            continue
    return {"ok": bool(stopped), "stopped": stopped}


@router.get("/api/workspaces/{name}/layout")
def api_layout_get(name: str):
    """Dashboard card layout, persisted per workspace."""
    name = _workspace_name(name)
    path = os.path.join(_workspace_dir(name), "layout.json")
    if not os.path.isfile(path):
        return {"workspace": name, "layout": None}
    try:
        with open(path, encoding="utf-8") as handle:
            return {"workspace": name, "layout": json.load(handle)}
    except (OSError, json.JSONDecodeError):
        return {"workspace": name, "layout": None}


@router.put("/api/workspaces/{name}/layout")
async def api_layout_put(name: str, request: Request):
    name = _workspace_name(name)
    if not _workspace_exists(name):
        raise HTTPException(404, f"工作区不存在：{name}")
    body = await request.json()
    layout = body.get("layout")
    if not isinstance(layout, dict):
        raise HTTPException(400, "layout must be an object")
    path = os.path.join(_workspace_dir(name), "layout.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(layout, handle, ensure_ascii=False, indent=1)
    return {"ok": True, "path": _rel(path)}

@router.get("/api/dashboard")
def api_dashboard(run: str = "", workspace: str = ""):
    """Aggregated snapshot for one workspace."""
    scope = _resolve_workspace(workspace)
    run_dirs = [(root, directory) for root, directory in _workspace_run_dirs(scope)]
    runs = []
    for root, directory in run_dirs:
        rows = _parse_metrics(directory)
        if not rows:
            continue
        rel = os.path.relpath(directory, root)
        label = f"{scope}/{rel.replace(os.sep, '/')}"
        parts = rel.split(os.sep)
        steps = [r for r in rows if "train" in r]
        evals = [r for r in rows if "eval" in r]
        metrics_path = os.path.join(directory, "metrics.jsonl")
        runs.append({
            "name": label,
            "label": label,
            "path": _rel(directory),
            "root": _rel(root),
            "workspace": scope,
            "profile": scope,
            "run_id": parts[-1] if parts else label,
            "mtime": os.path.getmtime(metrics_path),
            "rows": rows,
            "step": (steps[-1]["step"] if steps else 0),
            "total": (steps[-1].get("total") if steps else 0),
            "train": (steps[-1]["train"] if steps else None),
            "eval": (evals[-1]["eval"] if evals else None),
            "samples": _run_samples(directory),
        })

    active = None
    for job in JOBS.items.values():
        if job.kind == "train" and job.status == "running" and (job.workspace or "") == scope:
            active = {
                "id": job.id, "kind": job.kind,
                "step": job.last.get("step"), "total": job.last.get("total"),
                "profile": job.profile, "workspace": job.workspace, "run": job.run_name,
            }
    gpu = {"util": None, "mem_used": None, "mem_total": None, "temp": None, "name": None}
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3,
        ).stdout.strip().splitlines()[0]
        name, util, used, total, temp = [part.strip() for part in out.split(",")]
        gpu = {"name": name, "util": int(util), "mem_used": round(int(used) / 1024, 1),
               "mem_total": round(int(total) / 1024, 1), "temp": int(temp)}
    except Exception:
        pass

    runs.sort(key=lambda r: r["mtime"], reverse=True)
    chosen = next((r for r in runs if r["name"] == run), None) or (runs[0] if runs else None)
    return {
        "gpu": gpu,
        "active": active,
        "workspace": scope,
        "runs": [{k: r[k] for k in (
            "name", "label", "path", "root", "workspace", "profile", "run_id", "mtime",
            "step", "total", "train", "eval",
        )} for r in runs],
        "run": chosen["name"] if chosen else None,
        "rows": chosen["rows"] if chosen else [],
        "samples": chosen["samples"] if chosen else [],
    }
