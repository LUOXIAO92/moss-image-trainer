# moved from app.py: L1338-1377, L1826-1999, L2067-2146, L2382-2449
"""Workspace CRUD, disk usage, run/process control, layout and dashboard."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from ... import runtime
from ..deps import RESERVED_WORKSPACES, _rel, _safe_name
from ..jobs import JOBS
from ..services.system import (_is_cli_train_process, _parse_metrics, _run_samples)
from ..services.workspace import (_dir_usage, _ensure_workspace_dirs, _has_live_cli_training,
                                  _resolve_workspace, _run_config_snapshot, _workspace_dir,
                                  _workspace_exists, _workspace_info, _workspace_logs_dir,
                                  _workspace_name, _workspace_root, _workspace_run_dirs,
                                  _workspace_run_list, _workspace_runs_dir, _workspaces_payload)

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
    _ensure_workspace_dirs(name)
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
    """Which run is training right now, derived from the workspace files."""
    name = _workspace_name(name)
    runs = runtime.scan_runs(os.path.abspath(_workspace_dir(name)))
    live = [run for run in runs if run["state"] in ("running", "starting", "stopping")]
    if live:
        run = live[0]
        return {"active": {"run": run["id"], "step": run.get("step"), "total": run.get("total"),
                           "status": "running", "age": run.get("age"), "state": run["state"]}}
    if _has_live_cli_training(name):
        # fall back to the log mtime only while a matching python CLI process lives
        logs = _workspace_logs_dir(name)
        best = None
        if os.path.isdir(logs):
            for entry in os.scandir(logs):
                if not entry.name.endswith(".log"):
                    continue
                age = time.time() - entry.stat().st_mtime
                if age < 120 and (best is None or age < best["age"]):
                    best = {"run": os.path.splitext(entry.name)[0], "step": None,
                            "total": None, "status": "running", "age": round(age, 1)}
        return {"active": best}
    return {"active": None}


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
        config_yaml, config_path, config_source = _run_config_snapshot(scope, parts[-1])
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
            # The run's own record of what it was trained with; empty when the
            # run predates the snapshot or never started.
            "config_path": config_path,
            "config_source": config_source,
        })

    active = None
    for job in JOBS.items.values():
        if job.kind == "train" and job.status == "running" and (job.workspace or "") == scope:
            active = {
                "id": job.id, "kind": job.kind,
                "step": job.last.get("step"), "total": job.last.get("total"),
                "profile": job.profile, "workspace": job.workspace, "run": job.run_name,
            }
    if active is None:
        # A training started in a terminal is not in this process's memory, and
        # the dashboard still has to say it is running. The heartbeat knows.
        for row in runtime.scan_jobs(_workspace_root()):
            if (row.get("workspace") or "") != scope or row.get("kind") != "train":
                continue
            if row.get("state") not in ("running", "starting", "stopping"):
                continue
            active = {
                "id": row.get("id"), "kind": "train",
                "step": row.get("step"), "total": row.get("total"),
                "profile": row.get("profile") or scope, "workspace": scope,
                "run": row.get("run_id") or "", "source": row.get("source") or "cli",
            }
            break
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
            "step", "total", "train", "eval", "config_path", "config_source",
        )} for r in runs],
        "run": chosen["name"] if chosen else None,
        "run_config_path": chosen["config_path"] if chosen else "",
        "run_config_source": chosen["config_source"] if chosen else "",
        "rows": chosen["rows"] if chosen else [],
        "samples": chosen["samples"] if chosen else [],
    }
