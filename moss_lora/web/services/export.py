# moved from app.py: L1754-1809, L2149-2210, L2296-2334
"""LoRA export jobs (single checkpoint and batch) and the exported inventory."""

from __future__ import annotations

import os
import threading
import time
import uuid

from fastapi import HTTPException, Request

from ... import settings as runtime_settings
from ..deps import _rel, _safe_name, _settings_snapshot
from ..jobs import JOBS, Job
from .launcher import _python, _run
from .workspace import (_resolve_workspace, _run_config_snapshot, _workspace_dir,
                        _workspace_exists, _workspace_logs_dir, _workspace_name,
                        _workspace_runs_dir)


async def export_batch(name: str, request: Request):
    """Export a range of checkpoints at once, either into a directory or a zip."""
    name = _workspace_name(name)
    if not _workspace_exists(name):
        raise HTTPException(404, f"工作区不存在：{name}")
    body = await request.json()
    run = _safe_name(body.get("run"), "run 名称")
    run_dir = os.path.join(_workspace_runs_dir(name), run)
    if not os.path.isdir(run_dir):
        raise HTTPException(404, f"run 不存在：{run}")
    formats = [f for f in (body.get("formats") or ["kohya"]) if f in ("kohya", "diffusers")]
    if not formats:
        raise HTTPException(400, "至少选择一种格式")
    spec = str(body.get("steps") or "").strip()
    target = str(body.get("target_dir") or "").strip()
    want_zip = bool(body.get("zip"))
    # no target and no zip -> write next to the run (per-step download button)

    config_text, config_path, _source = _run_config_snapshot(name, run)
    if not config_text:
        raise HTTPException(400, "找不到该 run 的配置快照")
    stamp = time.strftime("%Y%m%d-%H%M%S")
    exports_root = os.path.join(_workspace_dir(name), "exports")
    zip_path = os.path.join(exports_root, f"{run}-{stamp}.zip") if want_zip else ""
    out_dir = target or (os.path.join(exports_root, f"{run}-{stamp}") if want_zip else run_dir)
    command = [
        _python(), "-m", "moss_lora.cli", "export-batch",
        "--config", config_path,
        "--run-dir", run_dir,
        "--steps", spec,
        "--formats", ",".join(formats),
        "--out", out_dir,
    ]
    if zip_path:
        command += ["--zip", zip_path]
    if body.get("state"):
        command += ["--state"]
    log_dir = _workspace_logs_dir(name)
    os.makedirs(log_dir, exist_ok=True)
    settings, _ = _settings_snapshot()
    job = Job(
        id=uuid.uuid4().hex[:10],
        kind="export",
        config_path=config_path,
        command=command,
        profile=name,
        workspace=name,
        run_name=f"{run} batch {stamp}",
        output_dir=_rel(out_dir),
        log_path=os.path.join(log_dir, f"export-batch-{run}-{stamp}.log"),
        env=runtime_settings.child_env(settings),
    )
    JOBS.add(job)
    threading.Thread(target=_run, args=(job,), daemon=True).start()
    return {"id": job.id, "run": run, "out_dir": _rel(out_dir), "zip": _rel(zip_path) if zip_path else ""}

async def export_one(name: str, request: Request):
    """Export one checkpoint of a finished run (kohya or diffusers layout)."""
    name = _workspace_name(name)
    if not _workspace_exists(name):
        raise HTTPException(404, f"工作区不存在：{name}")
    body = await request.json()
    run = _safe_name(body.get("run"), "run 名称")
    run_dir = os.path.join(_workspace_runs_dir(name), run)
    if not os.path.isdir(run_dir):
        raise HTTPException(404, f"run 不存在：{run}")
    step = str(body.get("step") or "final")
    if step in ("final", "0", ""):
        checkpoint = os.path.join(run_dir, "checkpoints", "final.safetensors")
        suffix = "_final"
        label = "final"
    else:
        try:
            number = int(step)
        except ValueError:
            raise HTTPException(400, f"步数不合法：{step}")
        checkpoint = os.path.join(run_dir, "checkpoints", f"step{number:06d}.safetensors")
        suffix = f"_step{number:06d}"
        label = f"step{number:06d}"
    if not os.path.isfile(checkpoint):
        raise HTTPException(404, f"checkpoint 不存在：{os.path.basename(checkpoint)}")
    formats = [f for f in (body.get("formats") or ["kohya"])
               if f in ("kohya", "diffusers")]
    if not formats:
        raise HTTPException(400, "至少选择一种格式")
    config_text, config_path, _source = _run_config_snapshot(name, run)
    if not config_text:
        raise HTTPException(400, "找不到该 run 的配置快照")
    log_dir = _workspace_logs_dir(name)
    os.makedirs(log_dir, exist_ok=True)
    settings, _ = _settings_snapshot()
    target = str(body.get("target_dir") or "").strip() or run_dir
    job = Job(
        id=uuid.uuid4().hex[:10],
        kind="export",
        config_path=config_path,
        command=[
            _python(), "-m", "moss_lora.cli", "export",
            "--config", config_path,
            "--checkpoint", checkpoint,
            "--out", target,
            "--formats", ",".join(formats),
            "--suffix", suffix,
        ] + (["--state"] if body.get("state") else []),
        profile=name,
        workspace=name,
        run_name=f"{run}{suffix}",
        output_dir=_rel(run_dir),
        log_path=os.path.join(log_dir, f"export-{run}-{label}.log"),
        env=runtime_settings.child_env(settings),
    )
    JOBS.add(job)
    threading.Thread(target=_run, args=(job,), daemon=True).start()
    files = [f"{run}{suffix}{'' if len(formats) == 1 else '_' + fmt}.safetensors" for fmt in formats]
    if body.get("state"):
        files.append(f"{run}{suffix}_state.safetensors")
    return {"id": job.id, "run": run, "step": label, "files": files}

def loras_list(workspace: str = "", run: str = "", include_state: bool = False):
    scope = _resolve_workspace(workspace)
    roots = [_workspace_runs_dir(scope)]
    wanted_run = str(run or "").strip().strip("/")
    out = []
    seen: set[str] = set()
    for base in roots:
        if not os.path.isdir(base):
            continue
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = [name for name in dirnames if not name.startswith(".")]
            rel_dir = os.path.relpath(dirpath, base)
            run_name = os.path.basename(base) if rel_dir == "." else rel_dir.replace(os.sep, "/")
            if wanted_run and run_name != wanted_run:
                continue
            for name in filenames:
                if not name.endswith(".safetensors"):
                    continue
                # checkpoints/ holds training state, not shippable LoRAs
                if os.path.basename(dirpath) == "checkpoints":
                    continue
                if name.endswith("_state.safetensors") and not include_state:
                    continue
                path = os.path.join(dirpath, name)
                real = os.path.realpath(path)
                if real in seen:
                    continue
                seen.add(real)
                out.append({
                    "name": name,
                    "path": _rel(path),
                    "run": f"{scope}/{run_name}",
                    "workspace": scope,
                    "state": name.endswith("_state.safetensors"),
                    "size_mb": round(os.path.getsize(path) / 2**20, 1),
                    "mtime": os.path.getmtime(path),
                })
    return sorted(out, key=lambda x: -x["mtime"])
