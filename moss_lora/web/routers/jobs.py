# moved from app.py: L1480-1723, L1749-1751
"""Job creation, listing, SSE events, log tail and stop."""

from __future__ import annotations

import asyncio
import json
import os
import random
import signal
import threading
import time
import uuid

import yaml
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response, StreamingResponse

from ... import runtime
from ... import settings as runtime_settings
from ...config import apply_preset
from ..deps import _rel, _safe_name, _settings_snapshot, _write_text
from ..jobs import JOBS, Job, _terminate
from ..services.cache_cmd import _cache_parts, _cache_summary
from ..services.caption import (_caption_command, _caption_env, _caption_selection,
                                resolve_api_key)
from ..services.launcher import _command, _python, _run
from ..services.workspace import (_config_from_data, _resolve_workspace, _run_config_path,
                                  _run_is_live, _section, _workspace_cache_dir, _workspace_dir,
                                  _workspace_logs_dir, _workspace_root, _workspace_runs_dir,
                                  _workspace_target_run)

router = APIRouter()


@router.post("/api/jobs")
async def api_start(request: Request):
    body = await request.json()
    kind = body.get("kind")
    if kind not in ("train", "cache", "caption"):
        raise HTTPException(400, f"unknown job kind {kind}")
    datasets, styles, overwrite = _caption_selection(body) if kind == "caption" else ([], [], False)
    parts = _cache_parts(body) if kind == "cache" else []
    workspace = _resolve_workspace(body.get("workspace"))
    text = body.get("yaml") or ""
    run_arg = str(body.get("run") or "").strip()
    if run_arg:
        # A run names its own config; the file is the only thing read. Editing
        # it afterwards is allowed, but not while the run is training.
        run_arg = _safe_name(run_arg, "run 名称")
        candidate = _run_config_path(workspace, run_arg)
        if not os.path.isfile(candidate):
            raise HTTPException(404, f"运行没有配置存档：{run_arg}")
        if kind != "caption" and _run_is_live(workspace, run_arg):
            raise HTTPException(409, "本次运行进行中，配置已冻结")
        with open(candidate, encoding="utf-8") as handle:
            text = handle.read()
    if not text:
        raise HTTPException(400, 'yaml required (or pass {"run": "<name>"})')
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        mark = getattr(exc, "problem_mark", None)
        where = f" 第 {mark.line + 1} 行第 {mark.column + 1} 列" if mark else ""
        raise HTTPException(400, f"YAML 语法错误{where}: {getattr(exc, 'problem', exc)}")
    if not isinstance(data, dict):
        raise HTTPException(400, "配置必须是一个 YAML 映射")

    settings, resolved = _settings_snapshot()
    config = _config_from_data(workspace, data)
    apply_preset(config)
    problems = config.validate()
    if kind == "caption":
        # 只勾了正则集时，训练集路径空着也没关系
        if "train" in datasets:
            if not config.dataset.path or not os.path.isdir(os.path.expanduser(config.dataset.path)):
                raise HTTPException(400, f"训练集目录不存在：{config.dataset.path or '(空)'}")
        problems = []
    if kind == "cache" and "text" in parts and not str(config.model.text_encoder or "").strip():
        problems.append("文本编码需要 model.text_encoder（本地 Qwen3.5-9B 目录）")
    if problems:
        raise HTTPException(400, "配置有问题: " + "；".join(problems))

    profile = workspace
    python = _python()

    # The lock is the machine-wide truth: it also covers a run that a terminal
    # or an agent started, which the in-memory registry cannot see.
    holder = runtime.check_lock(_workspace_dir(workspace))
    if holder:
        raise HTTPException(
            409,
            f"已有任务占用 GPU：{holder.get('kind') or 'job'} pid {holder.get('pid')}"
            f"（{holder.get('run_id') or '-'}）",
            headers={"Retry-After": "10"},
        )

    running = JOBS.active()
    if running is not None:
        labels = {
            "train": "训练",
            "caption": "提示词生成",
            "cache": "缓存",
        }
        raise HTTPException(
            409,
            f"已有任务运行中：{labels.get(running.kind, running.kind)}（{running.id}）。"
            "请等待它结束，或先在训练页停止它。",
        )

    run_dir = ""
    run_id = ""
    run_label = config.name
    pre_commands: list[dict] = []
    stamp = time.strftime("%Y%m%d-%H%M%S")
    log_name = f"cache-{stamp}" if kind == "cache" else ""
    log_dir = _workspace_logs_dir(workspace)
    os.makedirs(log_dir, exist_ok=True)

    if kind == "train":
        workspace, run_id, run_dir = _workspace_target_run(workspace)
        config.output_dir = _workspace_runs_dir(workspace)
        config.name = run_id
        data["name"] = run_id
        data["output_dir"] = config.output_dir
        cache_section = _section(data, "cache")
        cache_section["dir"] = config.cache.dir
        if config.cache.reg_dir:
            cache_section["reg_dir"] = config.cache.reg_dir
        preview = _section(data, "preview")
        # Blank form fields are omitted from the workspace YAML; materialise the
        # backend defaults in the job snapshot so sampling, sidecars and the
        # baseline command all use the same concrete values.
        for key in ("steps", "guidance", "shift", "width", "height", "seed", "uncond_skip_layers"):
            preview.setdefault(key, getattr(config.preview, key))
        try:
            preview_seed = int(preview.get("seed", 42))
        except (TypeError, ValueError):
            preview_seed = 42
        if preview_seed < 0:
            # -1 means "random for this job"; resolve once so sidecars are
            # reproducible and every image has a concrete seed.
            preview["seed"] = random.SystemRandom().randint(0, 2**31 - 1)
        text = yaml.safe_dump(data, allow_unicode=True, sort_keys=False)
        cache_dir = os.path.abspath(os.path.expanduser(config.cache.dir))
        # 训练要两样东西：逐条 latent + 逐条文本嵌入。preview.safetensors 只给预览图用。
        summary = _cache_summary(config)
        if not summary["index_ready"]:
            raise HTTPException(
                400,
                f"缓存未就绪（{cache_dir}）：缺 index.json。"
                "先去「缓存」页点击「一键生成缓存（latent + 文本）」。",
            )
        if summary.get("legacy_index"):
            raise HTTPException(
                400,
                f"缓存格式不对（{cache_dir}）：不是当前版本的索引。"
                "请到「缓存」页勾上 latent 缓存和文本编码，重新生成后再训练。",
            )
        if summary.get("stale_sizing"):
            raise HTTPException(
                400,
                f"缓存的尺寸规格和当前配置不一致（{cache_dir}）："
                f"缓存按 {summary.get('sizing')} 生成，当前是 max_area={config.dataset.max_area}。"
                "到「缓存」页重新生成 latent 缓存后再训练。",
            )
        if summary.get("missing_text"):
            raise HTTPException(
                400,
                f"缓存缺文本嵌入（{summary['missing_text']}/{summary['items']} 条）：{cache_dir}。"
                "只做了 latent；请在「缓存」页勾上「文本编码」补一遍。",
            )
        log_name = run_id
        run_label = run_id
    elif kind == "cache":
        cache_section = _section(data, "cache")
        cache_section["dir"] = config.cache.dir
        if config.cache.reg_dir:
            cache_section["reg_dir"] = config.cache.reg_dir
        data["output_dir"] = config.output_dir
        text = yaml.safe_dump(data, allow_unicode=True, sort_keys=False)
        # Avoid clobbering a second cache job started in the same second.
        suffix = 2
        base_log = log_name
        while os.path.exists(os.path.join(log_dir, f"{log_name}.log")):
            log_name = f"{base_log}-{suffix}"
            suffix += 1
    elif kind == "caption":
        log_name = f"caption-{stamp}"
        run_label = f"caption-{stamp}"
    else:
        raise HTTPException(400, f"unknown job kind {kind}")

    job_id = uuid.uuid4().hex[:10]
    snapshot = os.path.join(log_dir, f"{job_id}.yaml")
    # Every kind passes this snapshot to its child process, so write it for all
    # of them. Only writing it for caption left train and cache pointing at a
    # file that never existed.
    _write_text(snapshot, text)
    if kind == "train" and body.get("baseline"):
        pre_commands.append({
            "stage": "baseline",
            "command": [
                python, "-m", "moss_lora.cli", "baseline",
                "--config", snapshot,
                "--out", os.path.join(_workspace_dir(workspace), "baseline"),
            ],
        })
    job_env = runtime_settings.child_env(settings, kind=kind)
    if kind == "caption":
        caption_settings = settings.get("caption") or {}
        _caption_env(job_env, caption_settings)
        if not job_env.get("VLM_BASE_URL"):
            raise HTTPException(400, "提示词生成还没配置视觉服务地址，去「提示词生成」页填写并保存后再开始。")
        if not job_env.get("VLM_API_KEY"):
            _, source = resolve_api_key(caption_settings)
            stored = str(caption_settings.get("api_key") or "").strip()
            if source == "missing":
                raise HTTPException(
                    400,
                    f"设置里存的是环境变量名 {stored}，但当前 WebUI 进程里读不到它的值："
                    "先 export 再重启 WebUI，或者直接把密钥填进「提示词生成」页（不保存，重开就没）。",
                )
            raise HTTPException(400, "提示词生成还没配置密钥：在「提示词生成」页填环境变量名或密钥，保存后再开始。")
    log_path = os.path.join(log_dir, f"{log_name}.log")
    try:
        os.makedirs(_workspace_cache_dir(workspace), exist_ok=True)
    except OSError:
        pass
    job = Job(
        id=job_id,
        kind=kind,
        config_path=snapshot,
        command=(_caption_command(config, settings.get("caption") or {}, log_dir, stamp,
                                  datasets, styles, overwrite)
                 if kind == "caption" else _command(kind, snapshot, parts)),
        profile=workspace,
        workspace=workspace,
        run_name=run_label,
        output_dir=_rel(run_dir) if run_dir else "",
        log_path=log_path,
        state_path=os.path.join(log_dir, f"{job_id}.json"),
        env=job_env,
        pre_commands=pre_commands,
    )
    JOBS.add(job)
    threading.Thread(target=_run, args=(job,), daemon=True).start()
    return {
        "id": job_id,
        "command": job.command,
        "config_path": _rel(snapshot),
        "log_path": _rel(log_path),
        "name": run_label,
        "run_id": run_id,
        "profile": workspace,
        "workspace": workspace,
        "output_dir": _rel(run_dir) if run_dir else None,
        "baseline": bool(pre_commands),
        "python": python,
    }


def _disk_jobs() -> list[dict]:
    """Every job any process started, read from the workspace, not from memory.

    This is what makes the WebUI show a run that a terminal or an agent
    started: same files, same list, and the same function the CLI reads.
    """
    return runtime.scan_jobs(_workspace_root())


_LEGACY_STATUS = {
    "running": "running",
    "starting": "running",
    "stopping": "stopping",
    "done": "done",
    "stopped": "stopped",
    "failed": "error",
    "interrupted": "error",
    "unknown": "error",
}


def _disk_job(job_id: str) -> dict | None:
    for entry in _disk_jobs():
        if entry.get("id") == job_id:
            return entry
    return None


@router.get("/api/jobs")
def api_jobs():
    """Merge the on-disk records with live in-memory handles.

    Disk wins on identity and outcome; memory only fills in detail that has not
    been flushed yet (the stage a pre-command is in).
    """
    merged: dict[str, dict] = {}
    for record in _disk_jobs():
        merged[record["id"]] = record
    for job in JOBS.all():
        entry = merged.get(job.id) or {}
        entry.update({
            "id": job.id,
            "kind": job.kind,
            "source": entry.get("source") or "server",
            "state": entry.get("state") or job.status,
            "workspace": job.workspace,
            "profile": job.profile,
            "run_id": job.run_name,
            "run_dir": entry.get("run_dir") or job.output_dir,
            "log": entry.get("log") or job.log_path,
            "config_path": job.config_path,
            "stage": job.stage or entry.get("stage") or "",
            "phase": job.stage or entry.get("phase") or "",
            "baseline_status": job.baseline_status,
            "baseline_error": job.baseline_error,
            "pid": job.last_pid or entry.get("pid"),
            "started": job.created,
            "finished": job.finished,
            "exit_code": job.exit_code,
            "last": job.last or entry.get("last"),
            "stale": entry.get("stale", False),
            "age": entry.get("age"),
            "_live": True,
        })
        merged[job.id] = entry

    items = []
    for entry in merged.values():
        state = str(entry.get("state") or "")
        created = float(entry.get("started") or 0.0)
        items.append({
            # legacy shape, still consumed by the WebUI
            "id": entry.get("id"),
            "kind": entry.get("kind") or "job",
            "status": _LEGACY_STATUS.get(state, state or "error"),
            "created": created,
            "last": entry.get("last") or {},
            "finished": float(entry.get("finished") or 0.0),
            "config_path": _rel(entry["config_path"]) if entry.get("config_path") else "",
            "profile": entry.get("profile") or entry.get("workspace") or "",
            "workspace": entry.get("workspace") or "",
            "run_name": entry.get("run_id") or "",
            "output_dir": _rel(entry["run_dir"]) if entry.get("run_dir") else "",
            "log_path": _rel(entry["log"]) if entry.get("log") else "",
            "stage": entry.get("stage") or entry.get("phase") or "",
            "baseline_status": entry.get("baseline_status") or "",
            "baseline_error": entry.get("baseline_error") or "",
            # the honest vocabulary
            "state": state,
            "source": entry.get("source") or "server",
            "stale": bool(entry.get("stale")),
            "age": entry.get("age"),
            "pid": entry.get("pid"),
            "exit_code": entry.get("exit_code"),
        })
    items.sort(key=lambda item: item.get("created") or 0.0, reverse=True)
    return items


@router.get("/api/jobs/{job_id}")
def api_job_detail(job_id: str):
    """One job, projected from the same files the list uses."""
    for entry in api_jobs():
        if entry["id"] == job_id:
            return entry
    raise HTTPException(404, f"job not found: {job_id}")


@router.get("/api/jobs/{job_id}/events")
async def api_events(job_id: str, from_index: int = 0):
    """Live events for a job, whichever process owns it.

    Jobs this server spawned stream from memory. Everything else -- a run
    started from a terminal, or one that outlived a server restart -- is
    tailed from the workspace files, so the panel looks the same either way.
    """
    job = JOBS.items.get(job_id)
    if job is None:
        record = _disk_job(job_id)
        if record is None:
            raise HTTPException(404, f"job not found: {job_id}")
        return StreamingResponse(_tail_disk_job(record), media_type="text/event-stream")

    async def stream():
        index = from_index
        idle = 0.0
        while True:
            while index < len(job.events):
                event = job.events[index]
                index += 1
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                idle = 0.0
            if job.status != "running" and index >= len(job.events):
                yield "data: " + json.dumps({
                    "type": "closed",
                    "state": "done" if job.status == "done" else "stopped" if job.status == "stopped" else "failed",
                    "exit_code": job.exit_code,
                }) + "\n\n"
                return
            import asyncio

            await asyncio.sleep(0.3)
            idle += 0.3
            if idle > 15:
                yield ": keep-alive\n\n"
                idle = 0.0

    return StreamingResponse(stream(), media_type="text/event-stream")


async def _tail_disk_job(record: dict):
    """Stream a job we did not spawn by tailing its log and heartbeat."""
    import asyncio

    log_path = record.get("log") or ""
    run_dir = record.get("run_dir") or ""
    state_path = record.get("state_path") or ""
    offset = 0
    seen_step = None
    idle = 0.0
    while True:
        if log_path and os.path.isfile(log_path):
            try:
                with open(log_path, encoding="utf-8", errors="replace") as handle:
                    handle.seek(offset)
                    chunk = handle.read()
                    offset = handle.tell()
            except OSError:
                chunk = ""
            for line in chunk.splitlines():
                if line.strip():
                    yield "data: " + json.dumps({"type": "log", "line": line}, ensure_ascii=False) + "\n\n"
                    idle = 0.0
        beat = runtime.read_json(state_path) if state_path else None
        if beat:
            metrics = runtime.last_metrics(run_dir) if run_dir else None
            step = (metrics or {}).get("step")
            if step is None:
                step = beat.get("step")
            if step is not None and step != seen_step:
                seen_step = step
                event = runtime.progress_event({**(metrics or {}), "step": step,
                                                "total": beat.get("total") or (metrics or {}).get("total")})
                if event:
                    yield "data: " + json.dumps(event, ensure_ascii=False) + "\n\n"
                    idle = 0.0
            state = runtime.classify(beat)["state"]
            exit_code = beat.get("exit_code")
        else:
            state, exit_code = record.get("state") or "unknown", record.get("exit_code")
        if state in runtime.TERMINAL or state == "unknown":
            yield "data: " + json.dumps({"type": "closed", "state": state, "exit_code": exit_code}) + "\n\n"
            return
        await asyncio.sleep(0.5)
        idle += 0.5
        if idle >= 15:
            yield ": keep-alive\n\n"
            idle = 0.0


@router.get("/api/jobs/{job_id}/log")
def api_job_log(job_id: str):
    job = JOBS.items.get(job_id)
    path = job.log_path if job is not None else ""
    if not path:
        record = _disk_job(job_id) or {}
        path = record.get("log") or ""
    if not path or not os.path.isfile(path):
        raise HTTPException(404, "日志文件不存在")
    with open(path, encoding="utf-8", errors="replace") as handle:
        return Response(handle.read(), media_type="text/plain; charset=utf-8")


@router.post("/api/jobs/{job_id}/stop")
def api_stop(job_id: str):
    job = JOBS.items.get(job_id)
    if job is not None:
        return {"ok": _terminate(job), "state": "stopping"}
    record = _disk_job(job_id)
    if record is None:
        raise HTTPException(404, f"job not found: {job_id}")
    pid = record.get("pid")
    if not runtime.pid_alive(pid):
        raise HTTPException(409, "任务已经不在运行")
    # SIGINT lets the trainer checkpoint before it exits; SIGKILL would not.
    try:
        os.kill(int(pid), signal.SIGINT)
    except (ProcessLookupError, PermissionError) as exc:
        raise HTTPException(409, f"无法发送停止信号：{exc}") from exc
    return {"ok": True, "state": "stopping"}
