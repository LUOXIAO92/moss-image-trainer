# moved from app.py: L1480-1723, L1749-1751
"""Job creation, listing, SSE events, log tail and stop."""

from __future__ import annotations

import asyncio
import json
import os
import random
import threading
import time
import uuid

import yaml
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response, StreamingResponse

from ... import settings as runtime_settings
from ...config import apply_preset
from .. import deps
from ..deps import _rel, _settings_snapshot, _write_text
from ..jobs import JOBS, Job, _terminate
from ..services.cache_cmd import _cache_parts, _cache_summary
from ..services.caption import (_caption_command, _caption_env, _caption_selection,
                                resolve_api_key)
from ..services.launcher import _command, _python, _run
from ..services.workspace import (_resolve_workspace, _section, _workspace_cache_dir,
                                  _workspace_config, _workspace_dir, _workspace_logs_dir,
                                  _workspace_runs_dir, _workspace_target_run)

router = APIRouter()


@router.post("/api/jobs")
async def api_start(request: Request):
    body = await request.json()
    kind = body.get("kind")
    if kind not in ("train", "cache", "caption"):
        raise HTTPException(400, f"unknown job kind {kind}")
    datasets, styles, overwrite = _caption_selection(body) if kind == "caption" else ([], [], False)
    parts = _cache_parts(body) if kind == "cache" else []
    text = body.get("yaml") or ""
    if not text:
        raise HTTPException(400, "yaml required")
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        mark = getattr(exc, "problem_mark", None)
        where = f" 第 {mark.line + 1} 行第 {mark.column + 1} 列" if mark else ""
        raise HTTPException(400, f"YAML 语法错误{where}: {getattr(exc, 'problem', exc)}")
    if not isinstance(data, dict):
        raise HTTPException(400, "配置必须是一个 YAML 映射")

    settings, resolved = _settings_snapshot()
    workspace = _resolve_workspace(body.get("workspace"))
    config = _workspace_config(workspace, data)
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
    if kind == "caption":
        snapshot = os.path.join(log_dir, f"{job_id}.yaml")
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


@router.get("/api/jobs")
def api_jobs():
    return [
        {
            "id": j.id, "kind": j.kind, "status": j.status, "created": j.created,
            "last": j.last, "finished": j.finished,
            "config_path": _rel(j.config_path) if j.config_path else "",
            "profile": j.profile, "workspace": j.workspace,
            "run_name": j.run_name, "output_dir": j.output_dir, "log_path": _rel(j.log_path) if j.log_path else "",
            "stage": j.stage, "baseline_status": j.baseline_status, "baseline_error": j.baseline_error,
        }
        for j in sorted(JOBS.items.values(), key=lambda j: j.created, reverse=True)
    ]


@router.get("/api/jobs/{job_id}/events")
async def api_events(job_id: str, from_index: int = 0):
    job = JOBS.get(job_id)

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
                yield "data: {\"type\": \"closed\"}\n\n"
                return
            import asyncio

            await asyncio.sleep(0.3)
            idle += 0.3
            if idle > 15:
                yield ": keep-alive\n\n"
                idle = 0.0

    return StreamingResponse(stream(), media_type="text/event-stream")


@router.get("/api/jobs/{job_id}/log")
def api_job_log(job_id: str):
    job = JOBS.get(job_id)
    if not job.log_path or not os.path.isfile(job.log_path):
        raise HTTPException(404, "日志文件不存在")
    with open(job.log_path, encoding="utf-8", errors="replace") as handle:
        return Response(handle.read(), media_type="text/plain; charset=utf-8")

@router.post("/api/jobs/{job_id}/stop")
def api_stop(job_id: str):
    return {"ok": _terminate(JOBS.get(job_id))}
