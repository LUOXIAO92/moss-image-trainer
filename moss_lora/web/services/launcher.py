# moved from app.py: L557-559, L721-901
"""Child processes: command planning, log streaming and the job runner."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time

import yaml
from fastapi import HTTPException

from ..deps import ROOT, STEP_RE
from ..jobs import Job
from .cache_cmd import CACHE_PARTS


def _python() -> str:
    """The bundled interpreter is the only interpreter; no discovery/UI."""
    return sys.executable

def _command(kind: str, config_path: str, parts: list[str] | None = None) -> list[str]:
    python = _python()
    if kind == "cache":
        steps = []
        if "latent" in (parts or CACHE_PARTS):
            steps.append(f'"{python}" -m moss_lora.cli cache --config "{config_path}"')
        if "text" in (parts or CACHE_PARTS):
            steps.append(f'"{python}" -m moss_lora.cli encode-text --config "{config_path}"')
        if len(steps) == 1:
            return ["bash", "-c", steps[0]]
        # 先 latent 再文本，任一失败即停
        return ["bash", "-c", " && ".join(steps)]
    if kind == "train":
        return [python, "-m", "moss_lora.cli", "train", "--config", config_path]
    raise HTTPException(400, f"unknown job kind {kind}")


def _write_preview_sidecars(job: Job) -> None:
    """Write reproducibility metadata after a successful training job.

    The trainer owns sampling; this hook only records the seed/prompt that the
    exact run used.  A seed of -1 is resolved once per job before the snapshot
    is written, so every image has a concrete seed in its sidecar.
    """
    if not job.config_path or not os.path.isfile(job.config_path):
        return
    try:
        with open(job.config_path, encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        out_dir = os.path.join(str(data.get("output_dir") or ""), str(data.get("name") or ""))
        samples = os.path.join(out_dir, "samples")
        if not os.path.isdir(samples):
            return
        preview = data.get("preview") or {}
        seed = int(preview.get("seed", 42))
        prompts = [str(p) for p in (preview.get("prompts") or [])]
        entries = []
        for filename in sorted(os.listdir(samples)):
            match = re.fullmatch(r"step(\d+)_(\d+)\.png", filename)
            if not match:
                continue
            step, index = int(match.group(1)), int(match.group(2))
            entry = {
                "image": filename,
                "seed": seed,
                "step": step,
                "index": index,
                "prompt": prompts[index] if index < len(prompts) else "",
                "negative_prompt": str(preview.get("negative_prompt") or ""),
                "width": preview.get("width"),
                "height": preview.get("height"),
                "steps": preview.get("steps"),
                "guidance": preview.get("guidance"),
                "shift": preview.get("shift"),
            }
            sidecar = os.path.join(samples, f"{filename}.json")
            with open(sidecar, "w", encoding="utf-8") as handle:
                json.dump(entry, handle, ensure_ascii=False, indent=1)
            entries.append(entry)
        if entries:
            with open(os.path.join(samples, "seeds.json"), "w", encoding="utf-8") as handle:
                json.dump(entries, handle, ensure_ascii=False, indent=1)
    except Exception:
        # Sidecars are auxiliary; never fail a completed training job.
        return


def _stream_command(job: Job, command: list[str], log_handle, parse_steps: bool) -> int:
    """Run one child process, stream it into the job log, and return its code."""
    job.push({"type": "start", "command": " ".join(command)})
    if log_handle is not None:
        log_handle.write("$ " + " ".join(command) + "\n")
        log_handle.flush()
    job.process = subprocess.Popen(
        command, cwd=ROOT, env=dict(job.env or {}), stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, bufsize=1, start_new_session=True,
    )
    try:
        for line in job.process.stdout:
            line = line.rstrip()
            if not line:
                continue
            if log_handle is not None:
                log_handle.write(line + "\n")
                log_handle.flush()
            if parse_steps:
                match = STEP_RE.search(line)
                if match:
                    step, total, loss, avg, lr, sec, peak = match.groups()
                    event = {
                        "type": "progress", "step": int(step), "total": int(total),
                        "loss": float(loss), "avg": float(avg), "lr": float(lr),
                        "sec": float(sec), "vram": float(peak),
                    }
                    job.last = event
                    job.push(event)
                    continue
                if "preview ->" in line:
                    job.push({"type": "samples", "line": line})
            job.push({"type": "log", "line": line})
        return job.process.wait()
    finally:
        process = job.process
        job.process = None
        if process is not None and process.poll() is None:
            process.kill()


def _run(job: Job):
    job.env = dict(job.env or {})
    job.env.setdefault("PYTHONUNBUFFERED", "1")
    log_handle = None
    try:
        os.makedirs(os.path.dirname(job.log_path), exist_ok=True)
        log_handle = open(job.log_path, "a", encoding="utf-8")

        stopped = False
        for step in job.pre_commands:
            stage = str(step.get("stage") or "pre")
            command = step.get("command") or []
            job.stage = stage
            job.push({"type": "stage", "stage": stage, "status": "running"})
            if log_handle is not None:
                log_handle.write(f"[{stage}] starting\n")
                log_handle.flush()
            try:
                code = _stream_command(job, command, log_handle, parse_steps=False)
            except Exception as exc:  # a pre-step must never block training
                code = 1
                if log_handle is not None:
                    log_handle.write(f"[{stage}] failed to start: {exc}\n")
                    log_handle.flush()
            if job.status == "stopped":
                stopped = True
                break
            if code == 0:
                job.baseline_status = "done"
                job.push({"type": "stage", "stage": stage, "status": "done"})
                if log_handle is not None:
                    log_handle.write(f"[{stage}] done\n")
                    log_handle.flush()
            else:
                message = f"{stage} 失败（exit code {code}），继续训练"
                job.baseline_status = "error"
                job.baseline_error = f"exit code {code}"
                if log_handle is not None:
                    log_handle.write(f"[{stage}] {message}\n")
                    log_handle.flush()
                job.push({
                    "type": "warning",
                    "title": "基线采样失败",
                    "message": f"无 LoRA 基线没有生成（exit code {code}），训练会继续。",
                })
                job.push({"type": "stage", "stage": stage, "status": "error"})
        if stopped or job.status == "stopped":
            job.push({"type": "error", "message": "stopped"})
            return

        job.stage = "train"
        job.push({"type": "stage", "stage": "train", "status": "running"})
        code = _stream_command(job, job.command, log_handle, parse_steps=True)
        if job.status == "stopped":
            job.push({"type": "error", "message": "stopped"})
            return
        job.status = "done" if code == 0 else "error"
        if code == 0 and job.kind == "train":
            _write_preview_sidecars(job)
        if log_handle is not None:
            log_handle.write(f"[job] {'done' if code == 0 else 'error'} · exit code {code}\n")
            log_handle.flush()
        job.push({"type": "done" if code == 0 else "error", "message": f"exit code {code}"})
    except Exception as exc:  # pragma: no cover
        job.status = "error"
        job.push({"type": "error", "message": str(exc)})
        if log_handle is not None:
            log_handle.write(f"[job] error: {exc}\n")
    finally:
        if log_handle is not None:
            log_handle.close()
        job.finished = time.time()
