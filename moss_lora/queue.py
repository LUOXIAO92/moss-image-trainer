"""Continue every unfinished run in a workspace, one after another.

A run is "unfinished" when its ``metrics.jsonl`` stops below the ``total`` it was
configured for and a checkpoint exists to resume from.  The dashboard's one-click
button calls this; it is also usable from the CLI:

    python -m moss_lora.cli queue --workspace workspaces/<WORKSPACE>
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys

from .config import Config

def _read_metrics(run_dir: str) -> list[dict]:
    path = os.path.join(run_dir, "metrics.jsonl")
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _checkpoints(run_dir: str) -> list[tuple[int, str]]:
    directory = os.path.join(run_dir, "checkpoints")
    found = []
    if not os.path.isdir(directory):
        return found
    for name in os.listdir(directory):
        # Weight checkpoints only. ``stepNNNNNN.state.safetensors`` holds
        # optimizer state and is not a loadable LoRA on its own.
        match = re.fullmatch(r"step(\d+)\.safetensors", name)
        if not match:
            continue
        found.append((int(match.group(1)), os.path.join(directory, name)))
    return sorted(found)


def _snapshot(workspace: str, run: str) -> str | None:
    """A run resumes from the config it started with, and nothing else."""
    path = os.path.join(workspace, "runs", run, "config.yaml")
    return path if os.path.isfile(path) else None


def find_pending(workspace: str) -> list[dict]:
    """Runs that stopped short of their target and have a checkpoint to resume."""
    runs_dir = os.path.join(workspace, "runs")
    pending = []
    if not os.path.isdir(runs_dir):
        return pending
    for run in sorted(os.listdir(runs_dir)):
        run_dir = os.path.join(runs_dir, run)
        if not os.path.isdir(run_dir):
            continue
        rows = _read_metrics(run_dir)
        steps = [r for r in rows if "train" in r]
        if not steps:
            continue
        last = steps[-1]
        total = int(last.get("total") or 0)
        step = int(last.get("step") or 0)
        if total and step >= total:
            continue
        checkpoints = _checkpoints(run_dir)
        snapshot = _snapshot(workspace, run)
        if not snapshot:
            continue
        if not checkpoints:
            continue
        pending.append({
            "run": run,
            "step": step,
            "total": total or step,
            "checkpoint": checkpoints[-1][1],
            "checkpoint_step": checkpoints[-1][0],
            "config": snapshot,
        })
    return pending


def _workspace_config_path(workspace: str) -> str | None:
    """The newest run's config: the cache and queue operate per run."""
    runs_dir = os.path.join(workspace, "runs")
    if not os.path.isdir(runs_dir):
        return None
    found = []
    for name in os.listdir(runs_dir):
        path = os.path.join(runs_dir, name, "config.yaml")
        if os.path.isfile(path):
            found.append((os.path.getmtime(path), path))
    if not found:
        return None
    return max(found)[1]


def missing_cache(workspace: str) -> dict:
    """How many items still lack latents or text embeddings."""
    path = _workspace_config_path(workspace)
    if not path:
        raise SystemExit(f"queue: 没有可用的运行配置：{workspace}/runs/*/config.yaml")
    config = Config.load(path)
    from .cache import CacheBuilder

    builder = CacheBuilder(config, log=lambda *a: None)
    pending_latents = builder._pending("latent")
    pending_text = builder._pending("text_embeds")
    has_preview = os.path.exists(os.path.join(builder.dir, "preview.safetensors"))
    return {
        "latents": len(pending_latents),
        "text": len(pending_text),
        "preview": has_preview,
        "items": len(builder.items),
    }


def ensure_cache(workspace: str, log=print) -> None:
    config_path = _workspace_config_path(workspace)
    if not config_path:
        raise SystemExit(f"queue: 没有可用的运行配置：{workspace}/runs/*/config.yaml")
    log(f"queue: using {config_path}")
    config = Config.load(config_path)
    from .cache import CacheBuilder
    from .encode import run_encode_text

    builder = CacheBuilder(config, log=log)
    missing = len(builder._pending("latent"))
    if missing:
        log(f"cache: {missing} latent(s) missing -> building")
        builder.run_latents()
        builder.write_index()
    else:
        log("cache: latents complete")
    missing_text = len(builder._pending("text_embeds"))
    preview = os.path.join(builder.dir, "preview.safetensors")
    if missing_text or not os.path.exists(preview):
        log(f"cache: {missing_text} text embedding(s) missing -> encoding")
        run_encode_text(config_path, log=log)
    else:
        log("cache: text embeddings complete")


def run_pipeline(workspace: str, log=print) -> int:
    """Cache -> text -> resume/start every unfinished run."""
    ensure_cache(workspace, log=log)
    return run_queue(workspace, log=log)


def run_queue(workspace: str, log=print) -> int:
    pending = find_pending(workspace)
    if not pending:
        log("queue: nothing pending")
        return 0
    log(f"queue: {len(pending)} run(s) to finish")
    failures = 0
    for item in pending:
        log(f"queue: {item['run']} {item['step']}/{item['total']} "
            f"resume from {os.path.basename(item['checkpoint'])}")
        command = [
            sys.executable, "-m", "moss_lora.cli", "train",
            "--config", item["config"],
            "--steps", str(item["total"]),
        ]
        if item["checkpoint"]:
            command += ["--resume", item["checkpoint"]]
        code = subprocess.call(command)
        if code != 0:
            failures += 1
            log(f"queue: {item['run']} failed with exit code {code}")
    log(f"queue: finished ({failures} failure(s))")
    return 1 if failures else 0
