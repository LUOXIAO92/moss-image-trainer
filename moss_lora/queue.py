"""Continue every unfinished run in a workspace, one after another.

A run is "unfinished" when its ``metrics.jsonl`` stops below the ``total`` it was
configured for and a checkpoint exists to resume from.  The dashboard's one-click
button calls this; it is also usable from the CLI:

    python -m moss_lora.cli queue --workspace workspaces/<WORKSPACE>
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

from .config import Config

TRAIN_SNAPSHOTS = ("logs", "configs")


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
        if not name.startswith("step") or not name.endswith(".safetensors"):
            continue
        try:
            step = int(name[4:10])
        except ValueError:
            continue
        found.append((step, os.path.join(directory, name)))
    return sorted(found)


def _snapshot(workspace: str, run: str) -> str | None:
    for folder in TRAIN_SNAPSHOTS:
        path = os.path.join(workspace, folder, f"{run}.yaml")
        if os.path.isfile(path):
            return path
    config = os.path.join(workspace, "workspace.yaml")
    return config if os.path.isfile(config) else None


def _arm_configs(workspace: str) -> dict[str, str]:
    """Saved training configs under ``<ws>/configs``."""
    arms = {}
    for folder in ("configs",):
        directory = os.path.join(workspace, folder)
        if not os.path.isdir(directory):
            continue
        for name in os.listdir(directory):
            if name.endswith((".yaml", ".yml")):
                arms.setdefault(os.path.splitext(name)[0], os.path.join(directory, name))
    return arms


def find_pending(workspace: str) -> list[dict]:
    """Runs that stopped short of their target, plus configs that never started."""
    runs_dir = os.path.join(workspace, "runs")
    pending = []
    for run in sorted(_arm_configs(workspace)):
        if not os.path.isdir(os.path.join(runs_dir, run)):
            cfg_path = _arm_configs(workspace)[run]
            try:
                config = Config.load(cfg_path)
            except Exception:
                continue
            pending.append({
                "run": run, "step": 0, "total": int(config.train.steps),
                "checkpoint": None, "checkpoint_step": 0, "config": cfg_path,
            })
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


def missing_cache(workspace: str) -> dict:
    """How many items still lack latents or text embeddings."""
    config = Config.load(os.path.join(workspace, "workspace.yaml"))
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
    config_path = os.path.join(workspace, "workspace.yaml")
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
