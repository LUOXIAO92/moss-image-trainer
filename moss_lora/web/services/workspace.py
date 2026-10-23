# moved from app.py: L164-550, L1447-1460, L1812-1823, L1916-1923
"""Workspaces: directories, saved runs, cache inventory and config resolution."""

from __future__ import annotations

import copy
import json
import os
import re
import time
from dataclasses import asdict

import yaml
from fastapi import HTTPException

from ...config import Config
from ..deps import (RESERVED_WORKSPACES, ROOT, WORKSPACE_MARKER, _rel, _safe_name,
                    _settings_snapshot, _write_text)
from .system import _dir_size, _is_cli_train_process, _parse_metrics


# ---------------------------------------------------------------------------
# workspaces
# ---------------------------------------------------------------------------

def _workspace_root() -> str:
    _, resolved = _settings_snapshot()
    return resolved["paths"]["workspaces"]


def _workspace_dir(name: str) -> str:
    return os.path.join(_workspace_root(), name)



def _workspace_configs_dir(name: str) -> str:
    return os.path.join(_workspace_dir(name), "configs")


def _config_entry(path: str) -> dict:
    """Small summary the training page shows for each saved config."""
    entry = {"name": os.path.splitext(os.path.basename(path))[0], "path": _rel(path),
             "mtime": os.path.getmtime(path)}
    try:
        with open(path, encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        entry.update({
            "steps": (data.get("train") or {}).get("steps"),
            "rank": (data.get("lora") or {}).get("rank"),
            "quantize": (data.get("vram") or {}).get("quantize_dit"),
            "lr": (data.get("train") or {}).get("lr"),
            "dataset": (data.get("dataset") or {}).get("path") or "",
        })
    except Exception as exc:
        entry["error"] = str(exc)
    return entry


def _workspace_config_path(name: str) -> str:
    return os.path.join(_workspace_dir(name), WORKSPACE_MARKER)


def _workspace_name(value: str | None) -> str:
    name = _safe_name(value, "工作区名称")
    if name in RESERVED_WORKSPACES:
        raise HTTPException(400, f"{name} 是保留名称，不能作为工作区")
    return name


def _workspace_exists(name: str) -> bool:
    """A workspace is a directory; the config file is created on demand."""
    return os.path.isdir(_workspace_dir(name))


def _ensure_workspace_dirs(name: str) -> str:
    directory = _workspace_dir(name)
    for sub in ("", "configs", "train_cache", "reg_cache", "runs", "logs"):
        os.makedirs(os.path.join(directory, sub), exist_ok=True)
    return directory


def _workspace_names() -> list[str]:
    root = _workspace_root()
    if not os.path.isdir(root):
        return []
    names = []
    for name in sorted(os.listdir(root)):
        if name.startswith(".") or name in RESERVED_WORKSPACES:
            continue
        if os.path.isfile(os.path.join(root, name, WORKSPACE_MARKER)):
            names.append(name)
    return names


def _default_workspace_name() -> str:
    names = _workspace_names()
    if not names:
        raise HTTPException(400, "请先创建工作区")
    return names[0]


def _resolve_workspace(workspace: str | None) -> str:
    raw = str(workspace or "").strip()
    if not raw:
        return _default_workspace_name()
    name = _workspace_name(raw)
    if not _workspace_exists(name):
        raise HTTPException(404, f"工作区不存在：{raw}")
    return name


def _workspace_data(name: str) -> dict:
    path = _workspace_config_path(name)
    if not os.path.isfile(path):
        # deleting the folder (or only workspace.yaml) must not brick the app:
        # the defaults are code-side, so an empty dict is a valid workspace
        if not os.path.isdir(_workspace_dir(name)):
            raise HTTPException(404, f"工作区不存在：{name}")
        return {}
    with open(path, encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise HTTPException(400, f"工作区配置不是映射：{name}/{WORKSPACE_MARKER}")
    return data


def _workspace_config_text() -> str:
    data = asdict(Config())
    data["name"] = "moss-lora"
    data["dataset"]["path"] = ""
    data["preview"]["prompts"] = []
    data["model"]["dit"] = ""
    data["model"]["text_encoder"] = ""
    data["model"]["vae"] = ""
    data["cache"]["dir"] = "train_cache"
    data["output_dir"] = "runs"
    return yaml.safe_dump(data, sort_keys=False, allow_unicode=True)


def _workspace_config(name: str, data: dict | None = None) -> Config:
    """Resolve workspace-relative paths for a job/preflight.

    The YAML on disk stays portable (``cache`` / ``runs``).  Only the copy
    handed to the existing trainer/cache code is absolute.
    """
    _ensure_workspace_dirs(name)
    raw = copy.deepcopy(data if data is not None else _workspace_data(name))
    config = Config.from_dict(raw).resolve_paths()
    ws = _workspace_dir(name)
    # 空字段 = 工作区默认目录；Config.resolve_paths() 已经填好 train_cache
    config.cache.dir = config.cache.dir or "train_cache"
    config.output_dir = config.output_dir or "runs"
    if not os.path.isabs(config.cache.dir):
        config.cache.dir = os.path.abspath(os.path.join(ws, config.cache.dir))
    if config.cache.reg_dir and not os.path.isabs(config.cache.reg_dir):
        config.cache.reg_dir = os.path.abspath(os.path.join(ws, config.cache.reg_dir))
    if not os.path.isabs(config.output_dir):
        config.output_dir = os.path.abspath(os.path.join(ws, config.output_dir))
    if config.dataset.path and not os.path.isabs(config.dataset.path):
        config.dataset.path = os.path.abspath(os.path.join(ws, config.dataset.path))
    if config.dataset.reg_path and not os.path.isabs(config.dataset.reg_path):
        config.dataset.reg_path = os.path.abspath(os.path.join(ws, config.dataset.reg_path))
    return config


def _workspace_runs_dir(name: str) -> str:
    return os.path.join(_workspace_dir(name), "runs")


def _workspace_logs_dir(name: str) -> str:
    return os.path.join(_workspace_dir(name), "logs")


def _workspace_cache_dir(name: str) -> str:
    return os.path.join(_workspace_dir(name), "train_cache")


def _workspace_run_dirs(name: str):
    base = _workspace_runs_dir(name)
    if not os.path.isdir(base):
        return
    seen = set()
    for dirpath, dirnames, filenames in os.walk(base):
        dirnames[:] = [item for item in dirnames if not item.startswith(".")]
        if "metrics.jsonl" not in filenames:
            continue
        real = os.path.realpath(dirpath)
        if real in seen:
            continue
        seen.add(real)
        yield base, dirpath


def _workspace_latest_run(name: str) -> dict | None:
    best = None
    for root, directory in _workspace_run_dirs(name):
        rows = _parse_metrics(directory)
        steps = [r for r in rows if "train" in r]
        info = {
            "name": os.path.basename(directory),
            "path": _rel(directory),
            "step": (steps[-1]["step"] if steps else 0),
            "total": (steps[-1].get("total") if steps else 0),
            "mtime": os.path.getmtime(os.path.join(directory, "metrics.jsonl")),
        }
        if best is None or info["mtime"] > best["mtime"]:
            best = info
    return best


def _workspace_cache_items(name: str) -> int:
    path = os.path.join(_workspace_cache_dir(name), "index.json")
    if not os.path.isfile(path):
        return 0
    try:
        with open(path, encoding="utf-8") as handle:
            return len(json.load(handle).get("items", []))
    except Exception:
        return 0



def _workspace_info(name: str, include_size: bool = True) -> dict:
    path = _workspace_dir(name)
    stat = os.stat(path)
    try:
        data = _workspace_data(name)
    except HTTPException:
        data = {}
    latest = _workspace_latest_run(name)
    info = {
        "name": name,
        "path": _rel(path),
        "config_path": _rel(_workspace_config_path(name)),
        "kind": "workspace",
        "dataset": str(getPath(data, "dataset.path") or ""),
        "steps": latest["step"] if latest else 0,
        "latest_run": latest["name"] if latest else "",
        "cache_items": _workspace_cache_items(name),
        "mtime": stat.st_mtime,
    }
    if include_size:
        info["size_bytes"] = _dir_size(path)
        info["size_mb"] = round(info["size_bytes"] / 2**20, 1)
    return info


def getPath(object_: dict, path: str):
    value = object_
    for key in str(path).split("."):
        value = value.get(key) if isinstance(value, dict) else None
    return value


def _section(data: dict, key: str) -> dict:
    """Return ``data[key]`` as a mutable mapping.

    The form serialiser writes an all-blank section as ``key:`` (an empty
    mapping), which YAML loads as ``None``; ``setdefault`` would then hand back
    the ``None`` and blow up on assignment.
    """
    value = data.get(key)
    if not isinstance(value, dict):
        value = {}
        data[key] = value
    return value


def _portable_workspace_config(name: str, text: str) -> str:
    """Rewrite cache/output paths under this workspace to relative names."""
    try:
        data = yaml.safe_load(text) or {}
    except yaml.YAMLError:
        return text
    if not isinstance(data, dict):
        return text
    ws = os.path.realpath(_workspace_dir(name))
    for section, key, rel in (
        (data.get("cache") or {}, "dir", "train_cache"),
        (data.get("cache") or {}, "reg_dir", "reg_cache"),
        (data, "output_dir", "runs"),
    ):
        if not isinstance(section, dict) and section is not data:
            continue
        raw = str(section.get(key) or "").strip()
        if not raw:
            continue
        full = os.path.realpath(raw if os.path.isabs(raw) else os.path.join(ROOT, raw))
        if full == ws or full.startswith(ws + os.sep):
            section[key] = rel
    return yaml.safe_dump(data, allow_unicode=True, sort_keys=False, width=10**9)


def _run_config_snapshot(name: str, run: str) -> tuple[str, str, str]:
    """Return (yaml_text, path, source) for one run.

    One source only: ``runs/<run>/config.yaml``, written by the trainer when the
    run starts. Earlier versions guessed from the run's name -- a same-named
    file in ``logs/`` or ``configs/``, then the workspace defaults -- which
    labelled a CLI run with whatever file happened to share its name. A run
    without the snapshot reports no config instead of borrowing someone else's.
    """
    path = os.path.join(_workspace_runs_dir(name), run, "config.yaml")
    if not os.path.isfile(path):
        return "", "", ""
    with open(path, encoding="utf-8") as handle:
        text = handle.read()
    return _portable_workspace_config(name, text), _rel(path), "run"


def _run_checkpoints(directory: str) -> list[dict]:
    folder = os.path.join(directory, "checkpoints")
    out = []
    if not os.path.isdir(folder):
        return out
    losses = {}
    for row in _parse_metrics(directory):
        step = row.get("step")
        if step:
            losses[int(step)] = {"train": row.get("train"), "eval": row.get("eval")}
    for filename in os.listdir(folder):
        match = re.fullmatch(r"step(\d+)\.safetensors", filename)
        if match:
            step = int(match.group(1))
            out.append({"step": step, "file": filename,
                        "size_mb": round(os.path.getsize(os.path.join(folder, filename)) / 2**20, 1),
                        **losses.get(step, {})})
        elif filename == "final.safetensors":
            out.append({"step": "final", "file": filename,
                        "size_mb": round(os.path.getsize(os.path.join(folder, filename)) / 2**20, 1)})
    out.sort(key=lambda item: (item["step"] == "final", int(item["step"]) if item["step"] != "final" else 0))
    return out


def _workspace_run_list(name: str) -> list[dict]:
    runs = []
    for root, directory in _workspace_run_dirs(name):
        rows = _parse_metrics(directory)
        train_rows = [row for row in rows if "train" in row]
        eval_rows = [row for row in rows if "eval" in row]
        run = os.path.basename(directory.rstrip(os.sep))
        config_yaml, config_path, config_source = _run_config_snapshot(name, run)
        try:
            config_data = yaml.safe_load(config_yaml) if config_yaml else None
        except yaml.YAMLError:
            config_data = None
        sample_dir = os.path.join(directory, "samples")
        samples = 0
        if os.path.isdir(sample_dir):
            samples = len([f for f in os.listdir(sample_dir) if f.endswith(".png")])
        log_path = os.path.join(_workspace_logs_dir(name), f"{run}.log")
        runs.append({
            "name": run,
            "label": f"{name}/{run}",
            "workspace": name,
            "path": _rel(directory),
            "step": train_rows[-1]["step"] if train_rows else 0,
            "total": train_rows[-1].get("total") if train_rows else 0,
            "last_train": train_rows[-1].get("train") if train_rows else None,
            "last_eval": eval_rows[-1].get("eval") if eval_rows else None,
            "mtime": os.path.getmtime(os.path.join(directory, "metrics.jsonl")),
            "samples": samples,
            "config_yaml": config_yaml,
            "config": config_data,
            "config_path": config_path,
            "config_source": config_source,
            "log_path": _rel(log_path) if os.path.isfile(log_path) else "",
            "checkpoints": _run_checkpoints(directory),
        })
    runs.sort(key=lambda item: item["mtime"], reverse=True)
    return runs


def _workspaces_payload() -> dict:
    root = _workspace_root()
    workspaces = []
    for name in _workspace_names():
        try:
            workspaces.append(_workspace_info(name))
        except OSError:
            continue
    return {
        "root": _rel(root),
        "workspaces": workspaces,
        "default": workspaces[0]["name"] if workspaces else "",
    }

def _next_run_id(parent: str) -> str:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    run_id = stamp
    suffix = 2
    while os.path.exists(os.path.join(parent, run_id)):
        run_id = f"{stamp}-{suffix}"
        suffix += 1
    return run_id


def _workspace_target_run(workspace: str) -> tuple[str, str, str]:
    runs = _workspace_runs_dir(workspace)
    run_id = _next_run_id(runs)
    return workspace, run_id, os.path.join(runs, run_id)

def _dir_usage(path: str) -> dict:
    total = 0
    files = 0
    if os.path.isdir(path):
        for root, _dirs, names in os.walk(path):
            for name in names:
                try:
                    total += os.path.getsize(os.path.join(root, name))
                    files += 1
                except OSError:
                    continue
    return {"bytes": total, "mb": round(total / 2**20, 1), "files": files}

def _has_live_cli_training(name: str) -> bool:
    workspace = os.path.abspath(_workspace_dir(name))
    for item in os.listdir("/proc"):
        if not item.isdigit():
            continue
        if _is_cli_train_process(int(item), workspace):
            return True
    return False
