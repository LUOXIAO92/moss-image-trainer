# moved from app.py: L364-372, L903-959, L1032-1057, L2002-2064
"""System or GPU probing, directory sizing, metrics parsing and /proc scans."""

from __future__ import annotations

import json
import os
import sys
from typing import Any

from ... import settings as runtime_settings
from ...config import Config
from ..deps import ROOT, SETTINGS_REL, _rel, _settings_snapshot


def _dir_size(path: str) -> int:
    total = 0
    for dirpath, _, filenames in os.walk(path):
        for filename in filenames:
            try:
                total += os.path.getsize(os.path.join(dirpath, filename))
            except OSError:
                pass
    return total

GPU_QUERY = ("name,memory.total,memory.used,utilization.gpu,"
             "temperature.gpu,power.draw,power.limit")


def _nvidia_smi(query: str) -> list[str]:
    """Rows from ``nvidia-smi``, without ever touching torch.cuda.

    The web process must not create a CUDA context: it would pin hundreds of MB
    of VRAM while a training subprocess needs the card. ``mem_get_info()`` and
    ``torch.cuda.is_available()`` both create one, so every VRAM number the
    server reports comes from this subprocess instead.
    """
    import subprocess as _sp

    out = _sp.run(
        ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=5,
    ).stdout.strip().splitlines()
    # 没有 GPU / 驱动被禁（WSL 掉 /dev/dxg）时 nvidia-smi 会把错误打到 stdout，
    # 行里没有逗号；别让它变成 IndexError 糊在 /api/system 上。
    return [line for line in out if line.count(",") >= 2]


def gpu_total_gb() -> float | None:
    """Total VRAM of the first card in GB, or None when the driver cannot say.

    Used to resolve ``vram.preset: auto`` outside a GPU process, so the estimate
    matches the tier a real run would pick.
    """
    # Three columns on purpose: the row filter needs a comma count that a
    # one-column query ("memory.total") would never reach.
    rows = _nvidia_smi("index,name,memory.total")
    if not rows:
        return None
    try:
        return float(rows[0].split(",")[-1].strip()) / 1024
    except (IndexError, ValueError):
        return None


def _system() -> dict:
    info: dict[str, Any] = {"python": sys.executable, "torch": "", "cuda": "", "gpu": {}}
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda"] = torch.version.cuda or ""
        rows = _nvidia_smi(GPU_QUERY)
        if rows:
            parts = [part.strip() for part in rows[0].split(",")]
            name, total_mb, used_mb = parts[0], parts[1], parts[2]

            def num(index: int) -> float | None:
                if index >= len(parts):
                    return None
                try:
                    return float(parts[index])
                except ValueError:
                    return None

            total = float(total_mb) / 1024
            used = float(used_mb) / 1024
            info["gpu"] = {
                "name": name,
                "total_gb": round(total, 1),
                "free_gb": round(total - used, 1),
                "used_gb": round(used, 1),
                # [N/A] 的字段留 None：没装驱动权限时温度/功耗读不到
                "utilization": num(3),
                "temperature": num(4),
                "power": num(5),
                "power_limit": num(6),
            }
    except Exception as exc:
        info["error"] = str(exc)
    cfg = Config().resolve_paths()
    info["models"] = {
        name: {"path": getattr(cfg.model, name), "exists": os.path.exists(getattr(cfg.model, name))}
        for name in ("dit", "text_encoder", "vae")
    }
    settings, resolved = _settings_snapshot()
    info["settings"] = {
        "path": SETTINGS_REL,
        "settings": settings,
        "resolved": resolved,
        "status": runtime_settings.settings_status(settings),
    }
    return info

def _run_samples(directory: str) -> list[str]:
    sample_dir = os.path.join(directory, "samples")
    if not os.path.isdir(sample_dir):
        return []
    files = sorted(
        f for f in os.listdir(sample_dir)
        if f.startswith("step") and f.endswith(".png") and f != "trajectory.png"
    )
    return [_rel(os.path.join(sample_dir, f)) for f in files]


def _parse_metrics(directory: str) -> list[dict]:
    path = os.path.join(directory, "metrics.jsonl")
    rows: list[dict] = []
    if not os.path.isfile(path):
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

def _cmdline(pid: int) -> str:
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as handle:
            return handle.read().replace(b"\x00", b" ").decode("utf-8", "replace")
    except OSError:
        return ""


def _is_training_process(pid: int, workspace: str) -> bool:
    cmd = _cmdline(pid)
    parts = cmd.split()
    if not parts or "moss_lora.cli" not in parts or "train" not in parts:
        return False
    # only the python process itself, never a shell wrapper that happens to
    # contain the same words in its command line
    if not parts[0].endswith("python"):
        return False
    return workspace in cmd


def _scan_training_pids(workspace: str) -> list[int]:
    found = []
    try:
        for entry in os.listdir("/proc"):
            if not entry.isdigit():
                continue
            pid = int(entry)
            if _is_training_process(pid, workspace):
                found.append(pid)
    except OSError:
        pass
    return found


def _proc_cmdline(pid: int) -> list[str]:
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as handle:
            raw = handle.read()
    except OSError:
        return []
    return [part.decode("utf-8", errors="replace") for part in raw.split(b"\0") if part]


def _is_cli_train_process(pid: int, workspace: str) -> bool:
    """Only match the real python CLI process, never its bash wrapper."""
    cmdline = _proc_cmdline(pid)
    if len(cmdline) < 2:
        return False
    argv0 = os.path.basename(cmdline[0])
    if not argv0.endswith("python"):
        return False
    joined = " ".join(cmdline)
    if "moss_lora.cli" not in joined or "train" not in cmdline:
        return False
    workspace = os.path.abspath(workspace)
    rel = os.path.relpath(workspace, ROOT)
    if workspace in joined or rel in joined:
        return True
    try:
        cwd = os.path.realpath(f"/proc/{pid}/cwd")
    except OSError:
        return False
    return cwd == workspace or cwd.startswith(workspace + os.sep)
