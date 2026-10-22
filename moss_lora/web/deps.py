# moved from app.py: L46-59, L127-162, L563
"""Paths, settings snapshot and stateless helpers shared by the web layer.

The mutable settings state (``SETTINGS`` / ``RESOLVED`` / ``FIRST_RUN`` and the
in-memory literal API key) lives here and only here.  Every other module reads
it as ``deps.SETTINGS`` so a settings save is visible everywhere instead of
handing out a stale copy.
"""

from __future__ import annotations

import copy
import os
import re
import sys
import threading
import uuid

from fastapi import HTTPException

# Fail fast for the deployer, not with a cryptic import error later.  The UI
# never exposes interpreter selection: the product uses its bundled .venv.
if not os.path.isfile(os.path.join(sys.prefix, "pyvenv.cfg")):
    print("请用 ./.venv/bin/python 启动（先运行 ./setup.sh 创建环境）", file=sys.stderr)
    raise SystemExit(1)

from .. import settings as runtime_settings


ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
WORKSPACE_MARKER = "workspace.yaml"
RESERVED_WORKSPACES = {"configs", "train_cache", "reg_cache", "runs", "logs", "baseline", "exports"}
SETTINGS_REL = os.path.relpath(runtime_settings.SETTINGS_PATH, ROOT)
NAME_RE = runtime_settings.NAME_RE
STEP_RE = re.compile(
    r"step (\d+)/(\d+)\s+loss ([0-9.]+)\s+avg ([0-9.]+)\s+lr ([0-9.eE+-]+)\s+([0-9.]+)s\s+peak ([0-9.]+) GiB"
)

_SETTINGS_LOCK = threading.RLock()
FIRST_RUN = runtime_settings.ensure_settings_file()
SETTINGS = runtime_settings.load_settings()
RESOLVED = runtime_settings.apply_environment(SETTINGS)

def _settings_snapshot() -> tuple[dict, dict]:
    with _SETTINGS_LOCK:
        return copy.deepcopy(SETTINGS), copy.deepcopy(RESOLVED)


def _settings_rel(path: str) -> str:
    try:
        rel = os.path.relpath(path, ROOT)
        return rel if not rel.startswith("..") else path
    except ValueError:
        return path


def _rel(path: str) -> str:
    return _settings_rel(path)


def _safe_name(value: str | None, label: str = "名称") -> str:
    """工作区 / run / 配置的名字：不允许空、不允许路径分隔符。"""
    raw = str(value or "").strip()
    if raw.endswith(".yaml"):
        raw = raw[:-5]
    if not raw:
        raise HTTPException(400, f"{label}不能为空")
    if not NAME_RE.match(raw):
        raise HTTPException(400, f"{label}只能使用字母、数字、点、下划线和短横线（1-64 字符）")
    return raw


def _write_text(path: str, text: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.{os.getpid()}.{uuid.uuid4().hex[:6]}.tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        handle.write(text)
    os.replace(tmp, path)


# 直接填的密钥只活在这个进程里：用户填环境变量名才落盘，填密钥不保存
_LITERAL_API_KEY = ""
