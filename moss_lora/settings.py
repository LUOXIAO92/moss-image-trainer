"""Persistent runtime settings (``settings.yaml``).

The product ships its own ``.venv``; the launcher and every child process use
``sys.executable``.  Settings only describe local paths and one runtime switch:

    paths:
      workspaces: workspaces     # workspace root
    runtime:
      allocator: expandable      # expandable | async | off
      gc_threshold: 0.9          # 仅 expandable 模式使用

``allocator`` 决定 ``PYTORCH_CUDA_ALLOC_CONF``：

* ``expandable``（默认）→ ``expandable_segments:True,garbage_collection_threshold:0.9``
  PyTorch 的缓存分配器在长训里容易攒下碎片，这个组合让分配器按需回收/重映射。
* ``async`` → ``backend:cudaMallocAsync``
  直接用 CUDA 的流式分配器，碎片由驱动管；部分驱动/WSL 上行为不同，需实测。
* ``off`` → 不设置，沿用 PyTorch 默认分配器。

``settings.yaml`` is machine-local and ignored by git.  It is created from
``settings.example.yaml`` on first run if it does not exist.
"""

from __future__ import annotations

import copy
import os
import re
import shutil
from typing import Any

import yaml

from .problems import Problem, problem

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SETTINGS_PATH = os.path.join(PROJECT_ROOT, "settings.yaml")
EXAMPLE_PATH = os.path.join(PROJECT_ROOT, "settings.example.yaml")
# 工作区 / run / 配置的名字规则（不允许路径分隔符等）
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")

ALLOCATORS = ("expandable", "async", "off")
DEFAULT_ALLOCATOR = "expandable"
DEFAULT_GC_THRESHOLD = 0.9

# 文本编码会把 9B 权重一次性拷进显存；expandable_segments 下这块大拷贝在部分
# 驱动/WSL 上直接报 "Failed to create GPU mapping"。
# 文本编码不受显存碎片影响，固定用默认分配器，不跟随上面的设置。
DEFAULT_ALLOCATOR_JOBS = ("encode-text", "encode-prompt")

# 打标 worker 的默认参数：采样值对齐官方 caption 建议（thinking 开、temp 1、top_p .95）
def default_caption_settings() -> dict:
    return {
        "trigger": "",
        "workers": 6,
        "retries": 3,
        "temperature": 1.0,
        "top_p": 0.95,
        "max_tokens": 16384,
        "thinking": True,
        # 本地视觉服务（OpenAI 兼容）；留空则回退到进程环境变量
        "base_url": "",
        # 只存环境变量名；直接填的密钥只在本次运行内存里用，不落盘
        "api_key": "",
        "model": "",
        # 留空 = 用代码里的内置模板；改动后按 JSON 覆盖
        "prompt_tags": "",
        "prompt_natural": "",
    }


def default_settings() -> dict:
    """No machine-specific default values at all."""
    return {
        "paths": {"workspaces": "workspaces"},
        "runtime": {"allocator": DEFAULT_ALLOCATOR, "gc_threshold": DEFAULT_GC_THRESHOLD},
        "caption": default_caption_settings(),
        # 界面语言：内置 zh/en/ja/ko，或用「设置 → 界面语言」导入的任意语言包
        "ui": {"language": "zh"},
    }


def _merge(base: dict, override: dict) -> dict:
    result = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = value
    return result


def normalize(data: dict) -> dict:
    settings = _merge(default_settings(), data if isinstance(data, dict) else {})
    paths = settings.get("paths") if isinstance(settings.get("paths"), dict) else {}
    settings["paths"] = {
        "workspaces": str(paths.get("workspaces") or "workspaces").strip() or "workspaces",
    }
    runtime = settings.get("runtime") if isinstance(settings.get("runtime"), dict) else {}
    allocator = str(runtime.get("allocator") or DEFAULT_ALLOCATOR).strip().lower()
    if allocator not in ALLOCATORS:
        allocator = DEFAULT_ALLOCATOR
    try:
        threshold = float(runtime.get("gc_threshold", DEFAULT_GC_THRESHOLD))
    except (TypeError, ValueError):
        threshold = DEFAULT_GC_THRESHOLD
    settings["runtime"] = {
        "allocator": allocator,
        "gc_threshold": min(0.99, max(0.0, threshold)),
    }
    caption = settings.get("caption") if isinstance(settings.get("caption"), dict) else {}
    defaults = default_caption_settings()
    try:
        workers = int(caption.get("workers", defaults["workers"]))
    except (TypeError, ValueError):
        workers = defaults["workers"]
    try:
        retries = int(caption.get("retries", defaults["retries"]))
    except (TypeError, ValueError):
        retries = defaults["retries"]
    try:
        temperature = float(caption.get("temperature", defaults["temperature"]))
    except (TypeError, ValueError):
        temperature = defaults["temperature"]
    try:
        top_p = float(caption.get("top_p", defaults["top_p"]))
    except (TypeError, ValueError):
        top_p = defaults["top_p"]
    try:
        max_tokens = int(caption.get("max_tokens", defaults["max_tokens"]))
    except (TypeError, ValueError):
        max_tokens = defaults["max_tokens"]
    ui = settings.get("ui") if isinstance(settings.get("ui"), dict) else {}
    code = str(ui.get("language") or "zh").strip().lower()
    if not re.match(r"^[a-z][a-z0-9-]{1,7}$", code):
        code = "zh"
    settings["ui"] = {"language": code}
    settings["caption"] = {
        "trigger": str(caption.get("trigger") or "").strip(),
        "workers": min(32, max(1, workers)),
        "retries": min(10, max(1, retries)),
        "temperature": min(2.0, max(0.0, temperature)),
        "top_p": min(1.0, max(0.01, top_p)),
        "max_tokens": min(32768, max(256, max_tokens)),
        "thinking": bool(caption.get("thinking", defaults["thinking"])),
        "base_url": str(caption.get("base_url") or "").strip(),
        "api_key": str(caption.get("api_key") or "").strip(),
        "model": str(caption.get("model") or "").strip(),
        "prompt_tags": str(caption.get("prompt_tags") or ""),
        "prompt_natural": str(caption.get("prompt_natural") or ""),
    }
    return settings


def alloc_conf(allocator: str, gc_threshold: float = DEFAULT_GC_THRESHOLD) -> str:
    """``PYTORCH_CUDA_ALLOC_CONF`` 的取值；返回空串表示不设置。"""
    if allocator == "async":
        return "backend:cudaMallocAsync"
    if allocator == "expandable":
        return f"expandable_segments:True,garbage_collection_threshold:{gc_threshold:.2f}"
    return ""


def alloc_conf_for(kind: str, allocator: str, gc_threshold: float = DEFAULT_GC_THRESHOLD) -> str:
    """按命令选择分配器：文本编码固定用默认分配器（见 ``DEFAULT_ALLOCATOR_JOBS``）。"""
    if str(kind or "") in DEFAULT_ALLOCATOR_JOBS:
        return ""
    return alloc_conf(allocator, gc_threshold)


def resolve_path(value: str, base: str = PROJECT_ROOT) -> str:
    """Expand ``~``/environment variables; relative paths are project-relative."""
    if not value:
        return ""
    value = os.path.expandvars(os.path.expanduser(str(value)))
    if not os.path.isabs(value):
        value = os.path.join(base, value)
    return os.path.abspath(value)


def resolve_settings(settings: dict) -> dict:
    settings = normalize(settings)
    from .config import expand

    workspaces_root = resolve_path(str(expand(settings["paths"]["workspaces"])))
    return {
        "paths": {"workspaces": workspaces_root},
        "workspaces_rel": os.path.relpath(workspaces_root, PROJECT_ROOT)
        if workspaces_root.startswith(PROJECT_ROOT + os.sep) or workspaces_root == PROJECT_ROOT
        else workspaces_root,
        "runtime": copy.deepcopy(settings["runtime"]),
        "settings": copy.deepcopy(settings),
    }


def _settings_status(settings: dict) -> dict:
    settings = normalize(settings)
    resolved = resolve_settings(settings)
    return {
        "paths": {
            "workspaces": resolved["paths"]["workspaces"],
            "workspaces_exists": os.path.isdir(resolved["paths"]["workspaces"]),
        },
        "runtime": copy.deepcopy(settings["runtime"]),
    }


def ensure_settings_file(path: str = SETTINGS_PATH) -> bool:
    """Create ``settings.yaml`` from the empty example on first run.

    Returns True when the file was just created, so the UI can show the
    first-run guide instead of pretending the machine is configured.
    """
    if os.path.exists(path):
        return False
    try:
        if os.path.exists(EXAMPLE_PATH):
            shutil.copyfile(EXAMPLE_PATH, path)
        else:
            with open(path, "w", encoding="utf-8") as handle:
                yaml.safe_dump(default_settings(), handle, sort_keys=False)
    except OSError:
        return False
    return True


def load_settings(path: str | None = None) -> dict:
    path = path or SETTINGS_PATH
    try:
        with open(path, encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
    except FileNotFoundError:
        return normalize({})
    except (OSError, yaml.YAMLError):
        # A broken per-machine file must not prevent the product from starting.
        return normalize({})
    return normalize(data if isinstance(data, dict) else {})


def save_settings(settings: dict, path: str | None = None) -> dict:
    # 默认值在调用时取：测试 / 多进程都能临时指向别的文件
    path = path or SETTINGS_PATH
    settings = normalize(settings)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        yaml.safe_dump(settings, handle, sort_keys=False, allow_unicode=True)
    os.replace(tmp, path)
    return settings


def apply_environment(settings: dict) -> dict:
    """Resolve relative paths against the project root for child processes."""
    resolved = resolve_settings(settings)

    return resolved


def child_env(settings: dict, base: dict | None = None, kind: str = "") -> dict:
    env = dict(base or os.environ)
    resolved = apply_environment(settings)
    env["PYTHONUNBUFFERED"] = "1"
    env.pop("PYTORCH_CUDA_ALLOC_CONF", None)
    conf = alloc_conf_for(
        kind, resolved["runtime"]["allocator"], resolved["runtime"]["gc_threshold"]
    )
    if conf:
        env["PYTORCH_CUDA_ALLOC_CONF"] = conf
    return env


def configure_allocator(settings: dict | None = None, kind: str = "") -> str:
    """在当前进程里设置 ``PYTORCH_CUDA_ALLOC_CONF``。

    必须在任何 CUDA 分配之前调用（PyTorch 在分配器初始化时才读这个变量）。
    ``kind`` 是命令行子命令（如 ``encode-text``），文本编码会被强制成默认分配器。
    返回实际生效的值，空串表示沿用默认分配器。
    """
    resolved = apply_environment(settings if settings is not None else load_settings())
    runtime = resolved["runtime"]
    conf = alloc_conf_for(kind, runtime["allocator"], runtime["gc_threshold"])
    os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
    if conf:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = conf
    return conf


def merge_settings(current: dict, patch: dict) -> dict:
    """Deep-merge a partial settings payload onto the current settings."""
    return _merge(normalize(current), patch if isinstance(patch, dict) else {})


def validate_settings(data: dict) -> tuple[dict[str, "Problem"], list[str], dict[str, Any]]:
    """Return ``(errors, warnings, status)`` for an unsaved payload.

    Missing paths are warnings: the workspace root may be created later, and the
    three model file paths live in each workspace's config. Each error is a code
    plus parameters, keyed by the settings field it belongs to, so the browser can
    put a translated sentence next to that field.
    """
    candidate = normalize(data)
    errors: dict[str, Problem] = {}
    warnings: list[str] = []
    status = _settings_status(candidate)

    workspaces = candidate["paths"].get("workspaces", "")
    if not workspaces:
        errors["paths.workspaces"] = problem("err.settings.workspaces_empty")
    else:
        resolved = resolve_path(str(workspaces))
        if os.path.exists(resolved) and not os.path.isdir(resolved):
            errors["paths.workspaces"] = problem("err.settings.workspaces_not_dir", path=resolved)
    return errors, warnings, status


def settings_status(settings: dict) -> dict:
    return _settings_status(settings)


if __name__ == "__main__":
    # 启动脚本用：在 python 进程真正跑训练之前把 PYTORCH_CUDA_ALLOC_CONF 算出来。
    #   export PYTORCH_CUDA_ALLOC_CONF="$(.venv/bin/python -m moss_lora.settings --print-env)"
    import sys

    if "--print-env" in sys.argv:
        runtime = load_settings()["runtime"]
        print(alloc_conf(runtime["allocator"], runtime["gc_threshold"]))
