# moved from app.py: L706-718, L962-1025
"""Cache job parts and the cache-readiness summary."""

from __future__ import annotations

import json
import os
import time


from ...config import Config, expand
from ..deps import _rel
from ..errors import fail


CACHE_PARTS = ("latent", "text")


def _cache_parts(body: dict) -> list[str]:
    """缓存任务要生成哪几样：latent / 文本，勾几个跑几个，顺序固定。"""
    raw = body.get("parts")
    if raw is None:
        raw = ["latent", "text"]
    parts = [part for part in CACHE_PARTS if part in (raw or [])]
    if not parts:
        raise fail(400, "err.cache.parts_required")
    return parts

def _cache_job(workspace_dir: str) -> dict | None:
    """The progress a cache build publishes, whoever started it.

    A build started from a terminal writes the same file as one started in the
    WebUI, so the panel can show either without a second code path. The path is
    derived from the workspace, because a config built by the web layer carries
    absolute paths but no workspace root.
    """
    from ... import runtime
    from ...cache import PROGRESS_NAME

    path = os.path.join(workspace_dir, PROGRESS_NAME) if workspace_dir else ""
    record = runtime.read_json(path) if path else None
    if not record:
        return None
    state = runtime.classify(record)
    return {
        "state": state["state"],
        "stale": state["stale"],
        "age": state["age"],
        "status": record.get("status") or "",
        "phase": record.get("phase") or "",
        "done": record.get("done"),
        "total": record.get("total"),
        "rate": record.get("rate"),
        "unit": record.get("unit") or "",
        "pid": record.get("pid"),
        "started": record.get("started") or 0.0,
        "time": record.get("time") or 0.0,
        "finished": record.get("finished") or 0.0,
        "exit_code": record.get("exit_code"),
        "cache_dir": record.get("cache_dir") or "",
        "history": record.get("history") or {},
    }


def _looks_like_cache_process(pid: int) -> bool:
    """True when the pid's command line is one of ours and is a cache build."""
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as handle:
            command = handle.read().replace(b"\0", b" ").decode("utf-8", "replace")
    except OSError:
        return False
    return "moss_lora" in command and ("cache" in command or "encode-text" in command)


def _cache_stop(workspace_dir: str) -> dict:
    """Stop a running cache build, whoever started it.

    The button used to be dead for a build started in a terminal: it asked the
    in-memory registry, which only knows jobs this server spawned. The published
    heartbeat carries the pid, and the pid is checked against its own command
    line before any signal goes out.
    """
    import signal

    from ... import runtime
    from ...cache import PROGRESS_NAME

    path = os.path.join(workspace_dir, PROGRESS_NAME) if workspace_dir else ""
    record = runtime.read_json(path) if path else None
    if not record:
        raise fail(404, "err.cache.stop_no_job")
    state = runtime.classify(record)
    if state["state"] not in ("running", "starting", "stopping"):
        raise fail(409, "err.cache.stop_finished")
    pid = int(record.get("pid") or 0)
    if pid <= 0 or not runtime.pid_alive(pid):
        raise fail(409, "err.cache.stop_dead")
    if not _looks_like_cache_process(pid):
        raise fail(409, "err.cache.stop_foreign", pid=pid)
    try:
        os.kill(pid, signal.SIGINT)
    except ProcessLookupError:
        raise fail(409, "err.cache.stop_dead") from None
    except PermissionError:
        raise fail(409, "err.cache.stop_denied") from None
    record.update({"status": "stopping", "time": time.time()})
    try:
        runtime.write_json_atomic(path, record)
    except OSError:
        pass
    return {"ok": True, "pid": pid, "state": "stopping"}


def _cache_summary(config: Config) -> dict:
    """Cache readiness without changing the historical ``items``/``ready`` keys."""
    cache_dir = os.path.abspath(os.path.expanduser(expand(config.cache.dir)))
    index_path = os.path.join(cache_dir, "index.json")
    preview_path = os.path.join(cache_dir, "preview.safetensors")
    index_ready = os.path.isfile(index_path)
    preview_ready = os.path.isfile(preview_path)
    from ...cache import empty_text_path, preview_cache_state, reg_cache_dir

    reg_dir = reg_cache_dir(config)
    empty_ready = os.path.isfile(empty_text_path(cache_dir))
    info = {
        "items": 0,
        "ready": index_ready,
        "dir": cache_dir,
        "dir_rel": _rel(cache_dir),
        "reg_dir": reg_dir,
        "reg_dir_rel": _rel(reg_dir),
        "index_ready": index_ready,
        "empty_text_ready": empty_ready,
        # preview.safetensors 只用于预览图，缺了不影响训练，前端不要当成阻塞项；
        # 但向量过期时预览会画成别的提示词，所以状态单独报给预检。
        "preview_ready": preview_ready,
        "preview_state": preview_cache_state(cache_dir, config.preview.prompts,
                                             config.preview.negative_prompt),
        "index_mtime": os.path.getmtime(index_path) if index_ready else None,
    }
    if not index_ready:
        return info
    with open(index_path, encoding="utf-8") as f:
        index = json.load(f)
    items = index.get("items", [])
    info["items"] = len(items)
    reg_items = [item for item in items if item.get("reg")]
    info["reg_items"] = len(reg_items)
    # 索引格式只认 2（训练集/正则集分目录 + 逐条 file 相对路径）；其它一律当不可用
    try:
        schema = int(index.get("schema", 0))
    except (TypeError, ValueError):
        schema = 0
    info["legacy_index"] = schema != 2
    if info["legacy_index"]:
        info["ready"] = False
    if index.get("reg_dir"):
        info["reg_dir"] = index["reg_dir"]
        info["reg_dir_rel"] = _rel(index["reg_dir"])
    info["train_items"] = info["items"] - info["reg_items"]
    # 逐条文件的实际落盘数量（训练集与正则集分目录）
    info["train_files"] = sum(
        1 for item in items
        if not item.get("reg") and os.path.isfile(os.path.join(cache_dir, item.get("file", "")))
    )
    info["reg_files"] = sum(
        1 for item in items
        if item.get("reg") and os.path.isfile(os.path.join(cache_dir, item.get("file", "")))
    )
    # 文本嵌入是第二条流水线（encode-text）；只有 latent 的时候训练会在读缓存时炸
    text_items = 0
    for item in items:
        try:
            if int(item.get("text_len") or 0) > 0:
                text_items += 1
        except (TypeError, ValueError):
            continue
    info["text_items"] = text_items
    info["missing_text"] = info["items"] - text_items
    if not info["legacy_index"] and info["missing_text"] > 0:
        info["ready"] = False
    # 尺寸规格对不上：latent 的宽高来自另一套 max_area/对齐，训练会拒绝加载
    from ...data import ALIGN, sizing_spec

    expected = sizing_spec(config.dataset.max_area, ALIGN)
    cached = index.get("sizing")
    info["sizing"] = cached or None
    info["stale_sizing"] = cached != expected
    if info["stale_sizing"]:
        info["ready"] = False
    try:
        dropout = float(getattr(config.train, "caption_dropout", 0.0) or 0.0)
    except (TypeError, ValueError):
        dropout = 0.0
    info["caption_dropout"] = dropout
    info["missing_empty_text"] = dropout > 0 and not empty_ready
    if info["missing_empty_text"]:
        info["ready"] = False
    return info
