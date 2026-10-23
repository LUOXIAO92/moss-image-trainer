# moved from app.py: L566-703, L1231-1234, L1463-1477
"""Caption jobs: service probing, API-key resolution and worker arguments."""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request


from .. import deps
from ..deps import _write_text
from .launcher import _python
from ..errors import fail


_ENV_NAME_RE = re.compile(r"^[A-Z_][A-Z0-9_]{0,63}$")

def resolve_api_key(caption: dict) -> tuple[str, str]:
    """Return ``(key, source)``; source is env / literal / legacy / missing / none.

    ``caption["api_key"]`` 存的是环境变量名（全大写那种）。名字在进程环境里没有值时
    报 ``missing``——那不是密钥，别拿去当密钥用。旧版 settings.yaml 里直接存的密钥
    长得不像变量名，按 ``legacy`` 继续用。
    """
    stored = str(caption.get("api_key") or "").strip()
    if stored:
        value = os.environ.get(stored, "").strip()
        if value:
            return value, "env"
    if deps._LITERAL_API_KEY:
        return deps._LITERAL_API_KEY, "literal"
    if stored:
        if _ENV_NAME_RE.match(stored):
            return "", "missing"
        return stored, "legacy"
    return "", "none"

def _probe_service(base_url: str, key: str) -> tuple[str, str, list[str]]:
    """ok / unauthorized / unreachable / error：拿 /models 验一次地址和密钥。"""
    if not base_url:
        return "unreachable", "还没填服务地址", []
    url = base_url.rstrip("/") + "/models"
    request = urllib.request.Request(url, headers={"Authorization": f"Bearer {key}"})
    try:
        with urllib.request.urlopen(request, timeout=8) as response:
            raw = response.read()
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            return "unauthorized", f"HTTP {exc.code}", []
        return "error", f"HTTP {exc.code}", []
    except Exception as exc:  # 连不上不算密钥错，允许先存着
        return "unreachable", type(exc).__name__, []
    ids: list[str] = []
    try:
        payload = json.loads(raw.decode("utf-8"))
        for item in payload.get("data") or []:
            if isinstance(item, dict) and item.get("id"):
                ids.append(str(item["id"]))
    except Exception:
        pass
    return "ok", "", ids


def _caption_env(job_env: dict, caption: dict) -> dict:
    """提示词生成的视觉服务参数走子进程环境变量，不写命令行（避免密钥进日志/ps）。"""
    base_url = str(caption.get("base_url") or "").strip()
    model = str(caption.get("model") or "").strip()
    if base_url:
        job_env["VLM_BASE_URL"] = base_url
    if model:
        job_env["VLM_MODEL"] = model
    key, _ = resolve_api_key(caption)
    if key:
        job_env["VLM_API_KEY"] = key
    return job_env


CAPTION_STYLES = ("tag_caption", "official")
CAPTION_STYLE_LABELS = {"tag_caption": "tag + caption", "official": "官方描述"}
CAPTION_DATASETS = ("train", "reg")


def _caption_args(images: str, caption: dict, styles: list[str], prompt_tag: str = "",
                  prompt_official: str = "", role: str = "train", overwrite: bool = False,
                  output: str = "") -> list[str]:
    """提示词生成 worker 的参数；服务地址/密钥/模型在设置页，风格和数据集在任务里选。"""
    args = [
        "-m", "moss_lora.captioner",
        "--images", images,
        "--styles", ",".join(styles),
        "--workers", str(caption.get("workers", 6)),
        "--retries", str(caption.get("retries", 3)),
        "--temperature", str(caption.get("temperature", 1.0)),
        "--top-p", str(caption.get("top_p", 0.95)),
        "--max-tokens", str(caption.get("max_tokens", 16384)),
    ]
    if output:
        args += ["--out", output]
    trigger = str(caption.get("trigger") or "").strip()
    if trigger:
        args += ["--trigger", trigger]
    if not caption.get("thinking", True):
        args += ["--no-thinking"]
    if overwrite:
        args += ["--overwrite"]
    base_url = str(caption.get("base_url") or "").strip()
    model = str(caption.get("model") or "").strip()
    if base_url:
        args += ["--base-url", base_url]
    if model:
        args += ["--model", model]
    # API key 只走环境变量（子进程 env），不出现在命令行里，避免进日志/ps
    if role == "reg":
        args += ["--role", "reg"]
    if prompt_tag:
        args += ["--prompt-tags", prompt_tag]
    if prompt_official:
        args += ["--prompt-natural", prompt_official]
    return args


def _caption_command(config, caption: dict, log_dir: str, stamp: str, datasets: list[str],
                     styles: list[str], overwrite: bool) -> list[str]:
    python = _python()

    def prompt_file(style: str, setting: str) -> str:
        text = str(caption.get(setting) or "")
        if not text.strip():
            return ""
        path = os.path.join(log_dir, f"caption-prompt-{style}-{stamp}.txt")
        _write_text(path, text)
        return path

    tag_prompt = prompt_file("tag", "prompt_tags")
    official_prompt = prompt_file("official", "prompt_natural")
    commands: list[list[str]] = []
    if "train" in datasets:
        commands.append([python] + _caption_args(
            os.path.abspath(os.path.expanduser(config.dataset.path)), caption, styles,
            tag_prompt, official_prompt, overwrite=overwrite))
    if "reg" in datasets:
        reg_path = os.path.abspath(os.path.expanduser(str(config.dataset.reg_path or "")))
        if not reg_path or not os.path.isdir(reg_path):
            raise fail(400, "err.caption.reg_missing")
        commands.append([python] + _caption_args(
            reg_path, caption, styles, tag_prompt, official_prompt, role="reg",
            overwrite=overwrite))
    if len(commands) == 1:
        return commands[0]
    quoted = " && ".join(" ".join(f'"{part}"' for part in cmd) for cmd in commands)
    return ["bash", "-c", quoted]

def _caption_prompt_defaults() -> dict:
    from ...captioner import NATURAL_PROMPT, TAGS_PROMPT

    return {"tags": TAGS_PROMPT, "natural": NATURAL_PROMPT}

def _caption_selection(body: dict) -> tuple[list[str], list[str], bool]:
    """数据集 / 风格 / 是否覆盖：提示词生成任务的三个选项。"""
    raw_datasets = body.get("datasets")
    if raw_datasets is None:
        raise fail(400, "err.caption.dataset_required")
    datasets = [item for item in raw_datasets if item in CAPTION_DATASETS]
    if not datasets:
        raise fail(400, "err.caption.dataset_required")
    raw_styles = body.get("styles")
    if raw_styles is None:
        raise fail(400, "err.caption.style_required")
    styles = [item for item in raw_styles if item in CAPTION_STYLES]
    if not styles:
        raise fail(400, "err.caption.style_required")
    return datasets, styles, bool(body.get("overwrite"))
