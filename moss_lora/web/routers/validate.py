# moved from app.py: L2458-2581
"""``POST /api/validate``: read-only preflight for the training form."""

from __future__ import annotations

import os

import yaml
from fastapi import APIRouter, Request

from ...config import expand
from ..deps import ROOT, _rel
from ..services.cache_cmd import _cache_summary
from ..services.workspace import _config_from_data, _resolve_workspace, _workspace_target_run

router = APIRouter()


@router.post("/api/validate")
async def api_validate(request: Request):
    """Read-only preflight: YAML, settings backend, cache and target run."""
    body = await request.json()
    text = body.get("yaml", "")
    if not text.strip():
        return {"ok": False, "yaml_error": {"message": "YAML 为空", "line": None, "column": None},
                "problems": [], "warnings": [], "cache": None, "run": None}
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        mark = getattr(exc, "problem_mark", None)
        return {
            "ok": False,
            "yaml_error": {
                "message": str(getattr(exc, "problem", exc)),
                "line": mark.line + 1 if mark else None,
                "column": mark.column + 1 if mark else None,
            },
            "problems": [], "warnings": [], "cache": None, "run": None,
        }
    if not isinstance(data, dict):
        return {"ok": False, "yaml_error": {"message": "配置必须是一个 YAML 映射", "line": 1, "column": 1},
                "problems": [], "warnings": [], "cache": None, "run": None}

    workspace = _resolve_workspace(body.get("workspace"))

    problems: list[str] = []
    warnings: list[str] = []
    try:
        config = _config_from_data(workspace, data)
        problems = config.validate()
    except Exception as exc:  # malformed types must not 500 the UI
        return {"ok": False, "yaml_error": {"message": f"配置结构错误: {exc}", "line": None, "column": None},
                "problems": [], "warnings": [], "cache": None, "run": None, "next_tab": "train"}

    reg_path = expand(config.dataset.reg_path or "")
    if reg_path:
        reg_full = reg_path if os.path.isabs(reg_path) else os.path.abspath(os.path.join(ROOT, reg_path))
        if not os.path.isdir(reg_full):
            problems.append(f"dataset.reg_path not found: {reg_full}")
    if int(config.dataset.reg_repeat or 1) < 1:
        problems.append("dataset.reg_repeat 必须是 >= 1 的整数")

    cache = _cache_summary(config) if config.cache.dir else {
        "dir": "", "items": 0, "ready": False, "index_ready": False,
        "preview_ready": False, "index_mtime": None,
    }
    if not config.cache.dir:
        problems.append("cache.dir 为空，无法检查缓存")

    prompts = [p for p in (config.preview.prompts or []) if str(p).strip()]
    if not prompts and not (cache or {}).get("preview_ready"):
        warnings.append("样图提示词为空：不会生成 preview.safetensors，训练时不会出样图。")
    if not config.train.eval_every:
        warnings.append("验证间隔为 0：训练不会写入 eval 损失。")
    if not config.train.sample_every:
        warnings.append("采样间隔为 0：训练不会生成样图。")

    # 缓存的问题单独说清楚，否则预检只是「不通过」而没有理由
    cache_problem = ""
    if config.cache.dir and cache:
        if not cache.get("index_ready"):
            cache_problem = f"缓存未生成（{cache.get('dir')}）：先到「缓存」页生成 latent 与文本嵌入。"
        elif cache.get("legacy_index"):
            cache_problem = (
                f"缓存索引过旧（{cache.get('dir')}）：训练集与正则集混在一起，"
                "请到「缓存」页重建一次。"
            )
        elif cache.get("missing_text"):
            cache_problem = (
                f"缓存缺文本嵌入（{cache['missing_text']}/{cache.get('items', 0)} 条）："
                "只做了 latent，请在「缓存」页点「一键生成缓存（latent + 文本）」。"
            )
    if cache_problem:
        problems.append(cache_problem)

    # First missing step wins.  The training page owns the three model paths,
    # so model problems point back there; dataset/cache have their own tabs.
    model_problems = [item for item in problems if item.startswith("model.")]
    dataset_problems = [item for item in problems if item.startswith("dataset.")]
    reg_problems = [item for item in dataset_problems if item.startswith("dataset.reg_")]
    focus_field = ""
    if model_problems:
        for key, field_id in (("dit", "cfg-model-dit"), ("vae", "cfg-model-vae"), ("text_encoder", "cfg-model-te")):
            if not getattr(config.model, key, ""):
                focus_field = field_id
                break
        next_tab = "train"
        focus_field = focus_field or "cfg-model-dit"
    elif reg_problems:
        next_tab = "dataset"
        focus_field = "cfg-reg-path"
    elif dataset_problems or not config.dataset.path:
        next_tab = "dataset"
        focus_field = "cfg-dataset-path"
    elif not cache["ready"]:
        next_tab = "cache"
    else:
        next_tab = "train"

    profile = workspace
    preview_workspace, preview_run_id, target = _workspace_target_run(workspace)
    workspace_label = workspace
    ok = not problems and bool(cache["ready"])
    return {
        "workspace": workspace_label,
        "ok": ok,
        "yaml_error": None,
        "problems": problems,
        "warnings": warnings,
        "cache": cache,
        "next_tab": next_tab,
        "focus_field": focus_field,
        "run": {
            "name": preview_run_id,
            "run_id_preview": preview_run_id,
            "profile": profile,
            "workspace": workspace_label,
            "output_dir": _rel(target),
            "exists": False,
            "will_rename": False,
        },
    }
