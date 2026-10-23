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
from ..errors import error_detail, render

router = APIRouter()


@router.post("/api/validate")
async def api_validate(request: Request):
    """Read-only preflight: YAML, settings backend, cache and target run."""
    body = await request.json()
    text = body.get("yaml", "")
    if not text.strip():
        return {"ok": False, "yaml_error": {**error_detail("err.yaml.empty"), "line": None, "column": None},
                "problems": [], "warnings": [], "cache": None, "run": None}
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        mark = getattr(exc, "problem_mark", None)
        problem = str(getattr(exc, "problem", exc))
        if mark:
            detail = error_detail("err.yaml.syntax", line=mark.line + 1, column=mark.column + 1,
                                  problem=problem)
        else:
            detail = error_detail("err.yaml.syntax_plain", problem=problem)
        return {
            "ok": False,
            "yaml_error": {**detail,
                           "line": mark.line + 1 if mark else None,
                           "column": mark.column + 1 if mark else None},
            "problems": [], "warnings": [], "cache": None, "run": None,
        }
    if not isinstance(data, dict):
        return {"ok": False,
                "yaml_error": {**error_detail("err.yaml.not_mapping"), "line": 1, "column": 1},
                "problems": [], "warnings": [], "cache": None, "run": None}

    workspace = _resolve_workspace(body.get("workspace"))

    problems: list[str] = []
    warnings: list[str] = []
    problem_codes: list[dict | None] = []
    warning_codes: list[dict | None] = []
    try:
        config = _config_from_data(workspace, data)
        problems = config.validate()
        problem_codes = [None] * len(problems)
    except Exception as exc:  # malformed types must not 500 the UI
        return {"ok": False,
                "yaml_error": {**error_detail("err.config.structure", error=exc),
                               "line": None, "column": None},
                "problems": [], "warnings": [], "cache": None, "run": None, "next_tab": "train"}

    reg_path = expand(config.dataset.reg_path or "")
    if reg_path:
        reg_full = reg_path if os.path.isabs(reg_path) else os.path.abspath(os.path.join(ROOT, reg_path))
        if not os.path.isdir(reg_full):
            problems.append(f"dataset.reg_path not found: {reg_full}")
            problem_codes.append(None)
    if int(config.dataset.reg_repeat or 1) < 1:
        problems.append(render("err.dataset.reg_repeat"))
        problem_codes.append({"code": "err.dataset.reg_repeat", "params": {}})

    cache = _cache_summary(config) if config.cache.dir else {
        "dir": "", "items": 0, "ready": False, "index_ready": False,
        "preview_ready": False, "index_mtime": None,
    }
    if not config.cache.dir:
        problems.append(render("err.cache.dir_empty"))
        problem_codes.append({"code": "err.cache.dir_empty", "params": {}})

    prompts = [p for p in (config.preview.prompts or []) if str(p).strip()]
    if not prompts and not (cache or {}).get("preview_ready"):
        warnings.append(render("err.warning.prompts_empty"))
        warning_codes.append({"code": "err.warning.prompts_empty", "params": {}})
    if not config.train.eval_every:
        warnings.append(render("err.warning.eval_off"))
        warning_codes.append({"code": "err.warning.eval_off", "params": {}})
    if not config.train.sample_every:
        warnings.append(render("err.warning.samples_off"))
        warning_codes.append({"code": "err.warning.samples_off", "params": {}})

    # 缓存的问题单独说清楚，否则预检只是「不通过」而没有理由
    cache_problem = None
    if config.cache.dir and cache:
        if not cache.get("index_ready"):
            cache_problem = {"code": "err.cache.not_built", "params": {"dir": cache.get("dir")}}
        elif cache.get("legacy_index"):
            cache_problem = {"code": "err.cache.legacy", "params": {"dir": cache.get("dir")}}
        elif cache.get("missing_text"):
            cache_problem = {"code": "err.cache.missing_text",
                             "params": {"dir": cache.get("dir"),
                                        "missing": cache["missing_text"],
                                        "items": cache.get("items", 0)}}
    if cache_problem:
        problems.append(render(cache_problem["code"], cache_problem["params"]))
        problem_codes.append(cache_problem)

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
        "problem_codes": problem_codes,
        "warnings": warnings,
        "warning_codes": warning_codes,
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
