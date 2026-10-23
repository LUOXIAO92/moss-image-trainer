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
from ..errors import error_detail
from ...problems import Problem, problem

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
        # Do not name this `problem`: the same function calls the imported
        # `problem()` helper further down, and a local assignment would shadow it
        # for the whole function, raising UnboundLocalError on every clean YAML.
        yaml_problem = str(getattr(exc, "problem", exc))
        if mark:
            detail = error_detail("err.yaml.syntax", line=mark.line + 1, column=mark.column + 1,
                                  problem=yaml_problem)
        else:
            detail = error_detail("err.yaml.syntax_plain", problem=yaml_problem)
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

    problems: list[Problem] = []
    warnings: list[Problem] = []
    try:
        config = _config_from_data(workspace, data)
        problems = config.validate()
    except Exception as exc:  # malformed types must not 500 the UI
        return {"ok": False,
                "yaml_error": {**error_detail("err.config.structure", error=exc),
                               "line": None, "column": None},
                "problems": [], "warnings": [], "cache": None, "run": None, "next_tab": "train"}

    reg_path = expand(config.dataset.reg_path or "")
    if reg_path:
        reg_full = reg_path if os.path.isabs(reg_path) else os.path.abspath(os.path.join(ROOT, reg_path))
        if not os.path.isdir(reg_full):
            problems.append(problem("err.dataset.reg_path_missing", path=reg_full))
    if int(config.dataset.reg_repeat or 1) < 1:
        problems.append(problem("err.dataset.reg_repeat"))

    cache = _cache_summary(config) if config.cache.dir else {
        "dir": "", "items": 0, "ready": False, "index_ready": False,
        "preview_ready": False, "index_mtime": None,
    }
    if not config.cache.dir:
        problems.append(problem("err.cache.dir_empty"))

    prompts = [p for p in (config.preview.prompts or []) if str(p).strip()]
    if not prompts and not (cache or {}).get("preview_ready"):
        warnings.append(problem("err.warning.prompts_empty"))
    if not config.train.eval_every:
        warnings.append(problem("err.warning.eval_off"))
    if not config.train.sample_every:
        warnings.append(problem("err.warning.samples_off"))

    # 缓存的问题单独说清楚，否则预检只是「不通过」而没有理由
    cache_problem = None
    if config.cache.dir and cache:
        if not cache.get("index_ready"):
            cache_problem = problem("err.cache.not_built", dir=cache.get("dir"))
        elif cache.get("legacy_index"):
            cache_problem = problem("err.cache.legacy", dir=cache.get("dir"))
        elif cache.get("missing_text"):
            cache_problem = problem("err.cache.missing_text", dir=cache.get("dir"),
                                    missing=cache["missing_text"], items=cache.get("items", 0))
    if cache_problem:
        problems.append(cache_problem)

    # First missing step wins.  The training page owns the three model paths,
    # so model problems point back there; dataset/cache have their own tabs.
    model_codes = {
        "err.config.field_empty", "err.config.field_variable", "err.config.field_missing",
        "err.config.text_encoder_no_config", "err.config.text_encoder_kind",
        "err.config.te_config_missing", "err.config.te_config_no_config",
    }
    reg_codes = {"err.dataset.reg_path_missing", "err.dataset.reg_repeat"}
    dataset_codes = {"err.config.dataset_path_empty", "err.dataset.missing",
                     "err.config.caption_source"}
    model_problems = [item for item in problems if item.code in model_codes]
    reg_problems = [item for item in problems if item.code in reg_codes]
    dataset_problems = [item for item in problems if item.code in dataset_codes]
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
        "problems": [item.render() for item in problems],
        "problem_codes": [item.detail() for item in problems],
        "warnings": [item.render() for item in warnings],
        "warning_codes": [item.detail() for item in warnings],
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
