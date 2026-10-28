"""HTTP errors as codes.

Every message the WebUI can show is a code plus its parameters. The code
travels in ``detail.code``, the rendered Chinese in ``detail.message`` so logs,
the CLI and any older client still read a sentence. The four i18n packs
translate ``err.<code>``; the templates live in :mod:`moss_lora.problems`
because the CLI needs them too.

    raise fail(404, "err.workspace.not_found", name=name)
"""

from __future__ import annotations

from fastapi import HTTPException

from ..problems import (CODES, Problem, json_safe, problem,  # noqa: F401  (re-exported)
                        render)

__all__ = ["CODES", "Problem", "error_detail", "fail", "json_safe", "problem", "render"]


def error_detail(code: str, **params) -> dict:
    """The JSON body every error carries: code, parameters, fallback message."""
    safe = json_safe(params)
    return {"code": code, "params": safe, "message": render(code, safe)}


def fail(status: int, code: str, headers: dict | None = None, **params) -> HTTPException:
    """Build the exception; callers ``raise fail(...)``."""
    return HTTPException(status_code=status, detail=error_detail(code, **params), headers=headers)
