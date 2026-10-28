"""An error body must survive JSON encoding.

    .venv/bin/python tests/test_error_envelope.py

``fail(404, "err.thumbnail.failed", error=exc)`` put the exception object into
``detail.params``. The Chinese message rendered fine, then Starlette serialized
the body, ``json.dumps`` hit ``TypeError: Object of type OSError is not JSON
serializable`` inside the exception handler, and uvicorn reported an ASGI
failure. The browser saw a dead request instead of a 404. One truncated sample
image was enough to trigger it.
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from moss_lora.problems import Problem, problem  # noqa: E402
from moss_lora.web.errors import error_detail, fail  # noqa: E402


def check_exception_parameter():
    detail = error_detail("err.thumbnail.failed", error=OSError("image file is truncated"))
    json.dumps(detail)  # the crash was here
    assert detail["params"]["error"] == "image file is truncated"
    assert "image file is truncated" in detail["message"]
    print("exception parameter: serializable and rendered")


def check_http_exception_body():
    exc = fail(404, "err.thumbnail.failed", error=RuntimeError("boom"))
    body = json.dumps({"detail": exc.detail})
    assert "boom" in body and exc.status_code == 404
    print("http body: 404 with a serializable detail")


def check_nested_parameters():
    detail = error_detail("err.config.structure", error=ValueError("bad"),
                          extra={"nested": [KeyError("k"), 3]})
    json.dumps(detail)
    assert detail["params"]["extra"]["nested"] == ["'k'", 3]
    print("nested parameters: coerced depth-first")


def check_problem_detail():
    item = problem("err.job.stop_failed", error=PermissionError("denied"))
    payload = item.detail()
    json.dumps(payload)
    assert payload["params"]["error"] == "denied"
    assert isinstance(Problem("err.job.stop_failed", {"error": OSError("x")}).detail()["params"]["error"], str)
    print("problem detail: same guarantee")


def main() -> int:
    check_exception_parameter()
    check_http_exception_body()
    check_nested_parameters()
    check_problem_detail()
    print("error envelope checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
