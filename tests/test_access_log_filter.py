"""Successful API GETs are polling, not events; keep them out of the access log.

    .venv/bin/python tests/test_access_log_filter.py

State-changing requests and error responses must still be visible. The filter
only drops GET /api/... records whose status is 2xx or 3xx.
"""

from __future__ import annotations

import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from moss_lora.web.server import QuietApiReads  # noqa: E402


def record(method: str, path: str, status: int) -> logging.LogRecord:
    return logging.LogRecord(
        "uvicorn.access", logging.INFO, "", 0, "", (
            "127.0.0.1:1234", method, path, "HTTP/1.1", status,
        ), None,
    )


def main() -> int:
    quiet = QuietApiReads()
    for path in (
        "/api/system",
        "/api/jobs",
        "/api/jobs/abc/events",
        "/api/workspaces/ws/active",
        "/api/dashboard?run=x",
    ):
        assert quiet.filter(record("GET", path, 200)) is False, path

    # Mutations are the events worth keeping.
    assert quiet.filter(record("POST", "/api/jobs", 200)) is True
    assert quiet.filter(record("DELETE", "/api/workspaces/ws/baseline", 200)) is True

    # Errors stay visible, including reads that failed.
    assert quiet.filter(record("GET", "/api/system", 500)) is True
    assert quiet.filter(record("GET", "/api/jobs", 404)) is True
    assert quiet.filter(record("GET", "/api/system", 302)) is False

    # Page and static reads are not API polling.
    assert quiet.filter(record("GET", "/index.html", 200)) is True
    assert quiet.filter(record("HEAD", "/api/system", 200)) is True

    # A malformed record is never hidden.
    malformed = logging.LogRecord("uvicorn.access", logging.INFO, "", 0, "", None, None)
    assert quiet.filter(malformed) is True

    print("access log filter checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
