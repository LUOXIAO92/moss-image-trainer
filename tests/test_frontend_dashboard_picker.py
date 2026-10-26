"""The dashboard has one run picker, and resume reaches the job API.

    .venv/bin/python tests/test_frontend_dashboard_picker.py
"""

from __future__ import annotations

import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC = os.path.join(ROOT, "moss_lora", "web", "static")


def read(path: str) -> str:
    with open(os.path.join(STATIC, path), encoding="utf-8") as handle:
        return handle.read()


def main() -> int:
    html = read("pages/dashboard.html")
    dashboard_js = read("js/pages/dashboard.js")
    events_js = read("js/events.js")
    jobsse_js = read("js/components/jobsse.js")

    assert 'id="dash-run"' not in html
    assert 'for="dash-run"' not in html
    assert html.count('id="dash-config-select"') == 1
    assert 'id="btn-resume"' in html
    assert 'id="dash-resume-status"' in html

    # Exact selector only; #dash-run-status and #dash-run-name are still valid
    # elements inside the run facts panel.
    for source, name in ((dashboard_js, "dashboard.js"), (events_js, "events.js")):
        assert not re.search(r"#dash-run(?![-\w])", source), f"{name} still queries the removed picker"

    assert "renderDashboardConfigOptions" in dashboard_js
    assert "loadDashboard(runLabel" in events_js
    assert "resume: options.resume" in jobsse_js
    assert "baseline: options.baseline" in jobsse_js
    assert "state.dashboard.run" in dashboard_js

    print("dashboard picker/resume wiring ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
