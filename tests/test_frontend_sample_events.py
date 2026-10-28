"""The page adds the samples an event names, instead of reloading the grid.

    .venv/bin/python tests/test_frontend_sample_events.py

The dashboard used to rebuild the sample grid on a timer, which listed the
directory at whatever moment the poll landed. A sample still being encoded was
listed, requested, and decoded as a truncated PNG. The trainer's ``samples``
event now carries the finished paths, and the poll leaves the grid alone while
that stream is live.
"""

from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC = os.path.join(ROOT, "moss_lora", "web", "static")


def read(*parts: str) -> str:
    with open(os.path.join(STATIC, *parts), encoding="utf-8") as handle:
        return handle.read()


def main() -> int:
    jobsse = read("js", "components", "jobsse.js")
    dashboard = read("js", "pages", "dashboard.js")
    dom = read("js", "core", "dom.js")

    assert "applySampleEvent" in jobsse, "the samples event must patch the grid"
    assert "import { loadActiveTraining, loadDashboard, applySampleEvent }" in jobsse
    samples_case = jobsse.split("case 'samples':", 1)[1].split("case 'stage':", 1)[0]
    assert "applySampleEvent(event)" in samples_case
    assert "loadDashboard(" in samples_case, "keep the fallback for payloads the page cannot apply"
    assert samples_case.index("applySampleEvent(event)") < samples_case.index("loadDashboard(")

    assert "export function applySampleEvent" in dashboard
    assert "state.sampleCompareData" in dashboard
    assert "renderSampleCompareGrid();" in dashboard.split("export function applySampleEvent", 1)[1]
    assert "streamOwnsSamples" in dashboard, "the poll must hand the grid to the event stream"
    assert "dataChanged && !streamOwnsSamples" in dashboard

    assert "&v=' + THUMB_STAMP" in dom, "thumbnails keep their page stamp, which the cache hinges on"

    with open(os.path.join(ROOT, "moss_lora", "web", "routers", "samples.py"), encoding="utf-8") as handle:
        assert 'v: str = ""' in handle.read(), "the endpoint accepts the page stamp"

    print("sample event wiring ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
