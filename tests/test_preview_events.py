"""A finished preview reaches the page as paths, not as a directory listing.

    .venv/bin/python tests/test_preview_events.py

The trainer logs ``preview -> [...]`` only after every image in the batch is
written. The launcher turns that line into a ``samples`` event carrying the
finished files, so the dashboard can add those cells without asking the server
to list ``samples/`` at a moment when a file may still be growing.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from moss_lora.web.deps import ROOT  # noqa: E402
from moss_lora.web.jobs import Job  # noqa: E402
from moss_lora.web.services.launcher import _preview_event  # noqa: E402


def job(workspace: str = "ws", run_name: str = "run-1") -> Job:
    return Job(id="job-1", kind="train", config_path="", command=[],
               workspace=workspace, run_name=run_name)


def sample(step: int, index: int) -> str:
    return os.path.join(ROOT, "workspaces", "ws", "runs", "run-1", "samples",
                        f"step{step:06d}_{index}.png")


def check_paths_reach_the_event():
    line = f"preview -> {[sample(3000, 0), sample(3000, 1)]} (12.3s)"
    event = _preview_event(job(), line)
    assert event["type"] == "samples"
    assert event["step"] == 3000
    assert event["run"] == "ws/run-1"
    assert [os.path.basename(path) for path in event["paths"]] == [
        "step003000_0.png", "step003000_1.png"]
    assert all(not os.path.isabs(path) for path in event["paths"]), "the page addresses paths relative to the repo"
    print("preview line: step, run and relative paths")


def check_other_previews_stay_plain():
    baseline = os.path.join(ROOT, "workspaces", "ws", "baseline", "baseline_0.png")
    event = _preview_event(job(), f"preview -> {[baseline]} (3.0s)")
    assert "paths" not in event and "step" not in event
    skipped = _preview_event(job(), "preview skipped: no cache")
    assert "paths" not in skipped
    print("baseline and skipped previews: no sample payload")


def check_unparsable_line_is_harmless():
    event = _preview_event(job(), "preview -> not a list (1.0s)")
    assert event["type"] == "samples" and "paths" not in event
    print("unparsable line: still a samples event, no payload")


def main() -> int:
    check_paths_reach_the_event()
    check_other_previews_stay_plain()
    check_unparsable_line_is_harmless()
    print("preview event checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
