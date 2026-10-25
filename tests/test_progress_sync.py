"""Progress shown by the WebUI must come from the metrics file, not the sparse beat.

    .venv/bin/python tests/test_progress_sync.py

Two regressions made the header and the dashboard disagree: the trainer started
writing ``tok ...`` between the step time and the peak, which stopped the SSE
parser, and ``scan_runs`` let the heartbeat's step (written every 20 steps)
override the metrics step. A short run therefore sat at step 0 while the
dashboard read the real step from ``metrics.jsonl``.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from moss_lora import runtime  # noqa: E402
from moss_lora.web.deps import STEP_RE  # noqa: E402


def check_current_and_legacy_log_lines() -> None:
    current = (
        "step 4/10  loss 0.3620  avg 0.3515  lr 1.00e-04  20.85s  "
        "tok 37765 (img 37240 + txt 525)  peak 14.6 GiB "
        "(reserved 14.8, device 16.5)"
    )
    match = STEP_RE.search(current)
    assert match, current
    assert match.groups() == (
        "4", "10", "0.3620", "0.3515", "1.00e-04", "20.85", "14.6", "14.8", "16.5",
    ), match.groups()

    legacy = (
        "step 1/10  loss 0.3253  avg 0.3253  lr 1.00e-04  37.90s  "
        "peak 12.6 GiB"
    )
    match = STEP_RE.search(legacy)
    assert match, legacy
    assert match.groups() == (
        "1", "10", "0.3253", "0.3253", "1.00e-04", "37.90", "12.6", None, None,
    ), match.groups()


def check_scan_runs_prefers_metrics_step() -> None:
    with tempfile.TemporaryDirectory() as root:
        run_dir = os.path.join(root, "runs", "run-short")
        os.makedirs(run_dir)
        now = time.time()
        runtime.write_json_atomic(
            os.path.join(run_dir, "training.json"),
            {
                "pid": os.getpid(),
                "step": 0,
                "total": 10,
                "status": "running",
                "phase": "train",
                "run_id": "run-short",
                "started": now,
                "time": now,
            },
        )
        with open(os.path.join(run_dir, "metrics.jsonl"), "w", encoding="utf-8") as handle:
            handle.write(json.dumps({
                "step": 4, "total": 10, "train": 0.362, "avg": 0.3515,
                "lr": 1e-4, "sec": 20.85, "peak": 14.6,
                "reserved": 14.8, "device": 16.5,
            }) + "\n")

        rows = runtime.scan_runs(root)
        assert len(rows) == 1, rows
        row = rows[0]
        assert row["step"] == 4, row
        assert row["total"] == 10, row
        assert row["last"]["step"] == 4, row["last"]
        assert abs(row["last"]["loss"] - 0.362) < 1e-9, row["last"]
        assert row["last"]["vram_device"] == 16.5, row["last"]


def main() -> int:
    check_current_and_legacy_log_lines()
    check_scan_runs_prefers_metrics_step()
    print("progress sync checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
