"""Front-end signal mapping: only negative exit codes are signals.

A normal failed job exits with code 1; showing ``SIGHUP`` for it was a real
regression.  This test extracts the actual functions from ``jobs.js`` and runs
them under node, so the guard is exercised rather than copied.

    .venv/bin/python tests/test_frontend_job_signal.py
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JOBS_JS = os.path.join(ROOT, "moss_lora", "web", "static", "js", "components", "jobs.js")


def function_source() -> str:
    source = open(JOBS_JS, encoding="utf-8").read()
    start = source.index("const SIGNAL_NAMES")
    end = source.index("export function jobFailureChip")
    return source[start:end].replace("export function", "function")


def main() -> int:
    node = shutil.which("node")
    if not node:
        print("skip: node not available")
        return 0
    script = function_source() + r"""
const cases = [
  ["positive exit code is not a signal", jobSignalName(1), ""],
  ["negative exit code maps", jobSignalName(-11), "SIGSEGV"],
  ["signal string passes through", jobSignalName("SIGSEGV"), "SIGSEGV"],
  ["explicit numeric signal field", jobSignal({ signal: 11 }), "SIGSEGV"],
  ["normal failure job", jobSignal({ exitCode: 1 }), ""],
  ["crash job fallback", jobSignal({ exitCode: -11 }), "SIGSEGV"],
];
const failed = cases.filter(([, actual, expected]) => actual !== expected);
if (failed.length) {
  console.error(JSON.stringify(failed));
  process.exit(1);
}
console.log("job signal mapping ok");
"""
    subprocess.run([node, "-e", script], check=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
