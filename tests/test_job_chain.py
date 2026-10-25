"""The job chain runs argv lists without a shell.

Windows has no POSIX shell, and the ``bash`` on PATH is usually WSL; the cache
and caption commands must not depend on it. The chain also has to stop at the
first nonzero exit.

    .venv/bin/python tests/test_job_chain.py
"""

from __future__ import annotations

import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from moss_lora.web.services.launcher import _chain  # noqa: E402


def check_single():
    command = [sys.executable, "-c", "print('one')"]
    assert _chain([command]) == command


def check_chain():
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    commands = [
        [sys.executable, "-c", "print('one')"],
        [sys.executable, "-c", "print('two')"],
    ]
    wrapped = _chain(commands)
    assert wrapped[1:3] == ["-m", "moss_lora.web.services.chain"], wrapped
    result = subprocess.run(
        wrapped, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        env=env, text=True, encoding="utf-8", errors="replace", timeout=60,
    )
    assert result.returncode == 0, result
    assert "one" in result.stdout and "two" in result.stdout, result


def check_failure_stops():
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    commands = [
        [sys.executable, "-c", "print('one')"],
        [sys.executable, "-c", "raise SystemExit(3)"],
        [sys.executable, "-c", "print('never')"],
    ]
    result = subprocess.run(
        _chain(commands), cwd=ROOT, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, env=env, text=True, encoding="utf-8",
        errors="replace", timeout=60,
    )
    assert result.returncode == 3, result
    assert "never" not in result.stdout, result


def main():
    check_single()
    check_chain()
    check_failure_stops()
    print("job chain checks passed")


if __name__ == "__main__":
    main()
