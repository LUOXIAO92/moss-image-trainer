"""Run a JSON list of argv commands in order, without a shell.

Windows has no POSIX shell: the ``bash`` on PATH is usually WSL, which cannot
execute the venv's Windows interpreter, and ``cmd.exe`` quoting is its own
minefield. The job launcher passes argv lists here instead, so paths with
spaces and Windows separators need no shell quoting.
"""

from __future__ import annotations

import json
import subprocess
import sys


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        return 2
    for command in json.loads(argv[1]):
        print("$ " + " ".join(command), flush=True)
        code = subprocess.call(command)
        if code:
            return code
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
