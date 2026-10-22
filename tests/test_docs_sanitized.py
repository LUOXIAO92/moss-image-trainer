#!/usr/bin/env python3
"""Scan the public files for anything that ties the repository to one machine.

Two rule sets live in the repository and stay generic on purpose: they describe
shapes (absolute paths, credential formats), never the words themselves. Anything
specific to a machine -- usernames, dataset names, private services -- belongs in
an untracked list next to this file::

    tests/.sanitize-local.txt      # one rule per line: <regex>\\t<label>

The script loads that file when it exists, so a fresh clone still runs, just with
the generic rules only.

    .venv/bin/python tests/test_docs_sanitized.py

HARD hits fail the run. SOFT hits only warn: measured numbers belong in
experiments/, not in public prose, but they are not a leak of identity.
"""
from __future__ import annotations

import glob
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOCAL_RULES = os.path.join(ROOT, "tests", ".sanitize-local.txt")

TARGETS = ["README.md", "AGENTS.md", "settings.example.yaml"]
TARGET_GLOBS = ["README.*.md"]
TARGET_DIRS = ["docs", "skills"]

# Shapes only: these say "this looks like a path / a token", not "this is my path".
HARD = [
    (r"/home/[A-Za-z0-9._-]+", "absolute home path"),
    (r"/mnt/[a-z]/", "mounted drive path"),
    (r"\bsk-[A-Za-z0-9]{20,}", "API token"),
    (r"\bghp_[A-Za-z0-9]{20,}", "GitHub token"),
    (r"\bAKIA[0-9A-Z]{16}\b", "AWS key id"),
    (r"\bBearer\s+[A-Za-z0-9._-]{20,}", "bearer token"),
]

SOFT = [
    (r"\d+(?:\.\d+)?\s?(?:GiB|MiB)", "measured VRAM"),
    (r"\d+(?:\.\d+)?\s?(?:s/step|it/s|tok/s|img/s|cap/s)", "measured throughput"),
    (r"\d+\s?°C", "measured temperature"),
    (r"\b(?:3090|4090|5090|A100|H100|H200|L40S)\b", "GPU model"),
]

SKIP_DIRS = {".git", ".venv", "experiments", "third_party", "workspaces", "node_modules", ".tmp"}
TEXT_SUFFIXES = {".md", ".yaml", ".yml", ".txt"}


def local_rules() -> list[tuple[str, str]]:
    """Machine-specific rules kept out of version control."""
    if not os.path.isfile(LOCAL_RULES):
        return []
    rules = []
    with open(LOCAL_RULES, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            pattern, _, label = line.partition("\t")
            rules.append((pattern, label or pattern))
    return rules


def iter_files() -> list[str]:
    found = []
    for name in TARGETS:
        path = os.path.join(ROOT, name)
        if os.path.isfile(path):
            found.append(path)
    for pattern in TARGET_GLOBS:
        found.extend(sorted(glob.glob(os.path.join(ROOT, pattern))))
    for directory in TARGET_DIRS:
        base = os.path.join(ROOT, directory)
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
            for filename in filenames:
                if os.path.splitext(filename)[1] in TEXT_SUFFIXES:
                    found.append(os.path.join(dirpath, filename))
    return sorted(set(found))


def scan() -> int:
    rules = local_rules()
    hard_hits: list[str] = []
    soft_hits: list[str] = []
    files = iter_files()

    for path in files:
        rel = os.path.relpath(path, ROOT)
        try:
            text = open(path, encoding="utf-8").read()
        except (OSError, UnicodeDecodeError):
            continue
        for pattern, label in HARD + rules:
            for match in re.finditer(pattern, text):
                line = text.count("\n", 0, match.start()) + 1
                hard_hits.append(f"{rel}:{line}: {label} -> {match.group(0)!r}")
        for pattern, label in SOFT:
            for match in re.finditer(pattern, text):
                line = text.count("\n", 0, match.start()) + 1
                soft_hits.append(f"{rel}:{line}: {label} -> {match.group(0)!r}")

    for hit in soft_hits:
        print(f"warn  {hit}")
    for hit in hard_hits:
        print(f"FAIL  {hit}")

    extra = f" +{len(rules)} local rule(s)" if rules else " (no local rules)"
    print(f"scanned {len(files)} file(s){extra}: {len(hard_hits)} hard hit(s), {len(soft_hits)} warning(s)")
    if hard_hits:
        print("Host-specific content found: move the wording to experiments/ or delete it.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(scan())
