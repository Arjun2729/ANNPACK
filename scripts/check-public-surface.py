#!/usr/bin/env python3
"""Fail CI when private/operator residue leaks into ANNPack's public surface."""

from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHECKER_PATH = Path(__file__).resolve().relative_to(ROOT).as_posix()

# Working notes and operator material. `launch/` as a whole is gitignored; these
# names are listed so that committing one fails loudly rather than silently.
FORBIDDEN_TRACKED_PATHS = {
    "CLAUDE.md",
}
FORBIDDEN_TRACKED_PREFIXES = ("launch/", "experiments/", ".claude/")
REQUIRED_TRACKED_PATHS = {
    "CONTRIBUTING.md",
}

# Test corpora and conformance artifacts are fixtures, not the operating
# surface: they legitimately contain odd bytes and stale strings, so the
# live-surface checks below skip them. ALWAYS_FORBIDDEN_TEXT is NOT skipped — a
# personal home directory is a privacy leak wherever it is published, and
# recorded tool output is exactly where one survives unnoticed.
EXCLUDED_PREFIXES = (
    "benches/history/",
    "spec/test-vectors/",
    "spec/conformance/",
    "fixtures/",
)

TEXT_SUFFIXES = {
    ".rs",
    ".py",
    ".js",
    ".mjs",
    ".ts",
    ".sh",
    ".md",
    ".json",
    ".toml",
    ".yaml",
    ".yml",
    ".html",
    # Recorded tool output. Captured stdout/stderr carries whatever absolute
    # paths the operator's machine had; scanning it is the whole point.
    ".txt",
    ".log",
    ".stderr",
    ".stdout",
}

# Checked in every tracked text file, including historical evidence.
ALWAYS_FORBIDDEN_TEXT = {
    "/Users/": "personal absolute macOS path",
    "/home/": "personal absolute Linux path",
}

# Absolute paths that belong to shared CI infrastructure rather than to a
# person. They are neutralized before the checks above run.
IMPERSONAL_ABSOLUTE_PREFIXES = ("/home/runner/",)

# Tool marketing footers carry no technical information and are not part of the
# public surface.
FORBIDDEN_LIVE_TEXT = {
    # `launch/` is gitignored operator material, so any public link into it is
    # dangling by construction.
    "launch/": "link into untracked operator material",
    "🤖 Generated with": "automated generation footer",
}


def tracked_paths() -> list[str]:
    output = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode(
        "utf-8"
    )
    return [path for path in output.split("\0") if path]


def is_scannable_text(path: str) -> bool:
    # The checker necessarily contains the strings it searches for. Its behavior
    # is exercised by CI, while the rest of the tree is scanned as data.
    if path == CHECKER_PATH:
        return False
    return Path(path).suffix.lower() in TEXT_SUFFIXES


def is_live_text(path: str) -> bool:
    return is_scannable_text(path) and not path.startswith(EXCLUDED_PREFIXES)


def main() -> int:
    tracked = set(tracked_paths())
    failures: list[str] = []

    for path in sorted(FORBIDDEN_TRACKED_PATHS & tracked):
        failures.append(f"forbidden tracked path: {path}")
    for path in sorted(p for p in tracked if p.startswith(FORBIDDEN_TRACKED_PREFIXES)):
        failures.append(f"forbidden tracked path: {path}")
    for path in sorted(REQUIRED_TRACKED_PATHS - tracked):
        failures.append(f"required public path missing: {path}")

    for path in sorted(tracked):
        if not is_scannable_text(path):
            continue
        file_path = ROOT / path
        try:
            text = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for prefix in IMPERSONAL_ABSOLUTE_PREFIXES:
            text = text.replace(prefix, "")
        checks = dict(ALWAYS_FORBIDDEN_TEXT)
        if is_live_text(path):
            checks.update(FORBIDDEN_LIVE_TEXT)
        for needle, description in checks.items():
            if needle in text:
                failures.append(f"{path}: {description}: {needle!r}")

    if failures:
        print("public-surface audit failed:")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print(
        "public-surface audit passed: no private operator files, personal paths, "
        "stale launch links, or tool marketing footers in live tracked text"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
