#!/usr/bin/env python3
"""Fail CI when private/operator residue leaks into ANNPack's public surface."""

from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

FORBIDDEN_TRACKED_PATHS = {
    "CLAUDE.md",
    "launch/LAUNCH-SURFACE.md",
    "launch/google-okf/OUTREACH.md",
    "spec/LAUNCH-GATES.md",
}
REQUIRED_TRACKED_PATHS = {
    "CONTRIBUTING.md",
    "launch/LAUNCH-GATES.md",
}

# Historical evidence and test corpora are records, not the current operating
# surface. They may legitimately quote old paths or tool output.
EXCLUDED_PREFIXES = (
    "launch/evidence/",
    "benches/history/",
    "spec/test-vectors/",
    "spec/conformance/",
    "fixtures/",
    "attic/",
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
}

FORBIDDEN_LIVE_TEXT = {
    "/Users/": "personal absolute macOS path",
    "spec/LAUNCH-GATES.md": "stale link to moved launch checklist",
    "launch/LAUNCH-SURFACE.md": "link to private founder launch surface",
    "launch/google-okf/OUTREACH.md": "link to private outreach material",
    "Co-Authored-By: Claude": "AI-tool co-author trailer",
    "Generated with Claude": "AI-tool generation footer",
    "Generated with ChatGPT": "AI-tool generation footer",
    "🤖 Generated with": "automated generation footer",
}


def tracked_paths() -> list[str]:
    output = subprocess.check_output(
        ["git", "ls-files", "-z"], cwd=ROOT
    ).decode("utf-8")
    return [path for path in output.split("\0") if path]


def is_live_text(path: str) -> bool:
    if path.startswith(EXCLUDED_PREFIXES):
        return False
    return Path(path).suffix.lower() in TEXT_SUFFIXES


def main() -> int:
    tracked = set(tracked_paths())
    failures: list[str] = []

    for path in sorted(FORBIDDEN_TRACKED_PATHS & tracked):
        failures.append(f"forbidden tracked path: {path}")
    for path in sorted(REQUIRED_TRACKED_PATHS - tracked):
        failures.append(f"required public path missing: {path}")

    for path in sorted(tracked):
        if not is_live_text(path):
            continue
        file_path = ROOT / path
        try:
            text = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for needle, description in FORBIDDEN_LIVE_TEXT.items():
            if needle in text:
                failures.append(f"{path}: {description}: {needle!r}")

    if failures:
        print("public-surface audit failed:")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print(
        "public-surface audit passed: no private operator files, personal paths, "
        "stale launch links, or AI-tool attribution footers in live tracked text"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
