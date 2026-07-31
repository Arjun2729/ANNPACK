#!/usr/bin/env python3
"""Fail CI when private/operator residue leaks into ANNPack's public surface."""

from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHECKER_PATH = Path(__file__).resolve().relative_to(ROOT).as_posix()

FORBIDDEN_TRACKED_PATHS = {
    "CLAUDE.md",
    "launch/LAUNCH-SURFACE.md",
    "launch/RELEASE-READINESS.md",
    "launch/google-okf/OUTREACH.md",
    "launch/LAUNCH-GATES.md",
    "spec/LAUNCH-GATES.md",
}
REQUIRED_TRACKED_PATHS = {
    "CONTRIBUTING.md",
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

# Attribution is disclosed, not suppressed: a `Co-Authored-By` trailer naming a
# model is required practice here (see CONTRIBUTING.md), so it is deliberately
# absent from this list. What stays forbidden is tool *marketing* residue, which
# carries no provenance information and is not a disclosure.
FORBIDDEN_LIVE_TEXT = {
    "/Users/": "personal absolute macOS path",
    "spec/LAUNCH-GATES.md": "link to a removed launch checklist",
    "launch/LAUNCH-GATES.md": "link to a removed launch checklist",
    "launch/RELEASE-READINESS.md": "link to the private release ledger",
    "launch/LAUNCH-SURFACE.md": "link to private launch material",
    "launch/google-okf/OUTREACH.md": "link to private outreach material",
    "🤖 Generated with": "automated generation footer",
}


def tracked_paths() -> list[str]:
    output = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode(
        "utf-8"
    )
    return [path for path in output.split("\0") if path]


def is_live_text(path: str) -> bool:
    # The checker necessarily contains the strings it searches for. Its behavior
    # is exercised by CI, while the rest of the live tree is scanned as data.
    if path == CHECKER_PATH or path.startswith(EXCLUDED_PREFIXES):
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
        "stale launch links, or tool marketing footers in live tracked text"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
