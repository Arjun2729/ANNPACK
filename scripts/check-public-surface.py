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

# Commit authorship is enforced from this commit forward, not retroactively.
# v0.5.1 and v0.6.1 were published before the rule existed and are already
# tagged; rewriting them to satisfy a new check would mean re-pointing published
# tags, which spec/COMPATIBILITY.md forbids for a reason that outweighs tidy
# metadata. Every commit built on top of this one must carry a real identity.
#
# A commit SHA rather than a date: the first version of this rule used
# `--since=<date>`, which git resolves in local time, so a probe commit made at
# 00:43 IST fell outside a cutoff that read as UTC and the gate passed a
# deliberately bad commit. A SHA has no timezone.
IDENTITY_BASELINE_COMMIT = "84fcc920c95a10c05b9f3b1e27a61904e5f6d753"

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


def commit_identity_failures() -> list[str]:
    """Reject commit authorship that leaks a machine or names no publisher.

    The tree-scanning checks above look only at file contents, so they said
    nothing while v0.5.1 and v0.6.1 were published with authors like
    `Jans <anantha@Jans-MacBook-Air.local>`: git had no configured identity and
    fell back to the local account and hostname. That publishes an operator's
    laptop name, and it varies between commits as the machine's display name
    changes -- two releases of the same project carried two different authors.

    It matters more here than it would elsewhere. This project's entire claim is
    that a consumer can tell who published what; commits that identify no
    publisher undercut it before anyone reads the specification.
    """
    baseline = subprocess.run(
        ["git", "merge-base", "--is-ancestor", IDENTITY_BASELINE_COMMIT, "HEAD"],
        cwd=ROOT,
        capture_output=True,
    )
    if baseline.returncode != 0:
        # Not the publishing lineage. The released history descends from a known
        # root commit; a working checkout with unrelated history is not what
        # ships, and enforcing authorship on it would fail on every run until
        # people stopped reading the output.
        print(
            "note: commit-identity check skipped -- this history does not "
            f"descend from {IDENTITY_BASELINE_COMMIT[:12]}, so it is not the "
            "published lineage"
        )
        return []

    output = subprocess.check_output(
        [
            "git",
            "log",
            f"{IDENTITY_BASELINE_COMMIT}..HEAD",
            "--format=%H%x00%an%x00%ae%x00%cn%x00%ce",
        ],
        cwd=ROOT,
    ).decode("utf-8")
    failures = []
    for line in output.splitlines():
        if not line:
            continue
        commit, author_name, author_email, committer_name, committer_email = line.split(
            "\0"
        )
        short = commit[:12]
        for role, name, email in (
            ("author", author_name, author_email),
            ("committer", committer_name, committer_email),
        ):
            # A hostname-derived address is what git invents when nothing is
            # configured; it is never a deliverable address.
            if email.endswith(".local") or email.endswith(".localdomain"):
                failures.append(
                    f"{short}: {role} email is machine-derived: {email!r} "
                    f"(set user.email, e.g. your GitHub noreply address)"
                )
            elif "@" not in email:
                failures.append(f"{short}: {role} email is not an address: {email!r}")
            if not name.strip():
                failures.append(f"{short}: {role} name is empty")
    return failures


def main() -> int:
    tracked = set(tracked_paths())
    failures: list[str] = []
    failures.extend(commit_identity_failures())

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
