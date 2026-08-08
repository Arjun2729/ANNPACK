#!/usr/bin/env python3
"""Fail when published documentation contradicts the implementation.

Documentation drifts silently. `web/index.html` advertised `v0.4.0-rc4` through
four releases, and `SECURITY.md` described freshness as unimplemented design
while `rust/src/release.rs` was verifying channel-state statements in CI.

Each assertion below is *conditioned on a probe of the code*, so it activates
when the implementation reaches a state that makes a claim false rather than
whenever someone edits prose. Full-document snapshots were deliberately avoided:
they break on rewording and teach people to regenerate them without reading.
"""

from __future__ import annotations

import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SELF = Path(__file__).resolve().relative_to(ROOT).as_posix()

#: Documentation a reader of the published project actually encounters.
LIVE_DOCS = [
    "README.md",
    "spec/SECURITY.md",
    "spec/CORE-v1.0-draft.md",
    "spec/COMPATIBILITY.md",
    "spec/EVIDENCE-v1.md",
    "spec/RELEASE-v1.md",
    "spec/FORMAT-v3.md",
    "spec/PROTOCOL-v1.md",
    "spec/decisions/0004-freshness-and-revocation.md",
    "web/index.html",
]


@dataclass
class Check:
    name: str
    #: Code fact that makes this check meaningful. Skipped when absent.
    probe: str
    probe_files: list[str]
    #: Regexes that must not appear in live documentation.
    forbidden: list[str]
    #: (file, regex) that must appear, so a correction cannot be silently deleted.
    required: list[tuple[str, str]]
    why: str


CHECKS = [
    Check(
        name="freshness is implemented, so nothing may call it unimplemented",
        probe="pub fn verify_channel_state",
        probe_files=["rust/src/release.rs"],
        forbidden=[
            r"[Nn]o (implemented )?freshness or revocation mechanism",
            r"freshness[^.\n]{0,40}\b(is|remains) (design only|unimplemented)",
            r"rollback resistance (is|as) an unsolved problem",
            r"\bdesign only, with no implementation\b",
        ],
        required=[
            ("spec/SECURITY.md", r"RELEASE-v1"),
            ("spec/EVIDENCE-v1.md", r"RELEASE-v1"),
            ("README.md", r"RELEASE-v1"),
        ],
        why="channel-state verification exists and runs in CI",
    ),
    Check(
        name="trust roles are separated, so no doc may describe one key for all",
        probe="ROLE_EMERGENCY_REVOCATION",
        probe_files=["rust/src/trust.rs"],
        forbidden=[
            r"same publisher key that signs packs",
            r"trusting a publisher is one decision, not two",
            r"one key (that )?(signs|performs) (every|all) (trust )?roles?",
        ],
        required=[
            (
                "spec/RELEASE-v1.md",
                r"may withdraw and may not promote|`current` honoured",
            ),
        ],
        why="four separate roles exist with independent thresholds",
    ),
    Check(
        name="rollback state is a scoped sequence and digest, not a root",
        probe="highest_sequence",
        probe_files=["rust/src/release.rs"],
        forbidden=[
            r"highest (accepted )?root",
            r"track(ing)? the newest accepted version/root",
        ],
        required=[
            ("spec/RELEASE-v1.md", r"highest_sequence"),
            ("spec/RELEASE-v1.md", r"statement_digest"),
        ],
        why="retained state records a sequence and a digest, keyed by scope",
    ),
    Check(
        name="the witnessed policy denies while transparency is unimplemented",
        # Meaningful only while nothing can produce `Verified`. A Sigsum
        # adapter (`transparency.rs`, ADR-0007) now does, so this check is
        # inactive: the probe is present and the old "always denies" wording
        # no longer applies. Kept, not deleted, as a record of the property
        # and in case a future refactor ever makes `Verified` unreachable
        # again without the documentation being updated to match.
        probe="TransparencyEvidence::Verified",
        probe_files=["rust/src/transparency.rs"],
        forbidden=[],
        required=[
            ("spec/RELEASE-v1.md", r"always denies in this release"),
        ],
        why="the CLI only ever reports transparency as unavailable",
        # Inverted: the probe must be ABSENT for the requirement to apply.
    ),
    Check(
        name="revocation is a status decision, never an integrity failure",
        probe="Currency::Revoked",
        probe_files=["rust/src/policy.rs"],
        forbidden=[
            r"revocation[^.\n]{0,60}\b(invalidates?|corrupts?)\b[^.\n]{0,30}\bartifact\b",
            r"revoked artifact[^.\n]{0,40}fails integrity",
            r"integrity[^.\n]{0,30}fails?[^.\n]{0,30}because[^.\n]{0,20}revoked",
        ],
        required=[
            ("spec/RELEASE-v1.md", r"Revocation MUST NOT change it"),
            ("spec/SECURITY.md", r"never an integrity failure"),
        ],
        why="revocation denies use while reporting artifact_integrity as valid",
    ),
    Check(
        name="the published version claim matches Cargo.toml",
        probe="",
        probe_files=[],
        forbidden=[],
        required=[],
        why="the demo page advertised v0.4.0-rc4 across four releases",
    ),
]


def probe_present(check: Check) -> bool:
    if not check.probe:
        return True
    for name in check.probe_files:
        path = ROOT / name
        if path.exists() and check.probe in path.read_text(encoding="utf-8"):
            return True
    return False


def live_documents() -> dict[str, str]:
    tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode()
    present = {p for p in tracked.split("\0") if p}
    return {
        name: (ROOT / name).read_text(encoding="utf-8")
        for name in LIVE_DOCS
        if name in present
    }


def version_claims(documents: dict[str, str]) -> list[str]:
    """Every advertised version string must match the package version."""
    cargo = (ROOT / "Cargo.toml").read_text(encoding="utf-8")
    version = re.search(r'^version = "([^"]+)"', cargo, re.M).group(1)
    failures = []
    patterns = [
        (r"Version `v([0-9][^`]*)`", "README version banner"),
        (r"candidate format · v([0-9][^\s<]*)", "demo page version badge"),
        (r"Arjun2729/ANNPACK@v([0-9][^\s`]*)", "README Action pin"),
    ]
    for name, text in documents.items():
        for pattern, label in patterns:
            for found in re.findall(pattern, text):
                if found != version:
                    failures.append(
                        f"{name}: {label} says v{found}, Cargo.toml says v{version}"
                    )
    return failures


def main() -> int:
    documents = live_documents()
    failures: list[str] = []

    for check in CHECKS:
        if check.name.startswith("the published version"):
            failures.extend(version_claims(documents))
            continue

        applies = probe_present(check)
        if check.name.startswith("the witnessed policy"):
            # This one applies while the probe is ABSENT.
            applies = not applies
        if not applies:
            continue

        for name, text in documents.items():
            if name == SELF:
                continue
            for pattern in check.forbidden:
                match = re.search(pattern, text)
                if match:
                    failures.append(
                        f"{name}: says {match.group(0)!r} but {check.why} "
                        f"[{check.name}]"
                    )
        for name, pattern in check.required:
            text = documents.get(name)
            if text is None:
                failures.append(f"{name} is missing but {check.why}")
            elif not re.search(pattern, text):
                failures.append(
                    f"{name}: must state {pattern!r} because {check.why} "
                    f"[{check.name}]"
                )

    if failures:
        print("documentation-consistency audit failed:")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print(
        f"documentation-consistency audit passed: {len(CHECKS)} implementation "
        "claims agree with published documentation"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
