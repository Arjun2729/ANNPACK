#!/usr/bin/env python3
"""Fail if the committed conformance packet is not what its generator produces.

`scripts/check-conformance-stable.py` already checks that a fresh run of the
suite matches the committed report. That compares the reference implementation
against the committed *vectors*, and both sides of that comparison are
committed files, so they stay consistent with each other while drifting
together away from what the builder now emits.

Nothing checked the other direction: that `artifacts/` and `vectors/` are still
what `scripts/build-conformance-packet.sh` produces from the committed corpus.
They were not. The packet was generated when the manifest section format was 3,
the builder has emitted 4 since ADR-0005's authenticated source digest, and the
conformance pack root had silently moved. The suite kept passing because every
artifact it read was the stale one.

This is the conformance packet's equivalent of
`scripts/check-demo-reproducibility.py`, which has always rebuilt the demo packs
and required byte-identical committed output.

Three files are excluded because the generator signs with a freshly generated
key on every run, by design -- the private key is deliberately never committed,
so the signature cannot be reproduced. Everything else must match exactly.
"""

from __future__ import annotations

import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
PACKET = "spec/conformance"

# Regenerated from a fresh random signing key every run; see the comment in
# build-conformance-packet.sh. Excluded from the freshness comparison, not from
# the packet.
NONDETERMINISTIC = {
    "spec/conformance/artifacts/conformance-v2-signed.annpack",
    "spec/conformance/artifacts/conformance-v2-signed.pub",
    "spec/conformance/vectors/signature.json",
}


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout


def dirty_packet_paths() -> set[str]:
    """Paths under the packet that differ from HEAD, including untracked ones."""
    out = git("status", "--porcelain", "--untracked-files=all", "--", PACKET)
    paths = set()
    for line in out.splitlines():
        if not line.strip():
            continue
        # "XY path" — a rename would carry " -> ", which the packet never emits.
        paths.add(line[3:].strip())
    return paths


def main() -> int:
    before = dirty_packet_paths()
    if before:
        print("refusing to run: the conformance packet already has local changes.")
        for path in sorted(before):
            print(f"  {path}")
        print("Commit or discard them first; this check regenerates in place.")
        return 2

    subprocess.run(
        ["./scripts/build-conformance-packet.sh"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    try:
        changed = dirty_packet_paths() - NONDETERMINISTIC
        if changed:
            print("the committed conformance packet is stale.")
            print("`scripts/build-conformance-packet.sh` produces different bytes for:")
            for path in sorted(changed):
                print(f"  {path}")
            print()
            print("Regenerate and commit the packet, including the reports:")
            print("  ./scripts/build-conformance-packet.sh")
            print("  ./spec/conformance/run.py --adapter ./scripts/reference-adapter.sh \\")
            print("      --implementation rust/annpack-reference \\")
            print("      --output spec/conformance/reference-report.json")
            return 1
        print("conformance packet is current: regeneration reproduces every")
        print("committed artifact and vector, signature material excepted.")
        return 0
    finally:
        # Always restore, so a local run leaves no trace and a CI failure does
        # not report a second, derived diff on top of the real one.
        git("checkout", "--", PACKET)
        for path in dirty_packet_paths():
            (ROOT / path).unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
