#!/usr/bin/env python3
"""Fail if a fresh conformance run disagrees with the committed reference report.

The packet is the contract handed to independent implementers. Running it against
ourselves on every build keeps it from rotting, and comparing against the
committed report makes any change to our own conformance an explicit, reviewed
edit rather than a silent drift.
"""
import json
import sys

fresh_path, committed_path = sys.argv[1], sys.argv[2]
fresh = json.load(open(fresh_path, encoding="utf-8"))
committed = json.load(open(committed_path, encoding="utf-8"))

for key in ("total", "passed", "failed", "conformant"):
    if fresh[key] != committed[key]:
        sys.exit(
            f"::error::conformance report drifted: {key} "
            f"{committed[key]} -> {fresh[key]}. Regenerate with "
            f"scripts/build-conformance-packet.sh and review the change."
        )

failed = [r["check"] for r in fresh["results"] if not r["pass"]]
if failed:
    sys.exit("::error::conformance failures: " + "; ".join(failed))

print(f"conformance stable: {fresh['passed']}/{fresh['total']}")
