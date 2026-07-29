#!/usr/bin/env python3
"""Synchronize public OKF fixture metadata from expected-roots.json."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_PATH = ROOT / "launch/google-okf/expected-roots.json"
README_PATH = ROOT / "launch/google-okf/README.md"
WEB_PATH = ROOT / "web/index.html"


def replace_once(text: str, pattern: str, replacement: str, label: str) -> str:
    updated, count = re.subn(pattern, replacement, text, count=1, flags=re.MULTILINE)
    if count != 1:
        raise SystemExit(f"expected exactly one {label}; found {count}")
    return updated


def render() -> dict[Path, str]:
    expected = json.loads(EXPECTED_PATH.read_text(encoding="utf-8"))
    source = expected["source"]
    roots = expected["artifacts"]
    revision = source["revision"]
    short_revision = revision[:7]
    ga4_root = roots["ga4"]

    readme = README_PATH.read_text(encoding="utf-8")
    readme = replace_once(
        readme,
        r"\| Revision \| `[^`]+` \(pinned by \[`reproduce\.sh`\]\(reproduce\.sh\)\) \|",
        f"| Revision | `{short_revision}` (pinned by [`reproduce.sh`](reproduce.sh)) |",
        "README revision row",
    )
    readme = replace_once(
        readme,
        r"\| Bundles \| `okf/bundles/\{[^`]+\}` \|",
        "| Bundles | `okf/bundles/{ga4, crypto_bitcoin, stackoverflow}` |",
        "README bundle row",
    )
    readme = replace_once(
        readme,
        r"\| Input \| OKF(?: v0\.2)? \|",
        "| Input | OKF v0.2 |",
        "README input row",
    )
    root_explanation = (
        "These roots compile the pinned OKF v0.2 source with "
        "`annpack-reference/0.4.0-rc4`. They identify this builder's exact "
        "artifact bytes; the reproduction script and CI fail on any unreviewed drift."
    )
    readme = replace_once(
        readme,
        r"(## Expected roots\n\n)[\s\S]*?(\n\n\| bundle \| artifact root \|)",
        rf"\1{root_explanation}\2",
        "README expected-roots explanation",
    )
    for name, root in roots.items():
        readme = replace_once(
            readme,
            rf"\| {re.escape(name)} \| `[0-9a-f]{{64}}` \|",
            f"| {name} | `{root}` |",
            f"README {name} root row",
        )
    readme = replace_once(
        readme,
        r"root=[0-9a-f]{64}",
        f"root={ga4_root}",
        "README live-demo root",
    )
    readme = replace_once(
        readme,
        r"Building Google's `acme_retail` v0\.2 exemplar at `[0-9a-f]+` produced 17 documents",
        f"Building Google's `acme_retail` v0.2 exemplar at `{short_revision}` produced 17 documents",
        "README acme revision",
    )

    web = WEB_PATH.read_text(encoding="utf-8")
    web = replace_once(
        web,
        r"root: '[0-9a-f]{64}',\n        q: 'What does the user_properties record contain\?',",
        f"root: '{ga4_root}',\n        q: 'What does the user_properties record contain?',",
        "browser GA4 preset root",
    )

    return {README_PATH: readme, WEB_PATH: web}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    rendered = render()
    drift = []
    for path, wanted in rendered.items():
        current = path.read_text(encoding="utf-8")
        if current != wanted:
            drift.append(path.relative_to(ROOT))
            if not args.check:
                path.write_text(wanted, encoding="utf-8")

    if args.check and drift:
        raise SystemExit(
            "OKF fixture metadata drift: " + ", ".join(str(path) for path in drift)
        )
    if drift:
        print("synchronized: " + ", ".join(str(path) for path in drift))
    else:
        print("OKF fixture metadata already synchronized")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
