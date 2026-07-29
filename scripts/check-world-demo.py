#!/usr/bin/env python3
"""Validate the world-facing static demo as a published claim contract."""

from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEB_PATH = ROOT / "web/index.html"
DOCS_PATH = ROOT / "docs/index.html"
web = WEB_PATH.read_text(encoding="utf-8")
docs = DOCS_PATH.read_text(encoding="utf-8")

if web != docs:
    raise SystemExit("docs/index.html must be an exact generated copy of web/index.html")

required = {
    "candidate status": "candidate format · v0.4.0-rc4",
    "integrity boundary": "This verifies artifact integrity, signature validity, and retrieved-passage provenance.",
    "identity limitation": "It does not establish publisher identity without an external trusted key",
    "freshness limitation": "prove freshness",
    "faithfulness limitation": "prove that a model answer follows from the passage",
    "valid untrusted state": "signature valid ✓ · identity untrusted",
    "valid trusted state": "signature valid ✓ · identity trusted",
    "unsigned state": "if (info.status === 'unsigned') return 'unsigned';",
    "root mismatch state": "✗ root mismatch",
    "visible open failure": "setStatus(`Open failed: ${error.message}`, 'error');",
    "visible offline failure": "setStatus(`Offline install failed: ${error.message}`, 'error');",
    "artifact integrity wording": "artifact integrity verified",
}
for label, marker in required.items():
    if marker not in web:
        raise SystemExit(f"world demo missing {label}: {marker!r}")

if web.count("await pack.verifySignatures();") != 2:
    raise SystemExit("world demo must verify signatures once remotely and once offline")

for label, marker in {
    "silent signature downgrade": "catch (_signatureError)",
    "identity overclaim": "artifact identity verified",
    "publisher overclaim": "publisher identity verified",
    "endorsement overclaim": "Google endorses ANNPack",
}.items():
    if marker in web:
        raise SystemExit(f"world demo contains {label}: {marker!r}")

match = re.search(r'<script type="module">\s*(.*?)\s*</script>', web, re.DOTALL)
if not match:
    raise SystemExit("world demo has no inline module script")

with tempfile.NamedTemporaryFile("w", suffix=".mjs", delete=False) as handle:
    handle.write(match.group(1))
    script = Path(handle.name)
try:
    subprocess.run(["node", "--check", str(script)], check=True)
finally:
    script.unlink(missing_ok=True)

print("world-facing demo claims, failure states, generated copy, and module syntax verified")
