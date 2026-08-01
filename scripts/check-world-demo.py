#!/usr/bin/env python3
"""Validate the world-facing static demo as a published claim contract."""

from __future__ import annotations

import json
import re
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEB_PATH = ROOT / "web/index.html"
DOCS_PATH = ROOT / "docs/index.html"
PACK_PATH = ROOT / "docs/packs/google-okf-ga4.annpack"
PUBLIC_KEY_PATH = ROOT / "docs/packs/google-okf-ga4.pub"
EXPECTED_ROOTS_PATH = ROOT / "launch/google-okf/expected-roots.json"
ANNPACK = ROOT / "target/release/annpack"
EXPECTED_PUBLIC_KEY = "03a107bff3ce10be1d70dd18e74bc09967e4d6309ba50d5f1ddc8664125531b8"

expected = json.loads(EXPECTED_ROOTS_PATH.read_text(encoding="utf-8"))
EXPECTED_ROOT = expected["artifacts"]["ga4"]
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
    "published root": EXPECTED_ROOT,
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

actual_public_key = PUBLIC_KEY_PATH.read_text(encoding="utf-8").strip()
if actual_public_key != EXPECTED_PUBLIC_KEY:
    raise SystemExit(
        f"published demo key {actual_public_key} != pinned {EXPECTED_PUBLIC_KEY}"
    )

if not ANNPACK.is_file():
    raise SystemExit("release compiler missing before world-demo contract check")
report = json.loads(
    subprocess.check_output(
        [str(ANNPACK), "inspect", str(PACK_PATH), "--json"],
        cwd=ROOT,
        text=True,
    )
)
actual_root = report["root_hash"]
if actual_root != EXPECTED_ROOT:
    raise SystemExit(f"published demo root {actual_root} != expected-roots {EXPECTED_ROOT}")

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

# The README quotes the pinned upstream revision and every expected root so a
# reader can check them without opening another file. Quoted values rot; this
# fails the build instead of publishing a stale one.
readme = (ROOT / "README.md").read_text(encoding="utf-8")
if expected["source"]["revision"] not in readme:
    raise SystemExit(
        f"README does not quote the pinned upstream revision "
        f"{expected['source']['revision']}"
    )
for name, root in expected["artifacts"].items():
    if root not in readme:
        raise SystemExit(f"README does not quote the expected {name} root {root}")

print(
    "world-facing demo claims, failure states, generated copy, module syntax, "
    "expected artifact root, public test key, and README reproduction values "
    "verified"
)
