#!/usr/bin/env python3
"""Assert the ANN-1 canonical execution chain reproduces its pinned digests.

A profile that names a model does not determine an embedding: native runtimes
select kernels from the host instruction set, so the same model yields different
vectors on arm64 and x64. The canonical machine pins the execution as well as
the model, and reproduces byte-identically across architectures and in a
browser.

This fails if any pinned input moves -- model, runtime binary, or the resulting
vectors -- rather than reporting a hash for a human to eyeball.
"""
from __future__ import annotations

import hashlib
import json
import re
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
PINS = json.loads((ROOT / "evals/canonical/canonical-pins.json").read_text())


def fail(message: str) -> None:
    print(f"::error::{message}", file=sys.stderr)
    raise SystemExit(1)


def digest(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


cache = ROOT / "evals/node_modules/@huggingface/transformers/.cache"
model = cache / PINS["model"]["path"]
if not model.is_file():
    fail(f"model artifact missing: {model}; run npm install --prefix evals and embed once")
if (seen := digest(model)) != PINS["model"]["sha256"]:
    fail(f"model artifact {seen[:16]} != pinned {PINS['model']['sha256'][:16]}")

runtime = ROOT / "evals/node_modules/onnxruntime-web/dist" / PINS["runtime"]["file"]
if not runtime.is_file():
    fail(f"runtime artifact missing: {runtime}")
if (seen := digest(runtime)) != PINS["runtime"]["sha256"]:
    fail(f"runtime artifact {seen[:16]} != pinned {PINS['runtime']['sha256'][:16]}")

out = ROOT / "target/okf-eval/canonical-passages.json"
# The reference implementation is the authority on its own digest. Recomputing
# it here would hash a different serialization -- JS and Python format floats
# differently -- and produce a second, silently divergent identity for the same
# vectors.
result = subprocess.run(
    ["node", "embed-canonical.mjs", str(out)],
    cwd=ROOT / "evals/canonical", check=True, capture_output=True, text=True,
)
match = re.search(r"vectors sha256 ([0-9a-f]{64})", result.stdout)
if not match:
    fail(f"reference implementation emitted no digest:\n{result.stdout}{result.stderr}")
seen = match.group(1)
if seen != PINS["vectors"]["passages_sha256"]:
    fail(
        f"canonical passage vectors {seen[:16]} != pinned "
        f"{PINS['vectors']['passages_sha256'][:16]}; the model and runtime "
        f"artifacts matched, so the computation changed"
    )

print(
    f"canonical ANN-1 execution verified: model {PINS['model']['sha256'][:12]}, "
    f"runtime {PINS['runtime']['sha256'][:12]}, "
    f"vectors {seen[:12]}, batch={PINS['execution']['batch']}, "
    f"threads={PINS['execution']['numThreads']}, max_tokens={PINS['execution']['max_tokens']}"
)
