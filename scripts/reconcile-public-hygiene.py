#!/usr/bin/env python3
"""One-shot branch reconciliation. Deletes itself before the final commit."""

import json
from pathlib import Path


def replace_exact(path: str, old: str, new: str) -> None:
    target = Path(path)
    text = target.read_text()
    if text.count(old) != 1:
        raise SystemExit(f"{path}: expected exactly one match for {old!r}")
    target.write_text(text.replace(old, new, 1))


replace_exact(
    "evals/ann9_crossmodel_killswitch.mjs",
    "import path from 'node:path';\n",
    "import path from 'node:path';\nimport { fileURLToPath } from 'node:url';\n",
)
replace_exact(
    "evals/ann9_crossmodel_killswitch.mjs",
    "// --- config -----------------------------------------------------------------",
    "// --- paths ------------------------------------------------------------------\n"
    "const REPO_ROOT = process.env.ANNPACK_REPO_ROOT\n"
    "  || path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');\n"
    "const FASTAPI_EVAL_ROOT = process.env.FASTAPI_EVAL_ROOT\n"
    "  || path.join(REPO_ROOT, 'target/fastapi-eval');\n\n"
    "// --- config -----------------------------------------------------------------",
)
replace_exact(
    "evals/ann9_crossmodel_killswitch.mjs",
    "  const roots = [\n"
    "    '/Users/anika/annpackv2/target/fastapi-eval/fastapi-src/docs/en/docs',\n"
    "    '/Users/anika/annpackv2/target/fastapi-eval/fastapi-src',\n"
    "    '/Users/anika/annpackv2/spec',\n"
    "    '/Users/anika/annpackv2',\n"
    "  ];",
    "  const roots = [\n"
    "    path.join(FASTAPI_EVAL_ROOT, 'fastapi-src/docs/en/docs'),\n"
    "    path.join(FASTAPI_EVAL_ROOT, 'fastapi-src'),\n"
    "    path.join(REPO_ROOT, 'spec'),\n"
    "    REPO_ROOT,\n"
    "  ];",
)

replace_exact(
    "evals/ann9_relevance_adapter.mjs",
    "import path from 'node:path';\n",
    "import path from 'node:path';\nimport { fileURLToPath } from 'node:url';\n",
)
old_paths = (
    "const DOCS_ROOT = '/Users/anika/annpackv2/target/fastapi-eval/fastapi-src/docs/en/docs';\n"
    "const QREL_FILES = [\n"
    "  '/Users/anika/annpackv2/target/fastapi-eval/qrels-labeled.jsonl',\n"
    "  '/Users/anika/annpackv2/launch/evidence/2026-07-20/workstream3-evals/fastapi-candidate-qrels.jsonl',\n"
    "];\n"
    "// anchors: repo prose DISJOINT from the English eval docs corpus (public-anchor-\n"
    "// set analogue). Broad roots for volume; the whole en docs tree (DOCS_ROOT) is\n"
    "// skipped inside proseSentences so no eval passage can leak in as an anchor.\n"
    "const ANCHOR_ROOTS = [\n"
    "  '/Users/anika/annpackv2/spec',\n"
    "  '/Users/anika/annpackv2/launch',\n"
    "  '/Users/anika/annpackv2/rust/src',\n"
    "  '/Users/anika/annpackv2/bindings',\n"
    "  '/Users/anika/annpackv2/README.md',\n"
    "  '/Users/anika/annpackv2/target/fastapi-eval/fastapi-src/fastapi',\n"
    "  '/Users/anika/annpackv2/target/fastapi-eval/fastapi-src/tests',\n"
    "];"
)
new_paths = (
    "const REPO_ROOT = process.env.ANNPACK_REPO_ROOT\n"
    "  || path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');\n"
    "const FASTAPI_EVAL_ROOT = process.env.FASTAPI_EVAL_ROOT\n"
    "  || path.join(REPO_ROOT, 'target/fastapi-eval');\n"
    "const DOCS_ROOT = process.env.FASTAPI_DOCS_ROOT\n"
    "  || path.join(FASTAPI_EVAL_ROOT, 'fastapi-src/docs/en/docs');\n"
    "const QREL_FILES = process.env.FASTAPI_QREL_FILES\n"
    "  ? process.env.FASTAPI_QREL_FILES.split(path.delimiter)\n"
    "  : [\n"
    "      path.join(FASTAPI_EVAL_ROOT, 'qrels-labeled.jsonl'),\n"
    "      path.join(REPO_ROOT, 'launch/evidence/2026-07-20/workstream3-evals/fastapi-candidate-qrels.jsonl'),\n"
    "    ];\n"
    "// anchors: repo prose DISJOINT from the English eval docs corpus (public-anchor-\n"
    "// set analogue). Broad roots for volume; the whole en docs tree (DOCS_ROOT) is\n"
    "// skipped inside proseSentences so no eval passage can leak in as an anchor.\n"
    "const ANCHOR_ROOTS = process.env.ANNPACK_ANCHOR_ROOTS\n"
    "  ? process.env.ANNPACK_ANCHOR_ROOTS.split(path.delimiter)\n"
    "  : [\n"
    "      path.join(REPO_ROOT, 'spec'),\n"
    "      path.join(REPO_ROOT, 'launch'),\n"
    "      path.join(REPO_ROOT, 'rust/src'),\n"
    "      path.join(REPO_ROOT, 'bindings'),\n"
    "      path.join(REPO_ROOT, 'README.md'),\n"
    "      path.join(FASTAPI_EVAL_ROOT, 'fastapi-src/fastapi'),\n"
    "      path.join(FASTAPI_EVAL_ROOT, 'fastapi-src/tests'),\n"
    "    ];"
)
replace_exact("evals/ann9_relevance_adapter.mjs", old_paths, new_paths)

latest = Path("benches/latest.json")
report = json.loads(latest.read_text())
report["recorded_at"] = "2026-07-29"
report["status"] = "historical local run; verify and query latency gates failed"
report["environment"] = "macOS developer machine; process-inclusive CLI timings"
report["binary"] = "target/release/annpack"
history = Path("benches/history/2026-07-29-macos-process-inclusive.json")
history.parent.mkdir(parents=True, exist_ok=True)
history.write_text(json.dumps(report, indent=2) + "\n")
latest.unlink()
Path("benches/README.md").write_text(
    "# Benchmarks\n\n"
    "The release source of truth is the benchmark gate executed by GitHub Actions on each pull request. "
    "There is intentionally no tracked `latest.json`: wall-clock CLI timings are machine- and scheduler-specific, "
    "and a copied local run quickly becomes misleading.\n\n"
    "Run `python3 benches/benchmark.py --enforce` for the release gate, or add "
    "`--output <path>` to preserve a dated report. `benches/history/` contains explicitly labelled historical runs, "
    "including failures; they are evidence, not current guarantees.\n"
)

readiness = Path("launch/RELEASE-READINESS.md")
text = readiness.read_text()
text = text.replace(
    "[`spec/LAUNCH-GATES.md`](../spec/LAUNCH-GATES.md)",
    "[`LAUNCH-GATES.md`](LAUNCH-GATES.md)",
)
text = text.replace(
    "✅ **closed** — live and independently verified",
    "✅ **closed** — live origin verified",
)
start = text.index("## What must happen next, in order")
end = text.index("Feature freeze holds throughout:", start)
replacement = (
    "## Outstanding external gates\n\n"
    "Repository maintenance cannot close the remaining release-readiness blockers:\n\n"
    "- an independent security review by an unaffiliated reviewer;\n"
    "- a genuinely independent second Core reader;\n"
    "- independently produced relevance labels and a hard-negative evaluation;\n"
    "- a credentialed GHCR publication under the rc4 release state; and\n"
    "- a production CDN origin with the required media type, CORS, caching, ETag, and Range behavior.\n\n"
    "Technical OKF reproduction facts and open interoperability questions live in "
    "[`google-okf/README.md`](google-okf/README.md). Contact strategy and commercial sequencing are maintained outside this repository.\n\n"
)
readiness.write_text(text[:start] + replacement + text[end:])

Path(__file__).unlink()
