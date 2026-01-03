# ANNPack

[![PyPI version](https://img.shields.io/pypi/v/annpack.svg)](https://pypi.org/project/annpack/)
[![CI](https://github.com/Arjun2729/ANNPACK/actions/workflows/ci.yml/badge.svg)](https://github.com/Arjun2729/ANNPACK/actions/workflows/ci.yml)

Serverless vector search: static `.annpack` files served over HTTP Range, searched in-browser via WASM + Transformers.js.

## Positioning (TL;DR)
ANNPack is a portable, static ANN index format + tooling for serving vector search over HTTP Range.
Use it when you want low‑ops distribution (CDN, S3, edge) and browser/WASM search.
Don’t use it when you need mutable, transactional, or real‑time indexed updates; use a vector DB instead.
It complements vector DBs by packaging snapshots into immutable, cacheable artifacts.

## Repository layout
- Primary Python package lives in `python/annpack/` and is accessed via the `annpack` CLI (`annpack build`, `annpack serve`, `annpack smoke`).
- `annpack-v2/` contains the experimental WASM demo and tooling; treat it as legacy/experimental and consider moving it to a separate repo later.
- `docs/` contains architecture, API/CLI usage, and WASM notes.
- `web/` contains the JS client (`web/packages/client`) and UI app (`web/apps/ui`).

## Quickstart (CLI)

```bash
pip install annpack
ANNPACK_OFFLINE=1 annpack build --input tiny_docs.csv --text-col text --output ./out/tiny --lists 256
annpack serve ./out/tiny --port 8000
annpack smoke ./out/tiny --port 8000
```

Optional installs:
- `pip install annpack[embed]` for real embeddings (torch + sentence-transformers)
- `pip install annpack[registry]` for the local PackHub service

What goes into `./out/tiny`:
- `tiny.annpack` (binary index)
- `tiny.meta.jsonl` (metadata rows)
- `tiny.manifest.json` (shard list; used by the UI)

Python API quickstart:
```python
from annpack.api import build_pack, open_pack

build_pack("tiny_docs.csv", "./out/tiny", text_col="text", id_col="id", lists=4, seed=0, offline=True)
pack = open_pack("./out/tiny")
print(pack.search("hello", top_k=5))
```

Examples:
- `examples/hello_world_build_and_search.py`
- `examples/hello_world_cli.sh`
- `examples/frontend_static_demo/` (build + host pack on a static server)

## Who is this for?
**Frontend builders** shipping static sites/apps who need semantic search without a backend.
- Build a pack locally → host on CDN/S3 → load in the browser with Range.
- Use the UI to inspect metadata + debug manifests.
- Ship search inside a static site or a WASM-enabled app.

**ML/infra engineers** who need reproducible, distributable ANN artifacts.
- Build packs deterministically (offline mode) → sign/verify → host cheaply.
- Ship delta updates with PackSets (append-only + tombstones).
- Serve packs via PackHub or any Range-capable server.

## 10-minute demo
Quick script (offline, deterministic):
```bash
bash examples/quickstart_10min.sh
```
Expected output includes:
- `PASS smoke`
- `READY: open http://127.0.0.1:<port>/`

Golden demo launcher (build + serve + UI command):
```bash
bash tools/run_demo.sh
```

Optional medium demo assets (downloaded, not in git):
```bash
bash tools/download_demo_assets.sh ./demo_assets
```

## Recorded demo checklist
1) Run `bash examples/quickstart_10min.sh`
2) Confirm `PASS smoke`
3) Open the printed URL and verify the UI shows “Ready”.

Troubleshooting:
- Port in use: pass `--port <free-port>` to `serve`/`smoke`.
- Missing manifest: ensure the build output dir contains `*.manifest.json`.
- CORS: `annpack serve` enables permissive CORS headers by default.
- Small datasets: `annpack build` automatically clamps `--lists` to the number of vectors to avoid FAISS small-data warnings.
- Offline/air-gapped builds: set `ANNPACK_OFFLINE=1` (dummy embeddings, no embed deps required).
- Real embeddings: install `annpack[embed]` and unset `ANNPACK_OFFLINE`.
- macOS "permission denied" for console scripts: remove quarantine and retry:
  - `xattr -dr com.apple.quarantine "$(python -c 'import site; print(site.getsitepackages()[0])')"`
- Avoid venvs named `#` (shell treats `#` as a comment).
- Determinism: manifests/meta are deterministic and clustering is seeded. Embeddings can vary across devices/backends; for strict reproducibility, set `ANNPACK_DEVICE=cpu` during builds.

## Offline mode
Set `ANNPACK_OFFLINE=1` to use deterministic dummy embeddings (no model downloads). This keeps CI and smoke tests fast and network-free. For real embeddings, install `annpack[embed]` and unset `ANNPACK_OFFLINE`.

## Benchmarks
See `docs/BENCHMARKS.md` and `docs/benchmarks/bench_report.md` for reproducible baseline numbers.

## Full Wikipedia 1M Demo

Build a ~1M document Wikipedia index with MiniLM (use the parquet-backed dataset):
```
annpack build \
  --hf-dataset wikimedia/wikipedia \
  --hf-config 20231101.en \
  --hf-split train \
  --output ./wikipedia_1M \
  --model all-MiniLM-L6-v2 \
  --lists 4096 \
  --max-rows 1000000 \
  --batch-size 512
```
Outputs: `wikipedia_1M.annpack`, `wikipedia_1M.meta.jsonl`, `wikipedia_1M.manifest.json`.

Resource notes:
- ~1M x 384-d float32 embeddings: ~1.5 GB RAM for vectors plus metadata; plan for several GB headroom.
- Build time varies by machine; tens of minutes on laptops is expected. To sanity-check, try a smaller build first, e.g. `--max-rows 100000 --lists 1024 --batch-size 256`.

## Final testing (serve wiring)
- Start server: `annpack serve ./out/tiny --port 8000`
- Run smoke: `annpack smoke ./out/tiny --port 8000` (expected: PASS smoke)
- Manual UI sanity: open the page, confirm it reaches Ready, presets reflect `n_lists`, and a bad manifest URL shows an error banner.
- Smoke test verifies wiring, not retrieval relevance (fidelity is covered by `fidelity_gate.py`).
- Sandboxed envs that forbid localhost bind can skip network smoke with `ANNPACK_SKIP_NET_TESTS=1` (CI does not set this).

## Stage 1 acceptance (automated)
Run the end-to-end acceptance script:
```bash
bash tools/stage1_acceptance.sh
```
It creates an isolated venv, installs the package, builds a tiny pack from `tiny_docs.csv`, and runs smoke. Expected last line: `PASS stage1 acceptance`.
- The script installs `setuptools`/`wheel` first to ensure the build backend is present.

## Stage 2 acceptance (API + determinism)
```bash
bash tools/stage2_acceptance.sh
```
This validates the public Python API, offline determinism, and CLI basics. Expected last line: `PASS stage2 acceptance`.

## Stage 3: Delta packs (PackSet)
Create a packset (base + deltas) and query with newest-wins + tombstones:
```bash
# Build base packset
python - <<'PY'
from annpack.packset import build_packset_base
build_packset_base("tiny_docs.csv", "./packset", text_col="text", id_col="id", lists=4, seed=123, offline=True)
PY

# Create delta (adds/updates + deletes)
python - <<'PY'
from annpack.packset import build_delta, update_packset_manifest
build_delta(
    base_dir="./packset/base",
    add_csv="delta_add.csv",
    delete_ids=[1],
    out_delta_dir="./packset/deltas/0001.delta",
    text_col="text",
    id_col="id",
    lists=4,
    seed=123,
    offline=True,
)
update_packset_manifest("./packset", "./packset/deltas/0001.delta", seq=1)
PY

# Query packset
python - <<'PY'
from annpack.api import open_pack
pack = open_pack("./packset")
print(pack.search("delta add", top_k=3))
pack.close()
PY
```

## Stage 4 acceptance (distribution + portability)
```bash
bash tools/stage4_acceptance.sh
```
This builds wheel + sdist, runs `twine check`, installs into fresh venvs, and validates CLI + offline build + search. Expected last line: `PASS stage4 acceptance`.

## Pre-talk gates
```bash
bash tools/repo_hygiene.sh
bash tools/clean_checkout_gate.sh
```
These ensure a clean release bundle and a fresh-clone pass.

## Docs
- `docs/ARCHITECTURE.md`
- `docs/API_USAGE.md`
- `docs/CLI_USAGE.md`
- `docs/WASM.md`
- `docs/POSITIONING.md`
- `docs/AUDIENCE_FRONTEND.md`
- `docs/AUDIENCE_ML_INFRA.md`
- `docs/VERIFY.md`
- `docs/BENCHMARKS.md`
- `docs/benchmarks/README.md`
- `CODE_OF_CONDUCT.md`
- `SUPPORT.md`
- `SECURITY.md`
- `RELEASE.md`
- `docs/ONE_PAGER.md`
- `docs/DISCOVERY_QUESTIONS.md`
- `docs/DEMO_SCRIPT.md`

## Web client + UI
- Client SDK: `web/packages/client` (published as `@annpack/client`)
- UI app: `web/apps/ui` (Vite + React)
- WASM build notes: `docs/WASM.md`

## Registry (local)
See `registry/README.md` for a local FastAPI-based pack registry with Range support.

## Architecture
- **Builder (Python)**: `annpack build` CLI → `.annpack` + `.meta.jsonl`.
- **Runtime (C/WASM)**: `ann_load_index`, `ann_result_size_bytes`, `ann_search` using HTTP Range reads.
- **Frontend (JS)**: Transformers.js MiniLM embeddings → WASM ANN search → render metadata (title/text/url).

## Legacy
- `build_fast.py` is LEGACY (Cohere 768D Wikipedia embeddings). Use `annpack build` (alias: `annpack-build`) with MiniLM instead.
- `ann_query_bytes` in `main.c` is retained for debugging but the UI uses `ann_search` exclusively.

## Project hygiene
- License: `LICENSE`
- Contributing: `CONTRIBUTING.md`
- Security: `SECURITY.md`

## File Format (unchanged)
- 72-byte header: magic, version, endian, header_size, dim, metric, n_lists, n_vectors, offset_table_ptr.
- Centroids (float32), then IVF lists: [count:u32][ids:int64*count][vecs:float16*dim*count], then offset table [offset:u64,length:u64] per list.

## ANNPack File Format & Ecosystem
ANNPack is a portable binary format for IVF-based ANN search. The current spec is documented in `docs/FORMAT.md`.

Pure Python reader/searcher example:
```python
import numpy as np
from annpack.reader import ANNPackIndex

with ANNPackIndex.open("my_index.annpack") as idx:
    dim = idx.header.dim
    q = np.random.randn(dim).astype(np.float32)
    q /= np.linalg.norm(q)
    hits = idx.search(q, k=10)
    print(hits)
```
Other languages can implement a reader using the format described in `docs/FORMAT.md`; the C/WASM runtime is one consumer.
