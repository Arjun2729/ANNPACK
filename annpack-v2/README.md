# ANNPack v2 Demo (WASM + Browser)

Minimal, reproducible setup to serve ANNPack indexes via HTTP Range and search them in-browser with a WASM runtime.

## Reproduce quickly
```bash
cd /Users/anika/Downloads/wikidemo/wikidemo/annpack-v2
./build.sh
PORT=8080 python3 server.py
# open http://127.0.0.1:8080/?v=5 (same host/port for all assets, disable cache)
```

## Prerequisites
- macOS Apple Silicon, Homebrew `emscripten` (`emcc` on PATH).
- Python 3.11/3.12 recommended (builder deps are unreliable on 3.14).

## Build the WASM runtime
```bash
cd /Users/anika/Downloads/wikidemo/wikidemo/annpack-v2
./build.sh
```

## Serve the demo
```bash
cd /Users/anika/Downloads/wikidemo/wikidemo/annpack-v2
python3 server.py       # uses HOST (default 127.0.0.1) and PORT (default 8080)
# open http://127.0.0.1:8080/?v=5 with cache disabled; same host/port for all assets
# index.html tries wikipedia_en_50k.* first, then falls back to generic_index.* if present, or manifest if available
```

## Builder quickstart (tiny shard)
```bash
cd /Users/anika/Downloads/wikidemo/wikidemo/annpack-v2
python3.12 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install polars faiss-cpu sentence-transformers torch datasets

# build a tiny shard from the repo's sample CSV
python tools/builder.py \
  --input ../tiny_docs.csv \
  --text-col text \
  --id-col id \
  --output generic_index \
  --lists 64 \
  --batch-size 32

# copy artifacts into the served folder
cp generic_index.annpack generic_index.meta.jsonl .
```
Notes:
- Defaults are tuned for small shards (2k–50k rows). 20k–100k vectors per shard keeps memory reasonable on laptops.
- Metadata defaults to snippets only; full text is excluded unless you pass `--include-text`.
- Recommended sharding: add `--shard-size 50000 --output-dir .` to produce `PREFIX.manifest.json` plus `PREFIX.shardNNN.{annpack,meta.jsonl}`. The manifest uses relative paths so it can be served directly from this folder. If `--shard-size 0`, you still get a single-shard manifest.
- IDs: if you provide `--id-col`, uniqueness is `(shard, id)`; without `--id-col` IDs are generated globally across shards.
 - Keep a clean Python 3.11/3.12 venv for real builds; Python 3.14 often breaks torch/transformers on macOS.

### Manifest
`PREFIX.manifest.json` schema: `version`, `created_at`, `model`, `dim`, `metric`, `n_lists`, `total_vectors`, `shard_size`, `shards[]`. Each shard has `name`, `annpack`, `meta`, `n_vectors`, and optional `id_min/id_max`. Paths are relative so a static server from the manifest directory works.

## Smoke test (definition of done)
```bash
cd /Users/anika/Downloads/wikidemo/wikidemo/annpack-v2
./smoke_test.sh   # rebuilds wasm, runs deterministic native tests, starts server (PORT default 8765), verifies Range + ANNP magic
```

## Correctness gate (FAISS vs ANNPACK)
- Install eval deps: `pip install -r requirements-eval.txt`
- Build FiQA shards + embedding dump:
```bash
python tools/build_beir.py --dataset fiqa --outdir data/fiqa --model sentence-transformers/all-MiniLM-L6-v2 --shard-size 50000 --lists 256 --batch-size 64
```
- Run fidelity check (native engine, FAISS ground truth, exits non-zero on failure):
```bash
./fidelity_gate.sh fiqa
# or: python tools/fidelity_gate.py --manifest data/fiqa/fiqa.manifest.json --sample 200 --k 10 --seed 42
```
Outputs go to `eval/fiqa_fidelity.json` and `.md` with overlap/jaccard/tau metrics and PASS/FAIL summary.

## What Fidelity Gate Proves
- With exact settings (probe=n_lists, max_scan=all) and GT built from the same vectors, ANNPack matches FAISS results.
- This validates the on-disk format, fp16 decode, ID mapping, and score ordering.

## What Fidelity Gate Does Not Prove
- It does not measure retrieval quality vs relevance labels (BEIR nDCG/Recall).
- It does not guarantee latency/cost; higher probe means more work.
- It does not guarantee tail behavior at low probe (routing misses still happen).

## Recommended nprobe presets (IVF)
Use fraction-based presets and clamp to [1, n_lists]:
- fast: round(n_lists * 0.03125)
- balanced: round(n_lists * 0.125)
- safe: round(n_lists / 5.3)
Tail behavior means the worst-case queries (min_overlap) lag the average; passing a min_overlap gate often requires a higher probe than the average suggests.

FiQA anchors (measured):
- lists=256: probe=128 gives high average overlap; probe=256 is exact (PASS).
- lists=1024 (one-shard): first PASS at probe=192 for min_overlap@10 >= 0.7.

Note: a fidelity mismatch is not the same as irrelevant results; it only means ANNPack and brute-force top-k disagree.

## Final testing (demo wiring)
1) Start server:
```bash
python -m http.server 8000
```
2) Run demo smoke:
```bash
python tools/demo_smoke.py --base http://localhost:8000
```
Expected: all OK lines; no 404s; shard URLs from the default manifest OK.

3) Manual UI sanity:
Open the page, confirm it reaches Ready, preset values reflect n_lists, and a bad manifest URL shows a red error banner with URL/status.

(Optional) Fidelity correctness (not part of smoke):
```bash
python tools/fidelity_gate.py --manifest data/fiqa/fiqa.manifest.json --queries fiqa --queries-path data/fiqa/beir_cache/fiqa --sample 200 --k 10 --seed 42 --mode exact --gt annpack
```

## Release Gate Checklist
1) Build FiQA (256 lists) and run exact fidelity:
```bash
python tools/build_beir.py --dataset fiqa --outdir data/fiqa --lists 256 --shard-size 50000 --batch-size 64
python tools/fidelity_gate.py --manifest data/fiqa/fiqa.manifest.json --queries fiqa --queries-path data/fiqa/beir_cache/fiqa --sample 200 --k 10 --seed 42 --mode exact --gt annpack
```
Expected key lines: `avg overlap@10: 1.000`, `PASS: True`.

2) Build FiQA (1024 lists) and sweep probes:
```bash
python tools/build_beir.py --dataset fiqa --outdir data/fiqa_1024 --lists 1024 --shard-size 50000 --batch-size 64
for p in 8 16 32 64 128 256; do
  python tools/fidelity_gate.py --manifest data/fiqa_1024/fiqa.manifest.json --queries fiqa --queries-path data/fiqa_1024/beir_cache/fiqa --sample 200 --k 10 --seed 42 --probe $p --max-scan 999999 --gt annpack
done
```
Expected: overlap improves with probe; `probe=192+` should pass the min-overlap gate (one-shard FiQA‑1024).
Observed (one-shard, n_lists=1024, sample=200, seed=42): min-overlap@10 first passes at `probe=192` (224/256 also pass).

If model downloads fail, ensure the model is cached and set `HF_HUB_OFFLINE=1` (or use `--offline-dummy` for a deterministic build).

3) Demo smoke:
```bash
python3 server.py
# open http://127.0.0.1:8080/?v=5, confirm shards load and a query returns results
```

Note: `#` is a shell comment. Don’t paste it as a standalone command.
