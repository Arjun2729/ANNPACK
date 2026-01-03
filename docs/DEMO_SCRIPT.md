# Demo Script

## 30‑second pitch
- ANNPack packages ANN indexes as static files.
- Serve over HTTP Range, search in browser or Python.
- Immutable, cacheable, and reproducible.

## 5‑minute demo
1) Run `bash examples/quickstart_10min.sh`.
2) Open the URL and show status “Ready”.
3) Run a query and show results.

## 10‑minute demo
1) Download demo assets: `bash tools/download_demo_assets.sh ./demo_assets`.
2) Serve: `annpack serve ./demo_assets --port 8000`.
3) Load UI and run a few queries.

## What to screenshot
- UI “Ready” state with manifest loaded.
- Search results list with scores.
- `annpack verify` output for a pack.
- PackSet manifest showing base + delta chain.
