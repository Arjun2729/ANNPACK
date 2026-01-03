# ANNPack One‑Pager

## Problem
Vector search is expensive to host and hard to ship to browsers/edge.

## Solution
ANNPack packages ANN indexes as static artifacts that can be served over HTTP Range and searched locally (WASM/Python).

## Why now
- Edge/CDN adoption is high.
- Browser ML is mainstream.
- Teams need reproducible, immutable artifacts.

## Demo
```bash
ANNPACK_OFFLINE=1 annpack build --input tiny_docs.csv --text-col text --output ./out/tiny --lists 4
annpack serve ./out/tiny --port 8000
```

## Next step
Share a pack URL and we’ll plug it into the UI.
