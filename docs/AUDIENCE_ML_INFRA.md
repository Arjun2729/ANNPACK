# Audience: ML / Infra Engineers

## Day‑to‑day pain
- “We need reproducible, immutable ANN artifacts.”
- “Serving vector DBs at the edge is too heavy.”
- “We need predictable rollback and signed artifacts.”

## ANNPACK flow (mapped to pain)
1) Build a base pack (offline or real embeddings).
2) Ship deltas for updates; use packsets for newest‑wins.
3) Verify and sign artifacts; host on Range‑enabled storage.

## Copy‑paste flow
```bash
ANNPACK_OFFLINE=1 annpack build --input tiny_docs.csv --text-col text --output ./out/tiny --lists 4
annpack verify ./out
annpack sign ./out --key ./keys/demo_ed25519.key
```

## Deployment patterns
- **Immutable snapshots**: versioned packs in object storage.
- **Delta rollouts**: packset manifest with base + deltas.
- **Registry**: local PackHub for upload + Range serving.

## Troubleshooting
- **Mismatched base hash**: rebuild delta against the current base.
- **Determinism**: use `ANNPACK_OFFLINE=1` or `ANNPACK_DEVICE=cpu`.
- **CI failures**: run `bash tools/stage_all.sh` locally.
