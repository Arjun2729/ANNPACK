# Positioning

ANNPack is a portable, static ANN index format and tooling for serving vector search over HTTP Range.

## Use it when
- You want immutable snapshot distribution via CDN/S3/edge.
- You need browser/WASM or offline search without a database.
- You can tolerate batch updates (pack rebuilds or deltas).

## Don’t use it when
- You need low‑latency writes or transactional updates.
- You need live filtering over frequently changing metadata.
- You need a managed vector DB with always‑fresh indexes.

## Replace vs complement
- **Complement** a DB: export snapshots for edge or offline use.
- **Replace** a DB: only if your data is static or updated in batches.

## Decision table
| Requirement | Fit | Notes |
|---|---|---|
| Static or batched updates | ✅ | Use packs or packsets |
| Real‑time updates | ❌ | Use a vector DB |
| Browser/WASM search | ✅ | Range + WASM |
| Heavy metadata filtering | ⚠️ | Pre‑filter offline |
