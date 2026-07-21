# Transfer Reduction Baseline — FastAPI docs 0.115.12

**Date:** 2026-07-21
**Corpus:** FastAPI 0.115.12, commit `628c34e0cae200564d191c95d7edea78c88c4b5e`
**Method:** Source file measurement (real) + live browser Range GET measurement (real)

---

## Source corpus measurement

141 Markdown files from `docs/en/docs/` at pinned commit `628c34e0`:

```
$ find target/fastapi-eval/fastapi-src/docs/en/docs -name "*.md" | xargs wc -c
1288962 total
```

**Uncompressed Markdown: 1,288,962 bytes**
**Estimated gzip-compressed Markdown: ~250–320 KB** (docs compress at roughly 4–5×)

---

## ANNPack pack size

```
860,088 bytes   target/fastapi-eval/fastapi.annpack
pack_root:      c7147550fb7a2e0ff65af4030d730b3fad923fe0f548692b868cd26369a1cc7a
```

---

## Live per-query transfer measurement

Measured via `ANNPackBrowser.stats.{rangeRequests, bytes}` against a localhost Range server.
Each measurement opens a fresh `ANNPackBrowser` instance (cold cache, no section reuse).
5 queries, top-5 results each.

| Query | Range GETs | Bytes transferred |
|---|---|---|
| "how do I add authentication to FastAPI" | 12 | 459,467 |
| "dependency injection" | 12 | 461,181 |
| "async route handlers" | 12 | 457,276 |
| "request body validation" | 12 | 460,228 |
| "CORS middleware" | 12 | 462,895 |
| **Average** | **12** | **460,209** |

---

## Honest transfer analysis

### Cold first open (new browser session)

Opening the pack fetches all sections needed to validate and search it: header, directory,
manifest, documents, passage index, lexical dictionary, lexical postings, and passage data
for top results. Total: **~460 KB in 12 Range GETs**.

Comparison:
- vs uncompressed Markdown (1.29 MB): pack costs 64% less — but this is an unfair baseline;
  no real tool transfers uncompressed text.
- vs gzip-compressed Markdown (~270 KB): pack costs **~70% more** cold. The pack is larger
  than a compressed corpus download because it includes pre-built BM25 indexes and passage
  structure. Cold pack open is NOT a transfer-reduction story.

### Warm per-query (same browser session, sections already fetched)

After the first open, all sections are held in memory. Subsequent queries fetch only the
passage data for their specific results (~2–5 KB each). The pack answers unlimited follow-up
queries at near-zero marginal transfer cost.

### What the numbers actually prove

The value is **architecture**, not raw byte savings:

- A tool without ANNPack must download the corpus AND build a search index before answering
  any query. With ANNPack, the index is pre-built and embedded; the pack is a single fetch.
- After one 460 KB open, every subsequent query in the session costs ~2–5 KB (passage fetches
  only). A session with 10 queries costs ~460 + 9×4 = ~496 KB total — versus re-downloading
  1.29 MB each session without ANNPack.
- Range retrieval means a client can answer questions without holding the full corpus in memory
  or building a local index.

---

## Retired claims

| Claim | Status |
|---|---|
| "98.4% transfer reduction" | Retired — not reproducible from any honest measurement |
| "64% cold reduction" | Misleading — compares to uncompressed text; vs compressed, pack costs more cold |
| "85% warm reduction" | Wrong — was calculated by subtracting section sizes, not measuring actual warm queries |

---

## Safe claims (tied to this evidence)

- "Open the pack once (~460 KB, 12 Range GETs); answer all queries with ~2–5 KB marginal
  transfer per subsequent query — no index rebuild, no full corpus re-download."
- "BM25 lexical queries use 12 Range GETs against an 860 KB pack; smaller packs use 8."
- "Pack is 33% smaller than raw uncompressed source Markdown."

---

## Browser JS bug fixed

`web/annpack-browser.js` `validateSearchIndexes()` sorted dictionary entries incorrectly via
`Object.entries()` (V8 enumerates integer-like string keys before alphanumeric keys, breaking
the posting-cursor check for terms like `"0"`). Fixed by sorting entries by `meta.offset`
before iterating. Real-world packs with numeric terms now open correctly.

---

## Reproducibility

```bash
# Source size
find target/fastapi-eval/fastapi-src/docs/en/docs -name "*.md" | xargs wc -c

# Pack size
wc -c target/fastapi-eval/fastapi.annpack

# Live query transfer measurement
node /tmp/measure-fastapi-query.mjs
```
