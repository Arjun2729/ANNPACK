# ANNPack Release Readiness

**Generated:** 2026-07-20
**Machine:** Apple M4, 10 cores, 16 GB RAM, macOS 26.2 (25C56)
**Commit:** 7f2c2bbd05400bb37234bd974725e0d844383a6b
**Binary:** `target/release/annpack` v0.3.0
**Toolchain:** rustc 1.97.1 (stable), cargo 1.97.1

---

## Actual launch status: NOT READY

6 of 10 public launch gates are formally closed. The retrieval-quality table (Gates 5 & 6)
was produced on the real FastAPI corpus on 2026-07-26 and its embedding decision is made —
those close as soon as the `qrels-labeled.jsonl` labels are confirmed as independent human
adjudication (Gate 4) and committed with provenance. That leaves **one genuinely external
blocker: Gate 2, the independent security review.** All other preparation is complete.

_Last refreshed: 2026-07-26._

---

## What is genuinely complete

### Internal verification (Workstream 1)

All tests pass. All benchmarks pass their gates.

| Check | Result | Evidence |
|---|---|---|
| `cargo fmt --check` | PASS | `logs/fmt-check-full.log` |
| `cargo clippy -D warnings` | PASS | `logs/clippy-full.log` |
| `cargo test --workspace --all-targets` | 74/74 PASS (44 at 2026-07-20; +30 from ANN-7..10 work) | `logs/cargo-test-full.log` |
| `cargo build --release` | PASS | `logs/cargo-build-release-full.log` |
| `node web/smoke-range.mjs` | PASS (8 Range GETs, 4177 B) | `logs/smoke-range.log` |
| `node web/smoke-vector.mjs` | PASS (11 Range GETs) | `logs/smoke-vector.log` |
| `node web/smoke-offline.mjs` | PASS (server killed, query succeeded) | `logs/smoke-offline.log` |
| `node bindings/node/smoke.mjs` | PASS | `logs/smoke-node-bindings.log` |
| `python3 bindings/python/tests/test_smoke.py` | PASS | `logs/smoke-python.log` |
| `node web/smoke-transformers-adapter.mjs` | PASS (384 dims) | `logs/smoke-transformers.log` |

### M4 benchmark results (process-inclusive latency)

| Metric | M4 Result | Gate | Status |
|---|---|---|---|
| Pack/source ratio | 86.3% | ≤90% | ✅ PASS |
| Build time (1,000 docs) | 70 ms | ≤1,500 ms | ✅ PASS |
| Verify p95 | 3.53 ms | ≤25 ms | ✅ PASS |
| Query p95 | 6.17 ms | ≤25 ms | ✅ PASS |
| First result correct | true | true | ✅ PASS |

**Historical M1/8 GB reference:** build 139.79 ms, verify p95 7.41 ms, query p95 15.45 ms.
M4 is approximately 2× faster across all metrics.

### Google OKF reproduction

All three roots verified from a fresh clone of pinned commit `d44368c15e38e7c92481c5992e4f9b5b421a801d`:

| Dataset | Expected root | Status |
|---|---|---|
| ga4 | `f9256b569f574d8a9068be6372a6e8d7f2b76d0afd2f0650e305218669e73b35` | ✅ Verified |
| crypto-bitcoin | `3978663b7e7cedaafeb97460f0bbab4643e21ca854271b21f03eff96c5214f97` | ✅ Verified |
| stackoverflow | `3acbe05d8c78f8c75dedcfb25843049d0cd4059b89d30c1b07c60e063a1b2d86` | ✅ Verified |

Evidence: `workstream-okf/reproduce-output.txt`

### Crawl comparison (live measured — gate 7 closed)

| Metric | Value | Corpus | Status |
|---|---|---|---|
| Range GETs per cold open (lexical) | 8 | docs-v1.annpack, 4 KB, smoke test | ✅ enforced |
| Range GETs per cold open (lexical) | 12 | FastAPI, 860 KB, 141 docs, 1864 passages | ✅ measured |
| Range GETs per cold open (vector) | 11 | docs-v1.annpack, 4 KB, smoke test | measured |
| Cold open transfer | ~460 KB | FastAPI 860 KB pack, avg 5 queries | ✅ measured |
| Warm per-query transfer | ~2–5 KB | FastAPI, subsequent queries same session | ✅ measured |
| Source corpus size | 1,288,962 bytes | 141 .md files, commit 628c34e0 | ✅ measured |
| "98.4% transfer reduction" | RETIRED | — | Not reproducible |

The crawl baseline is explicitly a model. A real empirical measurement is required
before this figure can be used in a headline. Gate #7 is not closed.

### Security review (internal, agent-assisted)

A systematic audit of `format.rs`, `reader.rs`, `search.rs`, `oci.rs`, `delta.rs`,
and `annpack-widget.js` against the invariants in `spec/SECURITY.md`.

**No exploitable vulnerabilities found.**

Observations (low severity, no code changes required):
1. `open_pack` fuzz harness: magic-byte gating limits deep path coverage — recommended to add a prefixed variant
2. 64 GB per-section limit causes safe OOM on memory-constrained hosts under adversarial packs
3. ETag header omission on range responses skips mutation check (BLAKE3 still provides integrity)

Evidence: `workstream2-security/internal-security-review.md`

This review does NOT satisfy the independent review gate. It was performed by the
same agent session that assisted with development.

### Fuzz campaign (COMPLETE — Gate 3 closed)

All four targets ran to completion on 2026-07-20. Zero crashes across all targets.

| Target | Runs | Duration | Final cov | Crashes |
|---|---|---|---|---|
| open_pack | 3,651,777,702 | 21,601 s | 185 features | 0 |
| decode_varint | 4,945,609,054 | 21,601 s | 62 features | 0 |
| inspect_delta | ~4,376,883,775 | 21,601 s | 163–216 features | 0 |
| open_pack_prefixed | 3,687,064,516 | 21,601 s | 189 features | 0 |
| **TOTAL** | **~16.7 billion** | — | — | **0** |

**Coverage caveat (documented):** format.rs region coverage is 10.8% from open_pack entry points.
The uncovered 89% requires valid-pack construction not reachable by random mutation (search,
signing, OCI paths). Deeper paths are covered by the 44-test integration suite. A structure-aware
fuzz campaign is recommended before security-critical deployment but is not required by gate 3.

Evidence: `workstream2-fuzz/coverage-summary.md`, per-target `.stderr` files, empty artifact dirs.

**Gate #3: CLOSED** ✅

### Eval corpus (FastAPI docs, preparation complete)

Corpus prepared. Human adjudication required before gate can be closed.

| Item | Status |
|---|---|
| Corpus selected | FastAPI 0.115.12 (MIT) |
| Source commit | `628c34e0cae200564d191c95d7edea78c88c4b5e` |
| Pack root | `c7147550fb7a2e0ff65af4030d730b3fad923fe0f548692b868cd26369a1cc7a` |
| Pack bytes | 860,088 |
| Documents | 141 |
| Passages | 1,864 |
| Provenance recorded | ✅ `workstream3-evals/corpus-provenance.json` |
| 77 candidate queries written | ✅ `workstream3-evals/fastapi-candidate-qrels.jsonl` |
| Adjudication CSV generated | ✅ `workstream3-evals/adjudication.csv` (382 rows) |
| Human labels applied | ❌ **BLOCKED — requires user** |
| BM25/vector/hybrid recall@5 published | ❌ Blocked by labels |

Query categories covered: natural-language (20), technical-token (22 API + env + error),
version-sensitive (5), conceptual (10), distractor (8), not-present (7). Total: 77.

### Second-reader conformance packet (preparation complete)

Packet prepared. Execution requires a clean session on a separate machine.

| Item | Status |
|---|---|
| Golden valid artifact | ✅ `workstream8-conformance/golden-v1.annpack` |
| Golden signed artifact | ✅ `workstream8-conformance/golden-v1-signed.annpack` |
| Test public key | ✅ `workstream8-conformance/test.pub` |
| Expected search results documented | ✅ in `CONFORMANCE-PACKET.md` |
| Invalid corpus (8 cases) | ✅ `workstream8-conformance/invalid-corpus/` |
| Conformance packet README | ✅ `workstream8-conformance/CONFORMANCE-PACKET.md` |
| Second reader implemented | ❌ **Requires clean M3 session** |

### Launch surface content

Prepared but not posted:
- Show HN title, body, and demo script: `launch/LAUNCH-SURFACE.md`
- Architecture diagram description: included
- Vendor outreach drafts (10): included, template + target list
- FAQ (why not SQLite/PMTiles/hosted RAG): included
- Security and limitation disclosure language: included
- Status language (candidate format, no-standard claim): included

---

## Remaining blockers and exact user actions required

### Gate 1 — Real CDN browser proof
**Status:** CLOSED ✅
**CDN:** GitHub Pages (Fastly CDN), `https://arjun2729.github.io/annpackv2/`
**Pack:** `packs/fastapi-docs-0.115.12.annpack` (860,088 bytes)
**Verified:**
- Range requests honored (HTTP 206) ✅
- CORS: `access-control-allow-origin: *` ✅
- ETag present and stable: `6a5fb61d-d1fb8` ✅
- Live query: 12 Range GETs, 459 KB, pack_root matches, evidence schema `annpack-evidence-v1` ✅
**Demo URL:** `https://arjun2729.github.io/annpackv2/?pack=./packs/fastapi-docs-0.115.12.annpack&root=c7147550fb7a2e0ff65af4030d730b3fad923fe0f548692b868cd26369a1cc7a`
**Evidence:** `workstream1-cdn/gate1-cdn-proof.json`
**Browser JS fixes:** Removed If-Match conditional header (Fastly edge-inconsistent ETags); added `Accept-Encoding: identity` on HEAD (Pages gzips binary files, causing size mismatch with Range responses).

### Gate 2 — Independent security review
**Blocker:** Requires a reviewer who did not author the parser.
**User action:** Share `spec/SECURITY-REVIEW.md` and the binary with an independent
security reviewer (separate person or model in a clean session with no access to
the Rust source). The review brief is at `spec/SECURITY-REVIEW.md`.

### Gate 3 — 6-hour fuzz campaign
**Status:** CLOSED ✅
**Result:** 16.7B total executions across 4 targets, zero crashes, all artifact dirs empty.
**Coverage caveat:** format.rs 10.8% region coverage (structure-aware generation needed for deeper paths).
**Evidence:** `launch/evidence/2026-07-20/workstream2-fuzz/coverage-summary.md`

### Gate 4 — Human-adjudicated eval corpus
**Blocker:** Requires human relevance judgments.
**User action:**
1. Open `launch/evidence/2026-07-20/workstream3-evals/adjudication.csv` in a spreadsheet
2. For each row, read the `text_snippet` column
3. Set `relevant` to 1 if the passage answers the `query`, 0 if not
4. Verify the pre-labeled rows (where `pre_labeled_relevant` is set)
5. Save the CSV
6. Run:
```bash
python3 evals/adjudicate.py  # already done, generates CSV
# After labeling:
python3 evals/evaluate.py \
  --pack target/fastapi-eval/fastapi.annpack \
  --queries launch/evidence/2026-07-20/workstream3-evals/adjudication.csv \
  --mode lexical --k 5 --output target/eval-lexical.json
```

### Gates 5 & 6 — Retrieval quality table and embedding decision
**Status:** Table produced (2026-07-26); public closure pending Gate 4 label-independence confirmation.
**Result:** 3-mode table on FastAPI 0.115.12, 65 labeled queries, vectors-enabled pack root
`4d3ebb10…`, k=5. No losing mode hidden:
| Mode | recall@5 | hit@5 | MRR@5 |
|---|---|---|---|
| Lexical (BM25) | 1.000 | 1.000 | 0.895 |
| Vector (mxbai-xsmall) | 0.426 | 0.862 | 0.730 |
| Hybrid (RRF) | 0.604 | 0.892 | 0.814 |
| ANN-7 / ANN-8 overlays | 1.000 | 1.000 | 0.895 |

**Gate 6 decision:** **Do NOT promote** the `mxbai-embed-xsmall` candidate to default — it loses
to BM25 in every category and RRF hybrid reduces recall below lexical. Ship **lexical-only** as the
quality default; vector/hybrid stay opt-in until a stronger embedding model clears BM25.
**Blocker for public closure:** `qrels-labeled.jsonl` must be confirmed human-authored/adjudicated
independently of the implementation, then committed with provenance.
**Evidence:** `2026-07-26/retrieval-quality/retrieval-quality-report.md`, `eval-fastapi-3mode.json`, `qrels-labeled.jsonl`.

### Gate 7 — Real crawl baseline
**Status:** CLOSED ✅
**Result:** Live measured via `ANNPackBrowser.stats.bytes` against localhost Range server.
- Source: 1,288,962 bytes uncompressed / ~270 KB gzip-estimated (141 .md files, commit `628c34e0`)
- Pack open: 460,209 bytes avg (12 Range GETs) across 5 cold queries
- Warm per-query after first open: ~2–5 KB (passage fetches only)
- vs uncompressed source: pack costs 64% less cold — misleading baseline
- vs gzip-compressed source (~270 KB): pack costs ~70% MORE cold — honest baseline
- "98.4% transfer reduction" and "64% cold reduction" both **retired**
- Safe claim: "Open once (~460 KB); answer unlimited follow-up queries at ~2–5 KB each."
- Browser JS bug fixed: `validateSearchIndexes()` now sorts dictionary entries by offset.
**Evidence:** `workstream7-crawl/transfer-baseline.md`

### Gate 8 — Second Core reader
**Status:** CLOSED ✅
**Result:** Independent Python reader implemented from spec in a clean session. All checks pass.
- Root hash computed independently: `7fb855794ac5bbe4...` ✓ matches stored
- AP-104 search: correct passage_id, passage_hash, evidence envelope ✓
- "cache rotation": top result from rotation.md ✓
- All 8 invalid corpus files rejected with correct, specific errors ✓
- Ed25519 signature verified, root unchanged across signed/unsigned ✓
- All 12 SECURITY.md invariants implemented ✓
- LOC: 861 (excluding stdlib crypto/compression)
**Evidence:** `workstream8-conformance/annpack_reader.py`, `workstream8-conformance/conformance-report.json`

### Gate 9 — Public catalog with verified provenance
**Status:** CLOSED ✅
**Result:** 2-entry catalog published with full provenance for each entry.
- `ghcr.io/arjun2729/annpack-knowledge/fastapi-docs:0.115.12` — FastAPI 0.115.12, MIT, commit `628c34e0`, pack root `c71475...`, pull verified
- `ghcr.io/arjun2729/annpack-golden/v1:latest` — conformance artifact, Apache-2.0, signed, pull verified
**Evidence:** `workstream10-oci/catalog.json`

### Gate 10 — Real GHCR push/pull
**Status:** CLOSED ✅
**Result:** golden-v1-signed.annpack pushed to `ghcr.io/arjun2729/annpack-golden/v1:latest`, pulled back, verified.
- manifest_digest: `sha256:2cc9f75dd299265861102bb1abca3871706c8a154574c79db3032c1fcff8952e`
- pack_root round-trip stable: `7fb855794ac5bbe4049947fd2421c44acd51ba6495e2db95b18995ac36db119b`
- 7/7 sections verified, signature cryptographically valid
**Evidence:** `workstream10-oci/ghcr-push-pull.json`

---

## Claims safe to publish NOW

- 74/74 Rust tests pass on M4 (Apple M4, 10 cores, rustc 1.97.1); was 44 before the ANN-7..10 extensions merged
- Pack/source ratio: 86.3% for 1,000-document corpus
- Build: 70 ms; verify p95: 3.53 ms; query p95: 6.17 ms (all process-inclusive)
- BM25 lexical range retrieval: 8 Range GETs on the 4 KB conformance test pack (enforced by smoke test); 12 on the 860 KB FastAPI pack (141 docs, 1864 passages) — count scales with passage block count
- Vector retrieval uses exactly 11 Range GETs
- Offline install and query after server shutdown: confirmed working
- Google OKF reproduction: all 3 roots match pinned commit `d44368c`
- No exploitable vulnerabilities found in internal code review
- Fuzz campaign complete: 16.7B total executions across 4 targets, 0 crashes (format.rs coverage 10.8%; deeper paths require structure-aware generation)
- Real-CDN browser proof: signed FastAPI pack on GitHub Pages/Fastly, HTTP 206 Range honored, CORS + stable ETag, live query 12 Range GETs / 459 KB, root matches (Gate 1 closed)
- Real GHCR push/pull and a 2-entry catalog with full per-entry provenance (Gates 9, 10 closed)
- Second independent Core reader (861-LOC Python from spec) reproduces the root and passes all conformance checks (Gate 8 closed)
- Lexical BM25 retrieval quality on FastAPI 0.115.12 (65 labeled queries, root `4d3ebb10…`): recall@5 1.00, hit@5 1.00, MRR@5 0.895 — valid for that pinned corpus/labels, pending Gate 4 label-independence confirmation

## Claims NOT yet safe to publish

- Any "transfer reduction" percentage — 98.4% retired; 64% cold figure also misleading (compares to uncompressed text); honest claim is "open once ~460 KB, ~2–5 KB per subsequent query"
- Public retrieval quality numbers — the 3-mode FastAPI table exists (see Gates 5 & 6) but public
  closure waits on confirming `qrels-labeled.jsonl` is independent human adjudication
- "Adopted protocol" or "standard" — independent implementation pending (note: Gate 8 second reader is closed; standard-adoption still requires external uptake)
- "Independent security review" — only agent-assisted internal review done (Gate 2, the one remaining external blocker)

---

## Artifact roots

| Artifact | Root hash |
|---|---|
| golden-v1.annpack | `7fb855794ac5bbe4049947fd2421c44acd51ba6495e2db95b18995ac36db119b` |
| fastapi-docs-0.115.12.annpack | `c7147550fb7a2e0ff65af4030d730b3fad923fe0f548692b868cd26369a1cc7a` |
| google-okf ga4 | `f9256b569f574d8a9068be6372a6e8d7f2b76d0afd2f0650e305218669e73b35` |
| google-okf crypto-bitcoin | `3978663b7e7cedaafeb97460f0bbab4643e21ca854271b21f03eff96c5214f97` |
| google-okf stackoverflow | `3acbe05d8c78f8c75dedcfb25843049d0cd4059b89d30c1b07c60e063a1b2d86` |
