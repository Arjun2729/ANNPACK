# ANNPack Release Readiness

**Generated:** 2026-07-30
**Version:** `v0.4.0-rc4`
**Root scheme:** manifest section format 2 — artifact root + logical content root
**Machine:** Apple M4, 10 cores, 16 GB RAM, macOS 26.2
**Toolchain:** rustc stable (MSRV 1.88)

> **Scope:** this is the current release-candidate **evidence ledger** — what is
> proven on this build. The stricter **public-launch and public-claims**
> checklist lives in [`spec/LAUNCH-GATES.md`](../spec/LAUNCH-GATES.md); closing a
> gate here does not tick the corresponding stricter box there.

This document is regenerated at each release candidate. It states one date, one
version, and one root scheme. Historical evidence lives under
`launch/evidence/<date>/` and is never silently mixed in here.

---

## Status: NOT READY for external release

**6 of 11 gates closed.** The blockers are no longer implementation quality —
they are external validation and retrieval evidence, neither of which can be
closed from inside this repository.

| # | Gate | Status |
|---:|---|---|
| 1 | Internal verification (build, lint, tests, smokes, benches) | ✅ closed |
| 2 | Independent security review | ❌ **external — not started** |
| 3 | Fuzz campaign | ✅ closed |
| 4 | Independent human relevance labels | ❌ open |
| 4b | Conformance packet | ✅ **closed** |
| 5 | Publishable retrieval-quality table | ❌ open (withdrawn) |
| 6 | Embedding promotion decision | ❌ open (blocked on 4/5) |
| 7 | Crawl-vs-pack transfer claim | ⚠️ measured, not headline-safe |
| 8 | Second independent Core reader | ❌ **reopened** |
| 9 | Public CDN + reproducible demo | ✅ **closed** — live and independently verified |
| 10 | OCI catalog published | ⚠️ re-push owed under the new root scheme |

---

## What v0.4.0-rc4 changed

Rc4 supersedes the rc3 verifier implementation without changing the receipt
schema or any pack root. Rc3 correctly authenticated receipt labels and canonical
URLs, but its verifier trusted attacker-controlled directory lengths before
applying section and decompression-ratio limits, and it attempted zlib inflation
for every Documents section even though codec 0 is valid.

Rc4 validates the carried directory structure and committed lengths, bounds every
base64 component before decoding, caps proof depth, applies `MAX_SECTION_SIZE` and
the 256:1 expansion policy before allocation, handles codecs 0 and 1 explicitly,
and rejects unknown receipt schemas and codecs. Existing honest
`annpack-receipt-v2` receipts remain valid; packs do not need rebuilding.

The complete native, bindings, integrations, conformance, benchmark, WASM, and
four-way same-builder determinism matrix passed on the rc4 branch. The golden
artifact root remains unchanged.

## What v0.4.0-rc3 changed

An evidence-receipt hardening finding from the internal audit. A signed receipt
proved the passage *record* was in the signed artifact, but its descriptive and
provenance fields — `source_revision`, `pack`, `passage_id`/`passage_ordinal`,
and `canonical_url` — were carried at the top level and never bound. A receipt
legitimately issued and signed could therefore be rewritten to attribute an
authentic passage to a forged revision or an attacker-controlled URL while still
reporting `verified: true` under the publisher's trusted key.

The verifier now binds all of them: the first four against the already-carried
manifest and passage record (no new bytes), and `canonical_url` against the
Documents section — now carried in the receipt — whose stored bytes hash to a
directory entry that `pack_root` already commits. A URL claim with no Documents
section to authenticate it fails, so stripping the section cannot downgrade the
check. Receipt schema is bumped `annpack-receipt-v1` → `-v2`; the `EVIDENCE-v1`
verification procedure gains the corresponding steps, and the published
`evidence.json` vector is regenerated. **Artifact roots are unchanged** — this
touches the receipt format, not the container — so Markdown, OKF, golden, and
conformance-packet roots all carry over from rc2 untouched.

Regression coverage in `tests/receipt_tamper.rs` replays each forgery (including
the section-drop downgrade) and asserts verification now fails. Full suite
100 → 106 tests, clippy clean.

## What v0.4.0-rc2 changed

Two OKF conformance defects found while preparing outreach against the newly
published **OKF v0.2**. Both were ours, not upstream changes.

1. **We rejected a conformant bundle.** Our validator refused any `log.md`
   carrying frontmatter. Neither v0.1 nor v0.2 says that: v0.1's "Index files
   contain no frontmatter" governs `index.md`, and v0.2 §9 constrains only the
   body's date-grouped structure. The rule was invented, and it made Google's
   own v0.2 exemplar bundle (`acme_retail`, whose `log.md` carries `type: Log`)
   unbuildable. v0.2 §11 is explicit that consumers MUST NOT reject a bundle over
   additional frontmatter keys.
2. **We invented a version.** `okf_version` is optional (§12); absent means
   *undeclared*, not `0.1`. We recorded `source.version: "0.1"` for any bundle
   omitting it, mislabelling v0.2 content as v0.1.

Fixing (2) changes the artifact roots of OKF-sourced packs, hence rc2 under
[`spec/COMPATIBILITY.md`](../spec/COMPATIBILITY.md). Markdown-sourced packs are
unaffected: the golden root and the conformance packet root are unchanged.

Verified against upstream `acme_retail` at `3fcbb9f`: 17 documents, 47 passages,
and the v0.2 provenance/trust/lifecycle families (`generated`, `verified`,
`status`, `stale_after`, `tags`) all preserved losslessly in document metadata.

## What v0.4.0-rc1 changed

### Compatibility boundary made explicit

v0.3.1 removed the required manifest `builder` field while leaving the wire
version, manifest section format, and media type unchanged. New readers accepted
old packs; old readers failed on new packs with a bare ``missing field `builder` ``
error. That was a one-way break shipped as a patch release.

v0.4.0 bumps the **manifest section format to 2**. Unknown manifest versions are
now refused at the container boundary with an explicit version error rather than
failing inside JSON deserialization. Five `tests/compatibility.rs` tests pin the
behaviour in both directions, and `spec/test-vectors/compat/` keeps a permanent
v0.3-era artifact as a fixture.

### The cross-builder root claim was false and is corrected

v0.3.1 claimed "any conformant builder produces the identical content root."
Untrue: the root commits to DEFLATE output, block packing, section offsets, and
JSON serialization, none of which is normatively specified.

The value is now named the **artifact root** and documented as an identity for
one artifact. The CI job is renamed `same-builder-determinism` because that is
all it proves. A new **logical content root** (`passage_merkle_root`) provides
the builder-independent commitment. See
[ADR-0003](../spec/decisions/0003-artifact-root-and-logical-content-root.md).

### Core scoring is now normative

The specification previously said only that "terms containing digits or technical
punctuation receive an explicit exact-token boost" — no boost value, no
punctuation set, no tokenizer. A clean-room Python reader written from that text
chose boost `2.0`, a three-character punctuation set, and a tokenizer that split
`std::move` into `std` and `move`. It passed the conformance suite because the
golden queries never exercised those tokens.

FORMAT-v3 §6.1–6.2 now fully specify normalization, the seven-character technical
punctuation set, the boost value `3.0`, and the exact BM25 formula, with a worked
tokenization example.

### Evidence receipts

`annpack receipt` / `annpack verify-evidence` and the
`knowledge_evidence_receipt` MCP tool issue a self-contained proof that a passage
existed unmodified in a named artifact. Verification needs no pack, no network,
and no trust in the issuer. Measured: **4,306 bytes / 11 proof steps** on the
860 KB, 1,864-passage FastAPI pack. Specified in
[EVIDENCE-v1](../spec/EVIDENCE-v1.md).

### Defects fixed

| Defect | Effect |
|---|---|
| Malformed ANN-10 descriptor could steer the default lexical path | Core and extension conformance are now independent; an invalid descriptor is ignored entirely and profile requests are refused |
| Selecting one overlay loaded every overlay | Loader is scoped to the selected profile's `section_ids`; asserted profile-to-profile |
| Unknown profile kinds and empty `requires` silently "supported" | Both rejected; an unrecognized strategy is never reported as having run |
| Char/byte confusion in oversized multibyte passages | `source_byte_start/end` are now byte-accurate |
| Delta matcher lost 8-byte alignment after an odd-length match | Copy reuse on a localized change: **466 → 10,008 bytes** |
| MCP could not discover retrieval profiles | `knowledge_pack_info` returns profiles, support status, unmet capabilities, and derived-input provenance |
| ANN-7/8 specified an impossible `sidecar_digest` rejection rule | Removed; replaced with the re-derivation procedure that does work |
| Legacy C `ANNP` v1 format sat in `src/` at the repo root | Moved to `attic/legacy-ann-v1/` with an explanatory README |
| `docs/` and `web/` were hand-maintained duplicates | `docs/` is generated by `scripts/sync-docs-site.sh`; CI fails on drift |
| Tracked WASM was stale and CI masked it by rebuilding first | Rebuilt; CI now fails if tracked output differs from a fresh build |

---

## Gate 1 — Internal verification ✅

| Check | Result |
|---|---|
| `cargo fmt --check` | PASS |
| `cargo clippy --all-targets --all-features -D warnings` | PASS |
| `cargo test --all-targets --all-features` | PASS |
| `cargo build --release` | PASS |
| Node binding / Python binding / framework integrations | PASS |
| Browser smokes (base, range, offline, vector, transformers, OKF) | 6/6 PASS |
| `benches/benchmark.py --enforce` | PASS |
| `benches/crawl_vs_pack.py --enforce` | PASS |
| `benches/extensions_bench.py` | PASS |

Golden artifact root: `b1f63b4acdbee0a89de5c3455505be279845b4eda644c0d6c931814355a9d70b`

## Gate 2 — Independent security review ❌

Not started. The prior internal review was performed by the same agent session
that assisted development and **does not** satisfy this gate. A brief for an
external reviewer is ready at
[`external-review/SECURITY-REVIEW-BRIEF.md`](external-review/SECURITY-REVIEW-BRIEF.md).
Requires engaging and paying an unaffiliated researcher or firm.

## Gate 3 — Fuzz campaign ✅

Four targets, ~16.7 billion executions, zero crashes (2026-07-20 evidence).
Coverage caveat stands: `format.rs` region coverage is 10.8% from `open_pack`
entry points; deeper paths are covered by the integration suite. A
structure-aware campaign is recommended before security-critical deployment.

## Gates 4–6 — Retrieval quality ❌

The 2026-07-26 FastAPI report is
[withdrawn](evidence/withdrawn/2026-07-26-retrieval-quality/WITHDRAWN.md).
Saturated benchmark, vector rows not reproducible from committed inputs, and
ANN-7/8 rows evaluated a pack containing no overlays. `evals/evaluate.py` now
refuses `--compare-extensions` unless the pack declares the extension.

**No retrieval-quality claim is currently supportable.** Owed: a hard-negative
evaluation, committed complete inputs, separated passage/document metrics, and
independently produced labels.

## Gate 7 — Transfer claim ⚠️

Safe claim only: *"open once ≈460 KB, ≈2–5 KB per subsequent query in the same
session"* on the 860 KB FastAPI pack. The "98.4% reduction" figure is retired and
must not be revived.

## Gate 4b — Conformance packet ✅ (new)

[`spec/conformance/`](../spec/conformance/README.md) is complete: discriminating
corpus, artifacts, tokenizer vectors, exact IEEE-754 scoring vectors,
compatibility vectors, corruption corpus, signature vectors, a published receipt,
a one-command runner, and a machine-readable report. The reference
implementation scores **42/42**, and CI re-runs the packet on every build and
fails if the report drifts.

Building it surfaced a real interoperability hazard: **serde_json's default float
parser loses up to 1 ULP**, so a score written and read back does not compare
equal. Scores are now published as IEEE-754 bit patterns and the reference
enables `float_roundtrip`.

The packet also encodes the normativity rule: the specification is normative and
the reference implementation is what changes when they disagree.

## Gate 8 — Second independent Core reader ❌ (reopened)

Previously marked closed. **Reopened**, for two reasons:

1. The clean-room Python reader is repository-owned and agent-produced. It is
   valuable interoperability evidence and it is **not** an independent
   implementation. It is reclassified as a *clean-room, agent-assisted second
   implementation*.
2. It disagreed with the reference implementation on tokenization and boost —
   the divergence that prompted the normative scoring rules above. It must be
   re-run against the new conformance vectors, which assert exact scores.

Brief for a paid external implementer:
[`external-review/CORE-READER-BRIEF.md`](external-review/CORE-READER-BRIEF.md).

## Gate 9 — Public CDN + reproducible demo ✅

Live at <https://arjun2729.github.io/annpackv2/>, verified against the published
origin over HTTPS rather than a local server:

| Check | Result |
|---|---|
| `Accept-Ranges: bytes`, stable ETag | yes |
| Range GET returns `206` with exact bytes | yes (`ANNPACK3` magic at `0-7`) |
| Published client verifies published artifact | yes |
| Artifact root | `b6d50106…` — matches `expected-roots.json` |
| Range requests for one cold query | 8 |
| Transferred | 28,298 bytes |
| Evidence bound to root | yes |

`./launch/google-okf/reproduce.sh` verifies all three OKF roots from a clean
fetch of the pinned upstream revision.

One caveat: GitHub Pages serves the pack as `application/octet-stream`, not
`application/vnd.annpack.v3`. Range semantics are unaffected and the client does
not rely on the content type, but a production origin should set it.

## Gate 10 — OCI catalog ⚠️

The GHCR re-push requires publisher credentials and must be redone under the
v0.4.0 root scheme. Every root in this repository has been regenerated; roots in
`launch/evidence/2026-07-20/` and in `workstream10-oci/` are historical and are
labelled as such.

---

## Known open questions carried into external review

Declared, not discovered. None of these is an implemented protection, and none
may be described as one in outreach or documentation.

| # | Question | Status |
|---:|---|---|
| A | Is the bounded-inflation requirement adequately stated? | Rc4 enforces it in the receipt verifier; specification sufficiency remains externally unreviewed |
| B | Is the ADR-0004 freshness/revocation model sound? | **Design only — no implementation, no wire contract** |
| C | Is the receipt Merkle construction second-preimage resistant? | Deliberate choices, never externally reviewed |
| D | Is `format.rs` fuzz coverage (10.8% of regions) adequate? | Structure-aware campaign not run |

**A.** Until v0.4.0 `SECURITY.md` said only that a parser "enforces ...
decompression-ratio ... limits." Our clean-room reader read that as permitting a
post-hoc length check and inflated unbounded — while reporting full invariant
coverage. The reference implementation always bounded the decompressor; the
*specification* did not clearly require it. Now it does. Whether the new wording
suffices for an implementer who has not seen the note is for the reviewer.

**B.** Rollback resistance is an **unsolved problem in this release**. A receipt
for a superseded artifact verifies correctly and forever. ADR-0004 records the
intended model and is explicitly not built. Do not present freshness or
revocation as a capability.

---

## Current roots (v0.4.0-rc4; unchanged from rc2)

| Artifact | Artifact root |
|---|---|
| Golden `minimal-v3.annpack` | `b1f63b4acdbee0a89de5c3455505be279845b4eda644c0d6c931814355a9d70b` |
| `docs/docs-v1.annpack` | `c1a3cab853ec70f007672eeb46b3b39452b1d253ad67b888b2ff802ed497ecff` |
| `docs/docs-v2.annpack` | `1682d0515538aa128aa462a042c788f0f95f0aed217e3b1c6824f6bc740f9671` |
| FastAPI 0.115.12 | `49a1636457ac9ae0e4755bf232c718ae90cccba27933695f4a704eeddefec8a2` |
| Google OKF ga4 | `b6d50106c32ef2e9e944b98e589e81378948163d134ed53b26eeb5262327960b` |
| Google OKF crypto-bitcoin | `6b0f7d6c28a807db3a715bdc449add64482063c631ccc9aa563cbe69c82e2f03` |
| Google OKF stackoverflow | `3e81efeac44cfc743a6754750ef37c12e161dda827f1f0a929d41da5c545b2fe` |

Regenerate with `scripts/build-demo-packs.sh` and `launch/google-okf/reproduce.sh`.

---

## What must happen next, in order

1. **Engage the two paid external parties** (Gates 2 and 8). Nothing else
   substitutes; internal work cannot close either. Both briefs and the
   conformance packet they depend on are ready — this is now purely a funding
   decision.
2. **Build the hard-negative evaluation** (Gates 4–6). It is also the only way to
   learn whether ANN-1/7/8 are worth keeping.
3. **Re-push GHCR** under the v0.4.0 root scheme (Gate 10). Needs publisher
   credentials. The stricter production-CDN requirement — correct media type,
   caching, and CORS on a real origin — remains tracked separately in
   [`spec/LAUNCH-GATES.md`](../spec/LAUNCH-GATES.md); the GitHub Pages
   reproducible demo that closed Gate 9 does not satisfy it.
4. **Send the OKF technical-validation request** (see
   [`google-okf/OUTREACH.md`](google-okf/OUTREACH.md)) — asking for reproduction
   and review, not endorsement.

Feature freeze holds throughout: ANN-1 through ANN-10 are frozen. No new
extensions, rankers, or models until these gates close.
