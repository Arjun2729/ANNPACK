# Fuzz Coverage Summary

**Date:** 2026-07-20
**Machine:** Apple M4, 10 cores, 16 GB RAM
**Toolchain:** rustc 1.97.1 (stable), cargo-fuzz, nightly llvm-tools

---

## Execution totals (all targets, zero crashes)

| Target | Runs | Duration | Final cov features | Crashes |
|---|---|---|---|---|
| open_pack | 3,651,777,702 | 21,601 s | 185 | 0 |
| decode_varint | 4,945,609,054 | 21,601 s | 62 | 0 |
| inspect_delta | ~4,376,883,775 | 21,601 s | 163–216 | 0 |
| open_pack_prefixed | 3,687,064,516 | 21,601 s | 189 | 0 |
| **TOTAL** | **~16.7 billion** | — | — | **0** |

All artifact directories empty: `fuzz/artifacts/{open_pack,decode_varint,inspect_delta,open_pack_prefixed}/`

---

## Coverage analysis (llvm-cov region coverage)

Coverage measured via `cargo fuzz coverage` with nightly instrumented binaries.

### open_pack / open_pack_prefixed

| File | Regions | Missed | Covered |
|---|---|---|---|
| format.rs | 954 | 851 | 10.8% |
| reader.rs | 99 | 64 | 35.4% |
| search.rs | 1261 | 1261 | 0% |
| delta.rs | 727 | 727 | 0% |
| Total (all files) | 5268 | 5130 | 2.6% |

open_pack_prefixed reached 4 additional coverage features (189 vs 185) by bypassing the magic-byte gate, confirming it reaches slightly deeper directory/section parsing paths.

### decode_varint

| File | Covered |
|---|---|
| varint decode path | well-saturated (62 features, plateau) |
| search.rs (incidental) | ~1.9% |

### inspect_delta

| File | Covered |
|---|---|
| delta.rs | 31.4% |

---

## Coverage plateau interpretation

All targets reached their feature plateau within the first few minutes and did not discover new paths in the remaining 6 hours. This is expected and informative:

**What is covered:**
- Magic header validation (open_pack)
- Directory parsing (open_pack, partially via open_pack_prefixed)
- Section header reads and offset validation (open_pack)
- All reachable varint decode paths and edge cases (decode_varint)
- Delta header parsing, tag dispatch, copy/insert logic (inspect_delta ~31%)

**What is NOT covered and why:**
- `search.rs`, `signing.rs`, `oci.rs`: require a fully valid, internally consistent pack structure as a prerequisite. Random mutation from the entry points cannot construct valid BM25 indexes, IVF-flat structures, or signed section envelopes.
- Deeper format.rs paths (89% missed): section decompression, content hash verification, passage block parsing — all require the parser to first accept a valid header + directory, which libfuzzer's random mutation rarely produces even with the prefixed variant.

**Implication:** The zero-crash result is meaningful for the paths that ARE covered. For the uncovered paths, the guarantee is that the test suite (44 tests) and integration smoke tests provide coverage — the fuzz harnesses cannot reach them from random bytes without structure-aware generation.

---

## What would extend coverage

1. **Structure-aware generation**: A custom `Arbitrary` impl that generates valid-structure packs with mutated field values, targeting the decompression and verification paths.
2. **Seed corpus from real packs**: Seeding open_pack with actual .annpack files would immediately reach deeper parsing paths.
3. **Targeted harnesses**: A `verify_section` harness that takes (section_bytes, expected_hash) and tests the BLAKE3 verification path directly.

These would be meaningful improvements for a future fuzz campaign but are not required for gate 3 as stated.

---

## Conclusion for Gate 3

Gate 3 criteria as stated in `spec/LAUNCH-GATES.md`:
- "Complete at least one six-hour-per-target deep fuzz run across all three targets" ✅ (21,601s each)
- "Zero crashes" ✅
- "Preserve corpus and crash artifact directories" ✅

**Gate 3: CLOSED**

Coverage caveat: format.rs region coverage is 10.8% from the open_pack entry point. The uncovered paths require valid-pack construction not reachable by random mutation. This limitation is documented here and in RELEASE-READINESS.md. A structure-aware fuzz campaign is recommended before any security-critical deployment.
