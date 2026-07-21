# Internal Security Review — ANNPack v0.3.0

**Reviewer:** Claude Sonnet 4.6 (agent-assisted, not independent)
**Date:** 2026-07-20
**Commit:** 7f2c2bbd05400bb37234bd974725e0d844383a6b
**Scope:** Binary parser, delta, OCI transport, browser rendering
**Status:** INTERNAL ONLY — does not satisfy the independent reviewer gate

This review is performed by the same agent session that assisted with development.
It is useful for surfacing issues before independent review but MUST NOT be labeled
as "independent." The launch gate requires a reviewer who did not author the parser
or its tests, in a genuinely clean session.

---

## Binary Parser (format.rs, reader.rs, search.rs)

### Header parsing

- Magic bytes checked (`ANNPACK3`). ✅
- Format version checked (rejects non-3). ✅
- Header size field checked against `HEADER_SIZE` constant. ✅
- Reserved bytes [80..128] checked to be zero. ✅
- `section_count` bounded: `0 < count <= MAX_SECTIONS (16,384)`. ✅
- `directory_length` validated with `checked_mul` against `section_count * DIRECTORY_ENTRY_SIZE`. ✅
- Directory range validated with `checked_add` before IO. ✅
- Directory bytes allocated with `usize::try_from(directory_length)` — safe on all platforms. ✅
- Root hash computed from decoded entries and compared **before** any section is interpreted. ✅

### Directory parsing

- Reserved bytes per entry [76..80] checked to be zero. ✅
- Entries decoded with `chunks_exact(DIRECTORY_ENTRY_SIZE)` — no partial reads. ✅
- Strictly-increasing section ID order enforced; duplicate IDs detected at two levels (windowed scan + BTreeSet). ✅
- Duplicate singleton section types rejected (Signatures are explicitly allowed to repeat). ✅
- Unknown required sections rejected; unknown optional sections accepted. ✅
- Unknown required codecs rejected; unknown optional codecs accepted. ✅

### Section validation

- Per-section: `stored_length` and `logical_length` both checked against `MAX_SECTION_SIZE (64 GB)`. ✅
- Decompression-ratio limit: `stored_length * 256` or floor of `16 MB`, whichever is larger. Uses `saturating_mul` (safe: 64 GB × 256 = 17.6 TB, below u64::MAX). ✅
- `checked_file_range` uses `checked_add` for all offset+length; rejects beyond file length. ✅
- Section-header overlap detected. ✅
- Section-directory overlap detected. ✅
- Section pairwise overlap detected via sort+windows(2) — sufficient because sorting ensures transitivity. ✅
- Zero-length uncompressed sections allowed (stored == logical == 0). Not a vulnerability; consistent with spec.

### Section reading

- `read_stored_section`: BLAKE3 hash verified before returning bytes. ✅
- `read_section` (Deflate): decompression capped at exact `logical_length`; panics if result ≠ limit. ✅
- `read_section_range`: `checked_add` for end, bounds check against `stored_length`, `checked_add` for absolute. ✅
- Manifest re-validates type, required flag, size limit on every `manifest()` call. ✅

**Observation (low):** `vec![0u8; stored_length_usize]` in `read_stored_section` can allocate up to 64 GB per section — the per-section limit. On memory-constrained systems this causes a safe OOM panic. The constraint is enforced but the absolute ceiling is large. Consider documenting this as a resource-exhaustion risk for adversarial packs from untrusted sources.

### HTTP range reader

- `checked_range` called before issuing any HTTP request. ✅
- Status 206 checked; any other status rejected. ✅
- `Content-Range` header validated against exact expected value. ✅
- ETag stored at `open()`, compared on each range response. ✅
- Response body capped with `.take(buffer.len() as u64 + 1)` then length checked. ✅

**Observation (low):** If the server returns `ETag` on the HEAD request but omits it on subsequent range responses, the `if let (Some(expected), Some(actual))` guard skips the comparison. Content mutation is still caught by BLAKE3 section verification. The ETag check is defense-in-depth; its absence is safe but worth noting for the independent review.

**Observation (low):** `buffer.len() as u64 + 1` — if `buffer.len()` == `usize::MAX`, this overflows on a 64-bit system where `usize == u64`. Unreachable in practice because `buffer.len()` is bounded by `MAX_SECTION_SIZE` (64 GB) through section read paths.

### Varint decoder

- Loop bounded to 10 iterations (70 bits / 7 = 10). ✅
- Overflow check at shift=63: rejects `byte > 1`. ✅
- Truncation: out-of-bounds byte access returns error. ✅
- Non-termination: 10th iteration without terminator returns error. ✅

### Passage-block verification

- Per-block: `logical_length > MAX_PASSAGE_BLOCK_LOGICAL_SIZE (1 MB)` rejected. ✅
- Per-block compression ratio limit applied with `saturating_mul`. ✅
- Block range accumulation uses `checked_add`. ✅
- Passage record range computed with `checked_add`. ✅
- Posting offset and end both converted with `usize::try_from`. ✅
- Decompressed block verified by exact length match. ✅

### Vector index

- Vector count × stored_dimensions: `checked_mul`. ✅
- Total byte count: `checked_mul` + `checked_add`. ✅
- Non-finite vector values: rejected before search. ✅
- Dimension consistency enforced between profile and actual data. ✅

### Fuzz harness reachability

**Observation (medium):** The `open_pack` fuzz target calls `PackReader::open` on arbitrary bytes. The very first check is the 8-byte magic `ANNPACK3`. Libfuzzer will discover this prefix through coverage-guided mutation and the seed corpus, but a large fraction of random inputs are rejected immediately at the magic check. Adding an `open_pack_prefixed` variant that prepends the magic bytes would increase coverage of the directory and section parsing paths.

Confirmed: `decode_varint` and `inspect_delta` harnesses reach logical parsers directly — not gated by magic bytes.

---

## Delta Parser (delta.rs)

- Target length checked against `MAX_DELTA_TARGET_SIZE (512 MB)` before allocation. ✅
- Operation count bounded by three independent limits: `MAX_DELTA_OPERATIONS (1,000,000)`, `target_length + 1`, and `payload_size / 9` (minimum bytes per operation). All three enforced before `Vec::with_capacity`. ✅
- Zero-length Copy and Add operations rejected. ✅
- Per-operation logical length accumulated with `checked_add`. ✅
- Incremental check: `logical_length > target_length` after each operation. ✅
- Final check: `cursor == bytes.len() && logical_length == target_length`. ✅
- `apply_operations`: `extend_bounded` checks against capacity before each extend. ✅
- After reconstruction: full `PackReader::open + verify_all()` before accepting result. ✅
- Root hash verified against declared `target_root`. ✅
- Snapshot delta: `body.len() == target_length` checked. ✅
- `apply_delta`: base root must match delta's `base_root` field. ✅

No issues found in delta parsing.

---

## OCI Authentication and Transport (oci.rs)

### Credential handling

- Credentials rejected over non-HTTPS non-loopback transport. ✅
- Loopback detection covers `localhost`, `127.0.0.1`, `[::1]`. ✅
- Bearer realm: must be HTTPS or loopback with same-origin as registry. ✅
- Basic credentials not forwarded to cross-origin upload targets (cross-origin PUT uses plain `ureq::put` without auth header). ✅

### Bearer challenge parser

- `Bearer ` prefix checked case-insensitively. ✅
- Quoted-string escape handling: `\\x` → `x`. ✅
- Unterminated strings rejected. ✅
- Non-UTF-8 key and value bytes rejected. ✅
- Duplicate keys: last value wins (BTreeMap insert); not explicitly rejected but the duplicate-key case in real challenges is benign since only `realm`, `service`, `scope` are consumed.
- Commas inside scope values (e.g., `pull,push`) handled correctly because value parsing is quote-delimited, not comma-delimited. ✅

### Registry reference parser

- `..` in repository path rejected. ✅
- Uppercase characters rejected. ✅
- `?` and `#` in name rejected. ✅
- Repository byte allowlist: lowercase alphanumeric, `/`, `.`, `_`, `-`. ✅
- Tag byte allowlist: alphanumeric, `.`, `_`, `-`. ✅
- Digest references validated with `valid_sha256_digest`. ✅
- `valid_sha256_digest`: length=71, prefix "sha256:", lowercase hex only. ✅

**Observation (low):** `valid_sha256_digest` rejects uppercase hex. The OCI spec permits uppercase. Since ANNPack always generates digests as lowercase (via `{:x}`), received digest strings from non-conforming registries using uppercase would fail validation. This is conservative and safe.

### Blob upload redirect

- `Location` resolved via `resolve_location` using `url::Url::parse` then relative join. ✅
- Same-origin check uses `url::Url::origin()` which includes scheme, host, and port. ✅
- Cross-origin uploads require HTTPS scheme. ✅
- Foreign origin PUT does not carry the authorization header. ✅

**Observation (low):** `read_bounded_response` uses `take(limit.saturating_add(1))`. If `limit` = `u64::MAX`, saturation yields `u64::MAX` bytes read, and the bounds check `bytes.len() as u64 > limit` would be `u64::MAX > u64::MAX = false`. In practice, manifest limit is 16 MB and blob limit is the descriptor size, both far from `u64::MAX`.

### Pull integrity

- OCI SHA-256 digest verified after download. ✅
- Pack length vs. descriptor size verified. ✅
- `PackReader::open + verify_all()` before installation. ✅
- BLAKE3 content root verified against OCI annotation. ✅
- Atomic write: `create_new(true)` then `rename`. ✅
- Temporary file cleaned up on error. ✅

---

## Browser Rendering

- All untrusted pack strings (title, text, evidence, status) assigned via `textContent`. ✅
- No `innerHTML`, `insertAdjacentHTML`, `eval`, or `new Function` found in widget or browser client. ✅
- URLs rendered as link `href`; subject to embedding page CSP and browser navigation policy. Acceptable per SECURITY.md. ✅

---

## Summary

| Surface | Critical | High | Medium | Low | Observations |
|---|---|---|---|---|---|
| Binary parser | 0 | 0 | 0 | 1 | 64 GB per-section OOM (safe) |
| HTTP range reader | 0 | 0 | 0 | 2 | ETag skip; take overflow (unreachable) |
| Varint decoder | 0 | 0 | 0 | 0 | |
| Passage blocks | 0 | 0 | 0 | 0 | |
| Delta parser | 0 | 0 | 0 | 0 | |
| OCI transport | 0 | 0 | 0 | 2 | uppercase digest; saturation |
| Fuzz reachability | 0 | 0 | 1 | 0 | open_pack gated by magic bytes |
| Browser rendering | 0 | 0 | 0 | 0 | |

**No exploitable vulnerabilities found.**

All security invariants from `spec/SECURITY.md` are implemented. Arithmetic is checked
throughout. Allocation limits are applied before allocation. Verification precedes
interpretation. The principal security concern is resource exhaustion (64 GB section limit
causing safe OOM), not memory-safety or data-integrity violations.

### Recommended action before independent review

1. Add `open_pack_prefixed` fuzz variant that prepends `ANNPACK3` + valid header prefix to
   increase post-magic coverage depth.
2. Document the 64 GB per-section ceiling in the security model as a resource-exhaustion
   bound rather than hiding it in code.
3. Complete the independent review session on the M3 with the packet produced in Workstream 8.
   Agent-assisted review does not satisfy the gate.

---

*This review covers commit 7f2c2bbd. Any changes to the parser, OCI, or delta code require
re-review before re-marking the gate as attempted.*
