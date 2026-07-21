# ANNPack Core v1.0-draft — Second Reader Conformance Packet

This packet is for use in a **clean, separate session** to implement an independent
Core reader. Do not use this session's Rust source code or tests.

## Goal

Implement a read-only Core reader in a language other than Rust, in approximately
500 lines excluding standard JSON, HTTP, compression, and cryptographic libraries.
The target language should be Python, Go, TypeScript, or similar.

## Specifications to read (in order)

1. `spec/CORE-v1.0-draft.md` — normative requirements
2. `spec/FORMAT-v3.md` — binary encoding details
3. `spec/SECURITY.md` — security invariants (all MUST be implemented)
4. `spec/MEDIA-TYPES.md` — media type handling
5. `spec/PROTOCOL-v1.md` — HTTP range access

Do NOT read or reference:
- `rust/src/format.rs`
- `rust/src/reader.rs`
- `rust/src/search.rs`
- Any other Rust implementation files
- This session's notes or outputs beyond this packet

## Golden valid artifact

**File:** `golden-v1.annpack`
**Pack name:** `golden-docs`
**Version:** `1.0.0`
**Source revision:** `git:v1`
**Base URL:** `https://example.com/docs`
**Documents:** 3
**Passages:** 7
**Bytes:** 4184
**Root hash (BLAKE3, hex):** `7fb855794ac5bbe4049947fd2421c44acd51ba6495e2db95b18995ac36db119b`

The reader MUST independently compute this root hash from the directory and
verify it matches the stored root hash in the header.

## Golden signed artifact

**File:** `golden-v1-signed.annpack`
**Root hash (same as unsigned):** `7fb855794ac5bbe4049947fd2421c44acd51ba6495e2db95b18995ac36db119b`
**Signing key fingerprint:** `65727b7782488cf34bc72770ca91b2dd254a9514edaa2f451648cd43de4995fc`
**Public key file:** `test.pub` (Ed25519, 32 bytes, raw)
**Bytes:** 4744 (560 more than unsigned — one Signature section added)

Signature MUST NOT change the content root (signatures are excluded from root computation).

## Expected search results

### Query: "AP-104" (lexical mode)

The top result MUST be:
```json
{
  "passage_id": "073b6867886b39c069a287c9ea426dbada5275b76948257b201836e0878f7c2e",
  "heading_path": ["Authentication errors", "AP-104"],
  "source_path": "errors.md",
  "url": "https://vendor.example/docs/v1/errors#ap-104",
  "text": "`AP-104` means that the API key has expired...",
  "passage_hash": "3987c7a38a19a24cdf88aaf57e411700a01fc5fdb7398bd899215bd3706defc4"
}
```

Evidence envelope MUST include:
- `schema: "annpack-evidence-v1"`
- `pack_root: "7fb855794ac5bbe4049947fd2421c44acd51ba6495e2db95b18995ac36db119b"`
- `passage_id` matching the passage
- `passage_hash` matching the BLAKE3 hash of the decoded passage JSON

### Query: "cache rotation" (lexical mode)

Top result MUST be from `rotation.md`, heading "Key rotation".

## Invalid corpus — all MUST be rejected

Each file in `invalid-corpus/` MUST cause the reader to return an error and NOT
return results or crash with an unhandled exception:

| File | Expected rejection reason |
|---|---|
| `empty.annpack` | File too short for header |
| `magic-only.annpack` | File too short for header |
| `wrong-magic.annpack` | Bad magic bytes |
| `wrong-version.annpack` | Unsupported format version |
| `truncated-at-header.annpack` | Directory range exceeds file |
| `directory-bit-flip.annpack` | Root hash mismatch |
| `section-hash-mismatch.annpack` | Section BLAKE3 hash mismatch |
| `reserved-header-set.annpack` | Reserved header bytes nonzero |

## Required behaviors (from spec/SECURITY.md)

The reader MUST:

1. Check every addition and multiplication for overflow
2. Reject sections outside the source, overlapping sections, duplicate IDs
3. Reject noncanonical directory order (section IDs must be strictly increasing)
4. Reject nonzero reserved bytes in header and directory entries
5. Enforce section count limit (≤ 16,384)
6. Enforce manifest size limit (≤ 4 MiB)
7. Enforce per-section size limit (≤ 64 GiB)
8. Enforce decompression ratio limit (256× stored size, floor 16 MiB)
9. Verify directory-root binding BEFORE interpreting any section
10. Verify section BLAKE3 hash BEFORE decoding section payload
11. Terminate varint decoding after at most 10 bytes
12. Reject non-terminating varints

## Conformance runner

After implementing the reader, run this conformance check. Produce a JSON report:

```json
{
  "implementation": "your-language/your-name",
  "commit": "annpack-commit-hash-used-for-golden",
  "golden_root_computed": "...",
  "golden_root_matches": true,
  "golden_search_ap104_first_passage_id": "...",
  "golden_search_ap104_correct": true,
  "invalid_corpus_results": {
    "empty.annpack": "error: ...",
    "wrong-magic.annpack": "error: ..."
  },
  "all_invalid_rejected": true,
  "signed_root_unchanged": true,
  "lines_of_code": 450
}
```

## What counts as independent

- Written from the specifications above, not from the Rust source
- Not a translation of any Rust function or structure
- Not produced in this agent session (use the M3 in a fresh session with a clean context)
- Standard library functions for JSON, BLAKE3, Ed25519, zlib-decompress do not count toward the LOC budget

## What to submit

- Source code file(s)
- Conformance report JSON
- Session transcript showing the fresh context (no prior knowledge of implementation)
