Security fix for evidence receipts. **Artifact roots are unchanged from rc2** — no pack needs rebuilding, and the published OKF reproduction roots still hold.

## The defect

Through rc2, a receipt proved that the *passage bytes* were present in the named artifact: passage record → Merkle inclusion path → logical content root → manifest → directory → artifact root. That chain was sound.

It did not prove the *labels attached to those bytes*. `canonical_url`, `source_revision`, `pack`, `passage_id`, and `passage_ordinal` rode along outside the authenticated set.

So a genuine, correctly-signed, correctly-verifying receipt could have its `canonical_url` repointed at an unrelated page, or its `source_revision` changed, and it would still verify. The proof was real; the citation attached to it was not — which is the claim the format exists to make.

This is not remotely exploitable and needs an attacker who can hand you a receipt. That is precisely the receipt's threat model: it is designed to be passed between parties who do not trust each other.

## The fix

- Receipt schema is now `annpack-receipt-v2`, carrying `documents_section_id` and `documents_bytes_b64`.
- **New verification step 6** — the receipt's `passage_id`, `passage_ordinal`, `source_revision`, and `pack` must equal the corresponding fields of the authenticated passage record and manifest. Labels that disagree with the bytes they name no longer verify.
- **New verification step 7** — `canonical_url` is authenticated through the Documents section, whose stored bytes hash to a directory entry that `pack_root` already commits. The document matching the passage record's `document_id` must reproduce the declared URL.
- **Fail closed.** A `canonical_url` with no Documents section to authenticate it MUST fail, so stripping the section cannot silently downgrade a URL claim to unverified.
- Integrity is now established by steps 1–7; the signature remains a separate claim at step 8. The three claims — integrity, authenticity, identity trust — are still never merged, and a valid signature still never establishes publisher identity.
- `verify-evidence` reports four new independent lines: `passage metadata matches`, `source revision matches`, `pack matches`, `canonical url matches`.
- A `-logical` receipt from a non-ANNPack issuer MUST NOT carry a `canonical_url` it cannot authenticate.

Specification: [EVIDENCE-v1](https://github.com/Arjun2729/annpackv2/blob/v0.4.0-rc3/spec/EVIDENCE-v1.md).

## Migration

**Packs: nothing to do.** Artifact roots are byte-identical to rc2, confirmed by the golden test. `launch/google-okf/expected-roots.json` and the three published OKF reproduction roots are unaffected.

**Receipts: re-issue them.** An rc2-issued receipt that carries a `canonical_url` will not verify under an rc3 verifier. That is the downgrade defense working as designed, not a regression. Re-run `annpack receipt` against the same pack — the pack itself has not changed.

The conformance vector `spec/conformance/vectors/evidence.json` is updated to the v2 shape.

## Verification

`cargo test --workspace` — **106 passed, 0 failed**. Clippy clean. CI green across the native, WASM, and determinism matrix.

Six new receipt-hardening tests in `tests/receipt_tamper.rs`: one honest
control and five tamper cases. The passage-metadata case mutates both
`passage_id` and `passage_ordinal` together.

```
honest_receipt_verifies_and_binds_every_field
forged_canonical_url_fails
forged_source_revision_fails
forged_pack_identity_fails
forged_passage_metadata_fails
dropping_the_documents_section_cannot_downgrade_a_url_claim
```

## Status

Release candidate, not a final release. Unchanged from rc2: this is a candidate specification plus reference implementation, not an independently adopted standard. There is still no independent second Core reader, no external security review, and no supported retrieval-quality claim. Format defects reported during the RC period are expected to produce another candidate — see [COMPATIBILITY.md](https://github.com/Arjun2729/annpackv2/blob/v0.4.0-rc3/spec/COMPATIBILITY.md) and the Limits section of the README.

## Install

```bash
cargo install --git https://github.com/Arjun2729/annpackv2 --tag v0.4.0-rc3 annpack
```

Rust 1.88 or newer. Prebuilt binaries are attached below.
