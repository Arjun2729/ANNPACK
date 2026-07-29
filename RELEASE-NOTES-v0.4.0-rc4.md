`v0.4.0-rc4` hardens the standalone evidence-receipt verifier against attacker-controlled resource exhaustion and restores conformance for uncompressed Documents sections.

## Why rc3 is superseded

`v0.4.0-rc3` correctly bound receipt labels and `canonical_url` to authenticated artifact bytes, but its reference verifier had two defects:

1. It trusted the carried directory's `logical_length` before applying ANNPack's section-size and decompression-ratio limits. An untrusted receipt could therefore request excessive allocation before signature rejection.
2. It always attempted zlib inflation even though FORMAT-v3 permits codec 0 for an uncompressed section.

Do not use the rc3 verifier on receipts from untrusted parties. Packs and artifact roots remain valid; the defect is in receipt verification.

## What rc4 changes

- Accepts only the explicitly supported `annpack-receipt-v2` schema; unknown schemas fail rather than receiving accidental partial semantics.
- Caps proof length and base64-decoded receipt components before expensive work.
- Validates directory alignment, ordering, reserved bytes, duplicate prevention and section-size limits.
- Validates Manifest type, codec, stored/logical lengths and stored hash.
- Validates Documents stored length and hash before decoding.
- Handles codec 0 as exact raw bytes and codec 1 as bounded zlib.
- Applies `MAX_SECTION_SIZE` and the reference 256:1 expansion limit above 16 MiB before allocation.
- Rejects unsupported codecs and all exact-length mismatches.
- Splits passage ID and ordinal tampering into independent regressions.
- Adds codec, ratio, schema and proof-bound tests.

## Compatibility and roots

The receipt schema remains `annpack-receipt-v2`. Existing honest rc3-issued v2 receipts remain valid under rc4.

The pack container, passage records, logical content root and artifact-root computation are unchanged. Existing packs do not need rebuilding. The golden-root and same-builder determinism matrix must pass on the exact release commit before tagging.

## Receipt size

A v2 receipt contains a compact passage proof plus the artifact's stored Documents catalogue when it authenticates `canonical_url`. Receipt size therefore depends on document metadata and is not universally 2–5 KB. Measure it on the target corpus.

## Security claims

Rc4 addresses the known resource-bound and codec-handling defects. It does not claim an independent external security review, publisher identity without an external trusted-key binding, rollback resistance, or answer faithfulness.

## Migration

- **Pack publishers:** no pack rebuild expected.
- **Receipt issuers:** no schema migration expected; reissuing receipts is optional.
- **Receipt verifiers:** upgrade from rc3 before accepting untrusted receipts.
- **Consumers:** continue to treat signature validity, publisher identity and currency as separate claims.
