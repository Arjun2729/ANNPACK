# ADR-0001: Freeze Core and number optional extensions

Status: accepted, 2026-07-17.

## Decision

Freeze `annpack-core-v1.0-draft` around the smallest useful read-only contract and move implemented optional behavior into independently conformant numbered extensions. Do not assign contracts to unimplemented roadmap ideas.

Core contains the container, authoritative content/passages, citations, BM25, range access, BLAKE3 integrity, Ed25519 signatures, evidence envelopes, and well-known discovery. Vector retrieval, deltas, OCI, policy commerce metadata, and dependencies are ANN-1, ANN-2, ANN-3, ANN-5, and ANN-6 respectively. ANN-4 remains unassigned.

## Consequences

- A Core-only reader is fully conformant.
- Unknown optional sections remain safely ignorable.
- The target for a read-only Core client is roughly 500 lines excluding standard libraries.
- Existing `ANNPACK3` bytes remain valid; Core v1 is a conformance profile over wire format v3, not a gratuitous wire renumbering.
- Removing `-draft` is blocked on a genuinely independent reader.
