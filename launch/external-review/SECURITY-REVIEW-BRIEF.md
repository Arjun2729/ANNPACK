# Brief: independent security review of the ANNPack format and parser

**Deliverable:** a written security review of the ANNPack v3 wire format and its
reference parser, published in full including anything we disagree with.

**Budget guide:** $15,000–30,000. **Timeline:** 4 weeks from start.

---

## Threat model

An ANNPack artifact is **untrusted binary input even when it arrives from a
familiar domain**. The realistic attacker publishes or tampers with a pack and
gets a victim to open it in a CLI, an MCP server inside an agent loop, or a
browser tab. Memory-safety, resource-exhaustion, and provenance-forgery outcomes
all matter; so does anything that lets a pack misrepresent its own provenance.

The reference implementation is Rust (`unsafe`-free), so the interesting failures
are logic errors, resource exhaustion, and specification ambiguities that make a
*conformant* implementation insecure — not buffer overflows.

## Scope

| Area | Files |
|---|---|
| Container parse, bounds, root binding | `rust/src/format.rs`, `rust/src/reader.rs` |
| Retrieval, overlay validation, profile selection | `rust/src/search.rs`, `rust/src/conformance.rs` |
| Signatures | `rust/src/signing.rs` |
| Evidence receipts | `rust/src/evidence.rs`, `spec/EVIDENCE-v1.md` |
| Delta codec | `rust/src/delta.rs` |
| OCI registry client | `rust/src/oci.rs` |
| Browser runtime | `web/annpack-browser.js` |
| Normative claims | `spec/FORMAT-v3.md`, `spec/SECURITY.md` |

Out of scope: the withdrawn retrieval evaluation, `attic/`, and retrieval
*quality*.

## Questions we most want answered

1. **Can a pack forge provenance?** Can any artifact produce a valid-looking
   evidence envelope or `annpack-receipt-v1` for a passage it does not contain,
   or bind a passage to the wrong artifact root or source revision?
2. **Is the Merkle construction sound?** Leaves use
   `BLAKE3("ANNPACK3-PASSAGE-EVIDENCE\0" || record)`, interior nodes use
   `BLAKE3("ANNPACK3-EVIDENCE-NODE\0" || left || right)`, and odd levels
   **promote** rather than duplicate. We believe promotion avoids the classic
   duplicate-leaf collision and the separate domain separators prevent
   leaf/node confusion. Please try to break both claims. Second-preimage and
   cross-tree collisions are the specific worry.
3. **Can a derived section influence citable output?** ANN-7/ANN-8 overlays are
   supposed to affect ranking only and never contribute citable text or change a
   `passage_hash`.
4. **Can a malformed optional descriptor affect Core?** v0.4.0 separates
   `core_conformant` from `extensions_conformant` specifically to prevent this.
   Try to reach the default lexical path from an invalid ANN-10 descriptor.
5. **Resource exhaustion.** Decompression bombs, allocation amplification, and
   quadratic behaviour in the delta matcher, IVF selection, or overlay loading.
   Note the per-section limit is 64 GiB, which is safe-but-OOM on small hosts.
6. **Is `SECURITY.md` sufficient to implement securely?** Our own clean-room
   reader checked the declared decompression ratio and then inflated without an
   output bound. If the specification permits that reading, it is a spec defect,
   not just an implementation bug.
7. **Registry and range clients.** Credential handling, redirect-to-foreign-origin
   behaviour, bearer challenge parsing, and range-response validation.

## What we already know (please verify rather than rediscover)

- `sidecar_digest` is recorded provenance, **not** proof of derivation. It hashes
  the sidecar file, not the emitted section. Documented in `SECURITY.md`.
- Fuzzing: 4 targets, ~16.7B executions, 0 crashes, but `format.rs` region
  coverage is only 10.8% from `open_pack` entry points. A structure-aware
  campaign is the obvious gap; we would value your view on whether it is
  necessary before production use.
- The 64 GiB per-section limit permits a safe OOM on memory-constrained hosts.
- Policy metadata is declarative and enforces nothing after plaintext access.

## What happens to what you find

[`spec/COMPATIBILITY.md`](../../spec/COMPATIBILITY.md) is the commitment: a
format-changing finding before final release produces a new release candidate,
even if artifact roots change again, and a critical finding can withdraw a
candidate outright. Root stability never outranks correctness. Report freely; you
are not creating a versioning problem for us.

## Deliverables

1. A written report: findings with severity, reproduction, and suggested fixes.
2. Explicit answers to questions 1–7, including "no issue found" where that is
   the conclusion.
3. Any disagreement with our own characterisation of the threat model or the
   known issues above.

We will publish the report in full, including findings we have not fixed. Please
do not soften it.

## Prerequisites we have already met

- Apache-2.0, public source, reproducible build.
- `cargo test --all-features`: 97/97. Clippy `-D warnings` clean.
- A conformance packet with a one-command runner: `spec/conformance/`, 42/42.
- A corruption corpus and negative fixtures for each documented rejection rule.
- No `unsafe` in the reference implementation.
