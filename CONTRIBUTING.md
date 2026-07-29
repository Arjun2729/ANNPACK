# Contributing to ANNPack

ANNPack is an open candidate format and reference implementation. Bug reports,
security findings, conformance disagreements, interoperability results, and
independent implementations are welcome.

## Commit attribution

Do not add AI-tool co-author trailers, generated-by footers, or automated tool
signatures. Legitimate human co-authorship may be recorded normally. Commit
messages should explain what changed and why.

## Baseline Rust checks

Run these before opening a pull request:

```bash
cargo fmt --all -- --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-targets --all-features
```

These are baseline checks, not the entire release matrix. CI also runs release
builds, retrieval-harness checks, language bindings, framework integrations,
benchmarks, range-transfer gates, the Core conformance packet, browser/WASM
smokes, generated-site drift checks, and the same-builder determinism matrix.
The complete contract is in [`.github/workflows/ci.yml`](.github/workflows/ci.yml).

## Normativity

The specification is normative. When the specification and reference
implementation disagree, determine whether the specification is complete and
unambiguous. Fix the implementation when the contract is clear; report and
resolve a specification defect when it is not. Do not silently encode new wire
semantics in the reference implementation.

Project launch checklists, release ledgers, outreach plans, and commercial
strategy are not normative protocol material.

## Scope discipline

The protocol is under feature freeze while external validation is outstanding.
Defect fixes, security hardening, compatibility corrections, conformance work,
interoperability work, and changes required by external review are in scope. New
rankers, extensions, models, and speculative wire features are not.

## Security-sensitive changes

A pull request that changes parsing, verification, signatures, receipts,
resource bounds, compatibility, or a wire contract must not merge while a valid
P1 or P2 finding remains unresolved. Green CI proves the tested cases passed; it
does not replace adversarial review.

Security and wire-format pull requests should state:

- the threat or compatibility failure being addressed;
- all changed schemas and roots;
- resource bounds and failure behavior;
- isolated regression tests for each protected field or invariant;
- migration and supersession consequences.

## Reporting conformance findings

Test independent readers against `spec/conformance/`. Include the exact ANNPack
commit or tag, your implementation and toolchain version, the vector or artifact,
and the observed versus expected result. A disagreement is a high-value report,
not something to hide behind reference-implementation behavior.
