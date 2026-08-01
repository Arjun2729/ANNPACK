# Contributing to ANNPack

ANNPack is a candidate format and a reference implementation. Bug reports,
security findings, conformance disagreements, interoperability results, and
independent implementations are welcome.

## Baseline Rust checks

Run these before opening a pull request:

```bash
cargo fmt --all -- --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-targets --all-features
```

These are baseline checks, not the entire matrix. CI also runs release builds,
retrieval-harness checks, language bindings, framework integrations, benchmarks,
range-transfer gates, the Core conformance packet, browser/WASM smokes,
generated-site drift checks, and the same-builder determinism matrix. The
complete contract is in [`.github/workflows/ci.yml`](.github/workflows/ci.yml).

## Reporting a bug

Include the ANNPack commit or tag, the exact command, the artifact or fixture,
and the observed versus expected result. For a parsing or verification bug,
attach the smallest input that reproduces it.

## Reporting conformance or interoperability findings

Test independent readers against `spec/conformance/`. Include the ANNPack commit
or tag, your implementation and toolchain version, the vector or artifact, and
the observed versus expected result. A disagreement between two readers is a
useful report; please file it rather than working around it.

## Specification disagreements

The specification is normative. When the specification and the reference
implementation disagree, first determine whether the specification is complete
and unambiguous. Fix the implementation when the contract is clear; report and
resolve a specification defect when it is not. Do not encode new wire semantics
in the reference implementation alone.

## Portable public surfaces

Executable scripts and current generated reports must not contain personal
checkout paths or machine-specific binary locations. Derive the repository root
from the script location, accept explicit environment variables for out-of-tree
inputs, and label machine-specific measurements as dated historical evidence
rather than publishing them as `latest`.

## Scope

The protocol is under feature freeze. Defect fixes, security hardening,
compatibility corrections, conformance work, and interoperability work are in
scope. New rankers, extensions, models, and speculative wire features are not.

## Security-sensitive changes

A pull request that changes parsing, verification, signatures, receipts,
resource bounds, compatibility, or a wire contract should not merge while a
valid high-severity finding remains unresolved. Green CI proves the tested cases
passed; it does not replace adversarial review.

Security and wire-format pull requests should state:

- the threat or compatibility failure being addressed;
- all changed schemas and roots;
- resource bounds and failure behavior;
- isolated regression tests for each protected field or invariant;
- migration and supersession consequences.
