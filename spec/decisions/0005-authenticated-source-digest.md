# ADR-0005: Every artifact commits to the source bytes it was built from

Status: accepted, 2026-08-06. Introduced by manifest section format 4 in
v0.7.0-rc1.

## Context

The builder already computes a deterministic digest over the exact paths and
bytes it consumes, for every input format, at `ingest.rs`. That digest was placed
into authenticated artifact metadata only when the input format was OKF:

```rust
source: (corpus.input_format == InputFormat::Okf).then(|| SourceDescriptor { … })
```

For Markdown and MDX — the dominant input path — the digest existed in the build
report and nowhere in the artifact. Nothing the artifact commits to referred to
its own inputs.

The asymmetry had no design rationale. A source-tree digest is a property of the
build inputs, not a feature of one corpus format. It was authenticated for OKF
because OKF was where a source descriptor was first needed, and the condition was
never revisited.

### Why this surfaced now

Build provenance binds a source revision and tree digest to an artifact root. The
strength of that binding depends entirely on whether the artifact independently
commits to the same digest:

- **Authenticated.** The verifier recomputes the artifact root, reads the source
  digest from inside it, and compares it against the provenance. Disagreement is
  detectable.
- **Builder-carried only.** The verifier can confirm the builder *said* the
  digest, and nothing more. A builder that signs a false digest is
  indistinguishable from one that signs a true one.

Shipping provenance against the existing format would have made the claim strong
for OKF and weak for Markdown, and would have frozen that weakness into
`PROVENANCE-v1` — a format-dependent security property, silently weaker on the
majority path.

## Decision

**Every newly built artifact authenticates a source descriptor naming the input
format and a digest over the exact consumed bytes, regardless of input format.**

This is an intentional root-changing revision, recorded as one. Manifest section
format 4 requires the descriptor; formats 1–3 do not and never will.

### What the descriptor does and does not assert

`source.digest` commits to the bytes the builder consumed. It says nothing about
where they came from.

`source_revision` remains caller-supplied contextual metadata: a commit, a tag,
or any string the operator chose. The digest does not prove it. A builder can
record `git:deadbeef` alongside bytes that were never in that commit, and the
artifact cannot tell. Establishing that correspondence is the job of external
build provenance, signed by a workflow identity, and deliberately not of the
artifact.

No caller identity, repository name or workflow field enters the artifact. Those
are claims about the world outside the build, and an artifact cannot authenticate
them by containing them.

### Generalising rather than duplicating

The existing `SourceDescriptor` is reused, not paralleled. `format` becomes the
resolved input format (`markdown` or `okf`, never `auto`), and `version` stays
optional and OKF-specific, since Markdown has no corpus-format version to state.
A second source-digest representation would create two things to keep in
agreement and one place for them to diverge.

The digest is computed once, in ingestion, and both the artifact and
`build --json` read that single value. Two independent computations would be two
opportunities to disagree, and a test asserting they match is weaker than not
having two.

## Consequences

**Every newly emitted artifact root changes**, for every input format. The
manifest gains a field, the manifest is committed by the content root, so the
root moves. This is the cost and it is being paid deliberately.

Previously published artifacts keep their original roots and remain readable.
Nothing is rewritten: no published tag moves, no released artifact is
regenerated, and readers continue to compute old roots exactly as before.

Old readers reject format 4 explicitly at the container boundary rather than
misparsing it, because `SUPPORTED_MANIFEST_FORMAT_VERSIONS` is checked in
`PackReader::open` before any field is interpreted. That boundary is the one
v0.3.1 lacked, and it is what makes this bump safe rather than silent.

Absence of the descriptor in a format ≤3 artifact is legitimate history and is
reported as such, never as corruption. Absence in a format 4 artifact is a format
error.

Repository-owned fixtures, the golden artifact and the demo packs are regenerated
as outputs of the new release candidate. Each moved pin has exactly one cause.

## Alternatives rejected

**Leave Markdown and MDX externally bound only.** Rejected: it makes provenance
strength depend on input format, weakest on the dominant path, and bakes that
into the provenance specification. The economy is in the wrong place.

**An unauthenticated sidecar field.** Rejected: it cannot establish agreement
between a digest and an artifact, which is the entire property being sought. It
would let a verifier report a comparison it never made.

**A separate authenticated section, to avoid touching the manifest.** Rejected on
its own terms: any authenticated section is committed by the section directory
and therefore changes the artifact root too. It buys no root stability and costs
a second source-metadata representation.

**Emit the field under format 3.** Rejected: changing emitted bytes without
changing the version number is precisely the v0.3.1 mistake, which broke old
readers one-way and is why the version boundary exists.
