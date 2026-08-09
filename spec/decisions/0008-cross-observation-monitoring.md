# ADR-0008: Equivocation is detected by comparing observations, stored as an append-only local file

Status: accepted, 2026-08-09.

Wire format and CLI contract are normative in [RELEASE-v1](../RELEASE-v1.md)
§10. This record holds the decision and the reasoning.

## Context

[ADR-0007](0007-transparency-log-integration.md) closed one half of a gap
`policy.rs` had named since the policy engine was first built: a verified
channel-state statement proves who signed it, but not that the signing was
public or singular. Transparency evidence (§7.1) answers the "public" half —
a Sigsum proof shows a statement's digest was logged and witnessed. It does
not, and was explicitly documented as not, answering the "singular" half: a
statement can be genuinely logged and witnessed and still be one half of an
equivocating publisher's story, if a different, equally genuine statement was
shown to someone else and never compared against this one.

Comparing requires more than one observation. Everything built through Step
9a is stateless with respect to history — one artifact, one statement, one
proof, checked once. Detecting equivocation across independent sightings is a
different kind of problem: it needs a durable record of what has been seen,
which nothing in the codebase had before this.

## Decision

### Storage: a local, operator-owned, append-only file

The observation history is a JSON Lines file the operator controls directly
— `annpack release observe` appends to it, `annpack release monitor` reads
it. No database, no shared service, no ANNPack-run infrastructure. This
mirrors `RetainedState`'s existing shape (ADR-0004): a single file, read and
written by the operator's own tooling, with ANNPack providing the format and
the comparison logic but not the storage or transport. A fleet-wide,
multi-observer view is deliberately not this record's concern — that is
Step 10's control-plane layer, and pluggable storage can be added there if a
concrete need for it materializes, rather than speculatively generalizing a
single-operator tool now.

Line-oriented, not a single JSON array, because a history is meant to grow
by appending, and a truncated last write (a crashed process, a disk full
mid-write) should corrupt at most one line, not the entire file. `annpack
release observe` never rewrites earlier lines and never deduplicates: two
identical observations are two data points that really did see the same
statement twice, and collapsing them would be this module deciding what
counts as corroboration instead of honestly reporting what was fed to it.

### What counts as an incident: six conditions, not a single verdict

Consistent with this project's standing rule that verification stages stay
separate and are never merged into one boolean (`policy.rs`'s own
documentation states this as the rule most likely to be violated by
convenience), `annpack release monitor` reports six distinct incident kinds
rather than a single pass/fail:

- **`equivocation`** — the direct case: same sequence, different statement
  digest.
- **`conflict`** — more than one artifact root that nothing at a higher
  sequence ever explicitly superseded. Distinct from equivocation: two
  statements at *different* sequences can each claim to be current with no
  chain between them, which is not literally "the same sequence signed
  twice" but is the same underlying problem — an observer cannot tell which
  root is authoritative.
- **`authority_violation`** — a statement whose signatures met no authorised
  role's threshold, observed as though it were real. Reuses
  `release::verify_channel_state`'s existing authority computation rather
  than re-deriving it; there is exactly one place role-threshold logic lives.
- **`sequence_gap`** — a gap in observed sequence numbers. Named
  separately from the other five because it is not, by itself, evidence of
  misbehaviour: it is evidence that this monitor's view of history is
  incomplete, which is a fact worth surfacing on its own rather than folding
  into a false negative for the other checks.
- **`stale_local_state`** — the observed history contains an authorised,
  higher-sequence statement than a supplied retained-state file reflects.
  Checked against `RetainedState` (ADR-0004) directly; only the observation
  group matching that retained state's own scope is checked, and groups with
  no matching retained state simply get no check for this condition, the
  same "absent input, no check performed" shape used throughout this
  codebase rather than an error.
- **`revoked_root_advertised`** — a statement revoked a root; a
  later-or-equal-sequence statement still shows that root as current. The
  recommended semantic, chosen deliberately over "any observation of the
  root after the revocation, regardless of sequence": an earlier, already-
  superseded observation that merely predates the revocation is completely
  ordinary history, not an incident, and flagging it would make every clean
  history noisy.

### Comparison logic reuses existing verification, never re-derives it

`authority_violation` and `stale_local_state` both call
`release::verify_channel_state` per observation rather than re-implementing
role-threshold or signature checking. This is the same discipline
[ADR-0007](0007-transparency-log-integration.md) applied to Sigsum
verification and [ADR-0006](0006-build-provenance-envelope.md) applied to
provenance binding checks: a second implementation of "is this signature
from an authorised role" would be a second place for that logic to drift
from the first.

## What this does not claim

A monitor reports only on what it was shown. It does not fetch statements
from anywhere, and it does not claim completeness — a monitor that has only
ever observed one side of an equivocating publisher's story reports zero
incidents, correctly, because none is visible to it. `sequence_gap` exists
specifically so an incomplete view is visible as a distinct, honest signal
rather than silently indistinguishable from a genuinely clean history. This
mirrors §1's standing rule for the whole freshness model: absence of
evidence is not evidence of absence, and nothing here is permitted to read
"no incident found" as "no incident exists."

A verified transparency proof (§7.1) and a clean monitor report answer
different questions and neither substitutes for the other: transparency
evidence is about one statement's public visibility, monitoring is about
consistency across everything actually observed. A statement can be
witnessed and still equivocating, if the conflicting statement exists and
this monitor simply has not seen it yet.

## Alternatives rejected

**A pluggable storage backend from the start.** Considered and explicitly
deferred, not rejected outright — see "Storage" above. Building the
abstraction before a second real backend exists to justify it would be
speculative generality this project has consistently avoided elsewhere (the
same reasoning ADR-0006 gave for not inventing a bespoke provenance envelope
before DSSE's fit was clear).

**Folding all six conditions into a single boolean "equivocation detected"
verdict.** Rejected for the same reason `policy.rs` reports five separate
claims instead of one: a caller who only learns "something is wrong" cannot
distinguish an incomplete view (`sequence_gap`) from an active attack
(`equivocation`, `revoked_root_advertised`), and conflating them would either
make normal incomplete histories alarming or make real incidents easy to
miss under low-severity noise.
