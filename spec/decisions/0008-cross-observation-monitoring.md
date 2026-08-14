# ADR-0008: Equivocation is detected by comparing observations, stored as an append-only local file

Status: accepted, 2026-08-09.

Wire format and CLI contract are normative in [RELEASE-v1](../RELEASE-v1.md)
§10. This record holds the decision and the reasoning.

## Context

[ADR-0007](0007-transparency-log-integration.md) verifies that a channel-state
statement's digest was logged and witnessed. A logged statement may still
conflict with a statement shown to another observer.

Cross-observation equivocation detection requires durable history. Earlier
verification processes one artifact, statement, and proof without retaining
observations.

## Decision

### Storage: a local, operator-owned, append-only file

The observation history is an operator-controlled JSON Lines file.
`adyar release observe` appends to it; `adyar release monitor` reads it.
Adyar defines the format and comparison logic but does not provide storage or
transport infrastructure. The local-file model follows retained state in
ADR-0004. Fleet-wide storage is outside this decision and may be added when
another backend is required in Step 10's control-plane layer.

The line-oriented format limits a truncated append to one line. `adyar release
observe` does not rewrite or deduplicate entries. Identical entries remain
separate observations.

### Incident conditions

`adyar release monitor` reports six incident kinds separately, consistent with
the stage reporting in `policy.rs`:

- **`equivocation`** — the direct case: same sequence, different statement
  digest.
- **`conflict`** — more than one artifact root that no statement at a higher
  sequence explicitly superseded. This can arise from statements at different
  sequences.
- **`authority_violation`** — a statement whose signatures met no authorised
  role's threshold. The check uses `release::verify_channel_state`.
- **`sequence_gap`** — a gap in observed sequence numbers. It indicates an
  incomplete observation history, not necessarily misbehaviour.
- **`stale_local_state`** — the observed history contains an authorised,
  higher-sequence statement than a supplied retained-state file reflects. Only
  the observation group matching the `RetainedState` scope is checked. Other
  groups receive no check for this condition.
- **`revoked_root_advertised`** — a statement revoked a root; a
  later-or-equal-sequence statement still shows that root as current. An earlier
  observation that predates the revocation is not an incident.

### Verification reuse

`authority_violation` and `stale_local_state` call
`release::verify_channel_state` for each observation. Role-threshold and
signature checks are not reimplemented. The same reuse boundary applies to
Sigsum verification ([ADR-0007](0007-transparency-log-integration.md)) and
provenance binding ([ADR-0006](0006-build-provenance-envelope.md)).

## Limitations

A monitor reports only on supplied observations and does not fetch statements.
If it observes only one side of an equivocation, it reports no incident.
`sequence_gap` reports one form of incomplete history. A report with no incidents
does not establish that no incident exists.

A verified transparency proof (§7.1) establishes public visibility for one
statement. Monitoring checks consistency among supplied observations. A
witnessed statement may conflict with an unobserved statement.

## Alternatives rejected

**A pluggable storage backend from the start.** Deferred until a second backend
is required, consistent with the dependency decision in ADR-0006.

**Folding all six conditions into a single boolean "equivocation detected"
verdict.** A boolean cannot distinguish an incomplete view (`sequence_gap`)
from an active attack (`equivocation`, `revoked_root_advertised`).
