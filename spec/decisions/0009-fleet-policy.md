# ADR-0009: Fleet policy is a signed document with its own rotation, verified locally

Status: accepted, 2026-08-09.

Wire format and CLI contract are normative in [FLEET-v1](../FLEET-v1.md).

## Context

Everything through Step 9 answers what a publisher may assert and whether one
statement or proof checks out. None of it answers what an organization
running many verifiers requires them to check. Two verifiers configured
differently can both report "verified" while enforcing different things.

## Decision

### A new document, not a new role on `TrustRoot`

`FleetPolicy` is signed by keys the organization controls, independent of any
publisher's `TrustRoot`. A publisher and the organizations consuming its
releases are different parties; folding fleet requirements into the
publisher's trust root would let a publisher's compromise also override what
its consumers require, or require every consumer to trust the same root for
both questions.

### Flat signer set, not role-separated

`TrustRoot` separates `root`/`artifact`/`release_state`/`emergency_revocation`
because those roles answer different questions about publication authority.
`FleetPolicy` answers one question — what does this organization require —
so one signer set with a threshold is enough. Rotation still requires both a
threshold of the successor's own keys and a threshold of the prior's, the
same rule `TrustRoot` uses and for the same reason (ADR-0004 §Trust-root
authority and rotation): self-only lets anyone mint a policy, prior-only lets
a compromised old key install keys nobody controls.

### `required_verification_policy` reuses `policy::TrustPolicy`

The four existing tiers (`integrity_only` through
`authorized_current_witnessed`) already express what a verification call can
require. A fleet policy names one of them rather than inventing a parallel
scale.

### Transparency and workload requirements are pinned by digest, not embedded

`required_transparency_policy_digest` and `required_workload_trust_digest`
reference an exact Sigsum policy file or workload-trust configuration by
digest, rather than embedding the log/witness/workload key lists inside
`FleetPolicy` itself. A fleet requiring a specific trusted-log configuration
still manages that configuration as its own file, reviewed and rotated on its
own schedule; `FleetPolicy` states which one is required, not what it
contains.

### Compliance is revision-and-digest equality, evaluated locally

`evaluate_compliance` takes both the locally configured policy and the
required one as arguments and re-verifies each independently. An unverified
input reports `unavailable`, never `compliant`: a caller cannot make a
document count as compliant by asserting elsewhere that it was checked.
Matching revision numbers alone is not sufficient — the digest comparison
catches two documents that share a revision number but differ in content, a
case revision equality alone would miss.

## What this does not do

Fetch a required policy from anywhere. `evaluate_compliance` takes both
inputs as arguments; obtaining the required one from a control plane is
Step 10b.

Enforce `allowed_publishers`, `allowed_scopes`, or `deny_on_incident_kinds`
against a specific verification or monitor result. This record defines the
object and its own verification; wiring those fields against `verify
--policy` and `release monitor` output is separate, later work.

## Alternatives rejected

**A role inside `TrustRoot`.** Rejected: publisher trust and fleet
requirements are different parties' decisions with different blast radii,
the same reasoning ADR-0004 applied to separating `release_state` from
`emergency_revocation`.

**Embedding transparency and workload-trust configuration directly in
`FleetPolicy`.** Rejected: those configurations already have their own
lifecycle (a Sigsum policy file, a workload-trust list) and reviewing them
should not require re-signing a fleet policy that happens to reference them.
A digest binds the requirement without duplicating the content.

**Fetching required policy as part of this record.** Deferred: a network
fetch is a control-plane concern (Step 10b) with its own trust and
availability questions. Building the object first, and proving it verifies
and evaluates correctly against locally supplied files, keeps this record's
own scope checkable.
