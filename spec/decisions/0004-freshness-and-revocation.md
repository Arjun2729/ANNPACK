# ADR-0004: Release authorization is time-indexed and lives outside the artifact

Status: accepted and implemented, 2026-08-06. Supersedes the offline-first design
recorded under this number on 2026-07-27.

Wire formats, field names, verification order and limits are normative in
[RELEASE-v1](../RELEASE-v1.md). This record holds the decision and the reasoning.

## Context

An Adyar artifact is immutable and its receipts are permanent. A receipt for a
superseded artifact continues to verify offline. It does not state whether the
publisher still stands behind the artifact.

The architecture separates:

- **Historical validity.** Did these bytes exist, unmodified, in this artifact,
  at this source revision? Answerable offline, permanently, from the artifact and
  a receipt.
- **Release authorization.** Is this artifact the one a publisher currently
  stands behind for a named corpus and channel? This claim is time-indexed and
  cannot be answered from the artifact alone.

Treating currency as intrinsic makes a stale artifact look authoritative.
Expiring historical validity makes past evidence unavailable for audits.

### Why the previous design was replaced

The 2026-07-27 version of this ADR made three decisions this one reverses.

It **centred offline operation** and set a one-week statement validity for
disconnected consumers. An agent performing retrieval is normally connected,
and the long validity window lets an adversary serve superseded knowledge as
current for a week. The relevant deployment property is whether the consumer can
reach a trusted release authority at decision time.

It **had one publisher key sign both artifacts and freshness statements**,
reasoning that trusting a publisher should be one decision rather than two. That
makes compromise of the build pipeline sufficient to declare any artifact
current, and makes revocation impossible to delegate to a differently protected
key.

It **recorded rollback state as a highest-root marker**. Roots are unordered and
unscoped, so the marker cannot detect replay or isolate channel state. Sequence
orders statements; a digest detects conflicting statements at one sequence.

## Decision

**Release authorization is asserted by a publisher, scoped to a corpus and
channel, signed by a role-separated key, sequenced, time-bounded, and
distributed separately from the artifact it describes.**

### Artifacts stay silent about their own currency

A pack cannot revoke itself. An attacker serving an old artifact serves the old,
unrevoked bytes, so any in-pack field is unauthenticated with respect to its own
supersession. The artifact format is unchanged by this decision: no section, no
manifest field, no change to either content root.

### Publisher trust roots with separated roles

A trust root states which keys may act for a publisher, in which role, at what
threshold. Four roles are required:

| Role | May |
|---|---|
| `root` | sign trust roots and authorise successors |
| `artifact` | sign artifacts |
| `release_state` | select the current artifact for a channel |
| `emergency_revocation` | withdraw an artifact |

A key that signs artifacts does not decide which artifact is current or revoke
artifacts. Combining the roles would give one key all four powers.

**A revocation key may withdraw and may not promote.** The architecture permits a
statement to be signed by either `release_state` or `emergency_revocation`.
Honouring both equally would hand the revocation key the authority the split
exists to withhold, so they are honoured asymmetrically: a statement signed only
by `emergency_revocation` has its revocations acted on and its current-release
claim ignored. Taking something out of service and putting something into service
are different powers.

### Trust-root authority and rotation

A trust root is signed by its own `root` role. A successor must additionally be
signed by a threshold of the **prior** root's `root` role and must advance the
version. Requiring both signatures blocks two distinct attacks: prior-only would
let a compromised old key install keys nobody controls, and self-only would let
anyone mint a root and present it. Equal version is refused as well as lower —
two differently keyed roots at one version is publisher equivocation at the trust
layer.

First contact is the irreducible exception. A consumer with no prior root accepts
one on trust; nothing in the object distinguishes it from an attacker's. This is
reported, not hidden.

### Scope is publisher, corpus and channel

A statement authorises a release for exactly one scope. The scope a consumer
compares against must be established **outside** the statement — publisher from
the trusted root, corpus and channel from configuration. A statement that
supplies its own expectations is only ever compared against itself.

The first implementation of the reference CLI passed
the statement's own fields as the expected scope, so the check was tautological
and a `staging` statement verified cleanly for a consumer asking about
`production`. The library check was real and tested; the test called the library
directly, so it passed while the shipped binary had no path that could reject
cross-channel replay.

### Monotonic sequence and statement digest

Retained state is keyed on the expected scope and records the highest accepted
sequence, that statement's digest, and the artifact root it named.

The sequence orders statements, so replay of an older one is detectable. The
digest detects the case a sequence alone cannot: two different statements at the
same sequence, which is a publisher signing conflicting claims and is a security
event no amount of valid signing excuses.

State is written temp-sync-rename. An in-place write can leave a truncated file
that reads as *no retained state* on the next start, downgrading the client to
first contact.

### Assumptions

**A trusted clock.** Expiry cannot be evaluated without one. Without a trusted
clock, validity is unknown. An attacker-controlled clock could extend a
statement indefinitely.

**Durable state.** Rollback resistance requires retained state. Agent
infrastructure is frequently ephemeral, so first contact is common. Witness
cosignature timestamps supply bounded freshness to a consumer without durable
state.

### Three profiles

| Profile | Requires | Use when |
|---|---|---|
| **Online Standard** | signed statement, trusted clock, short validity, sequence state, fail-closed on refresh failure | ordinary operation; the adversary is a stale cache or a replayed response |
| **Online Evidenced** | the above, plus log inclusion, a witness quorum, and monitoring | the freshness claim will be shown to someone who does not trust the publisher, or the consumer keeps no durable state |
| **Cached/Disconnected** | cached statement and checkpoint, bounded by expiry, explicitly weaker | no release authority is reachable at decision time |

Online Standard rejects stale-cache and replay attacks. It does not detect a
publisher signing conflicting statements for different consumers.

Online Evidenced supports third-party inspection of publisher claims and
stateless consumers. Witness timestamps substitute for durable state in the
latter case.

### Transparency is consumed, not built

Adyar operates no log. The evidenced profile verifies inclusion proofs and
witness cosignatures from an external C2SP/Sigsum-compatible log. Operating
logs, running witnesses, key management and gossip are out of scope.

**Inclusion does not prove currency.** It cannot reveal a withheld newer
statement. Monitors detect equal-sequence conflicts, multiple current roots,
unauthorised role keys, sequence gaps, and revoked roots still advertised as
current after publication.

### Four verdicts, and `unknown` is not `current`

`current`, `superseded`, `revoked`, `unknown`. A statement that does not mention
an artifact yields `unknown`. `superseded` is policy; `revoked` is a security
event.

### Verification facts stay separate from acceptance decisions

`artifact_integrity`, `publisher_authority`, `currency` and the policy decision
are reported independently and never merged. A revoked artifact that is
genuinely authentic reports valid integrity, authorized publisher, revoked
currency, and a denied decision. **Revocation never changes the integrity fact.**

Known revocation denies under every policy, including the weakest. A consumer
told an artifact was withdrawn should not use it because it happened to ask a
smaller question.

## Consequences

- Receipts and artifacts remain historically verifiable after supersession or
  revocation, permanently and offline. Superseding a release does not
  retroactively invalidate evidence about what an agent read.
- Compromise is scoped by role rather than by publisher.
- A publisher who stops issuing statements downgrades every consumer to
  `unknown`, which must be reported.
- Rollback resistance is unavailable at first contact and after state loss.
- Statement validity is measured in hours, not a week. Short windows require
  reachable infrastructure.

## Non-goals

Operating a transparency log or witness network. Identity infrastructure beyond
key roles. A TUF-equivalent general update protocol. Passage-level access
control. Proving that no newer statement exists — no offline mechanism can, and
none is claimed.

## Alternatives rejected

**A revocation field inside the pack.** Revoking artifact *X* requires a
statement not carried by *X*, or an attacker serves the un-revoked copy.

**One publisher key for artifacts and release state.** Simpler, and it makes
build-pipeline compromise sufficient to declare any artifact current.

**Short-lived signatures over artifacts.** Forces re-signing on a timer and
conflates authenticity with currency. A signature answers who made this, not
whether it is current.

**A highest-root rollback marker.** Roots are unordered and unscoped; replay and
equivocation are both undetectable with one.

**Transparency as the base layer.** Rejected because it adds an availability
requirement to every query. It remains required for the evidenced profile.
