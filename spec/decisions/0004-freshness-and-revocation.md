# ADR-0004: Release authorization is time-indexed and lives outside the artifact

Status: accepted and implemented, 2026-08-06. Supersedes the offline-first design
recorded under this number on 2026-07-27.

Wire formats, field names, verification order and limits are normative in
[RELEASE-v1](../RELEASE-v1.md). This record holds the decision and the reasoning.

## Context

An ANNPack artifact is immutable and its receipts are permanent. That is correct
and it creates a gap: a receipt for a superseded artifact verifies perfectly,
offline, forever. Nothing in it says the knowledge is still the knowledge the
publisher stands behind, and nothing in it can, because it is a statement about
the past.

Two different questions were being answered by one mechanism:

- **Historical validity.** Did these bytes exist, unmodified, in this artifact,
  at this source revision? Answerable offline, permanently, from the artifact and
  a receipt.
- **Release authorization.** Is this artifact the one a publisher currently
  stands behind for a named corpus and channel? A claim about the present, and
  therefore time-indexed and unanswerable from the artifact alone.

Conflating them produces both errors. Treating currency as intrinsic makes a
stale artifact look authoritative; treating historical validity as expiring makes
past evidence unverifiable exactly when an audit needs it.

### Why the previous design was replaced

The 2026-07-27 version of this ADR made three decisions this one reverses.

It **centred offline operation**, and set a one-week statement validity because
that is what a disconnected consumer can tolerate. That optimises for a
deployment shape almost no consumer has — an agent performing retrieval is
connected by construction — and it bought that tolerance by giving an adversary a
week in which to serve superseded knowledge as current. The useful distinction is
not online versus offline but *can this consumer reach a trusted release
authority at decision time*, which is a property of a deployment, not of a
network.

It **had one publisher key sign both artifacts and freshness statements**,
reasoning that trusting a publisher should be one decision rather than two. That
makes compromise of the build pipeline sufficient to declare any artifact
current, and makes revocation impossible to delegate to a differently protected
key.

It **recorded rollback state as a highest-root marker**. A root is not ordered,
so it cannot detect replay; and it is not scoped, so state from one channel
compares against another. Ordering requires a sequence, and detecting a publisher
signing two different things at one sequence requires a digest.

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

Separation is the point. A key that signs artifacts must not thereby decide which
artifact is current, and neither should be able to revoke. A single all-powerful
publisher key would be simpler and would collapse every one of these into one
blast radius.

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

This is not hypothetical. The first implementation of the reference CLI passed
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

State is written temp-sync-rename. Writing in place permits a truncated file,
which on the next start reads as *no retained state* and silently downgrades a
client to first contact — precisely the exposure the state exists to close.

### Assumptions, stated rather than assumed

**A trusted clock.** Expiry cannot be evaluated without one. A consumer that
supplies none is told validity is unknown; it is never inferred from a local
clock nobody vouched for, because an adversary who can move a clock could
otherwise extend any statement indefinitely.

**Durable state.** Rollback resistance requires remembering. A consumer with no
retained state is at first contact and has none — and agent infrastructure is
frequently ephemeral, so cold start is the common case rather than the edge one.
This is why witness cosignature timestamps matter beyond equivocation detection:
they supply bounded freshness to a consumer that cannot remember anything.

### Three profiles

Selected by two questions, not one dial.

| Profile | Requires | Use when |
|---|---|---|
| **Online Standard** | signed statement, trusted clock, short validity, sequence state, fail-closed on refresh failure | ordinary operation; the adversary is a stale cache or a replayed response |
| **Online Evidenced** | the above, plus log inclusion, a witness quorum, and monitoring | the freshness claim will be shown to someone who does not trust the publisher, or the consumer keeps no durable state |
| **Cached/Disconnected** | cached statement and checkpoint, bounded by expiry, explicitly weaker | no release authority is reachable at decision time |

Online Standard is the practical default and defeats the realistic adversary. It
does not defeat a publisher signing conflicting statements to different
consumers, and it should not be described as if it does.

Online Evidenced is not merely a higher tier. A publisher-signed freshness claim
is worth very little when the publisher is the disputed party, which is the
situation ANNPack evidence exists for. It is also the only option for a
stateless consumer, because witness timestamps substitute for the durable state
such a consumer does not have.

### Transparency is consumed, not built

ANNPack operates no log. The evidenced profile verifies inclusion proofs and
witness cosignatures from an external C2SP/Sigsum-compatible log, the way it
consumes BLAKE3 and Ed25519. Operating logs, running witnesses, key management
and gossip are out of scope.

**Inclusion does not prove currency.** A witnessed old statement proves it was
published and that the publisher did not equivocate about it. It says nothing
about whether a newer statement exists, because withholding is invisible to
inclusion. Monitors detect equal-sequence conflicts, multiple current roots,
unauthorised role keys, sequence gaps, and revoked roots still advertised as
current — after the fact, which is what transparency provides and is not the
same as prevention.

### Four verdicts, and `unknown` is not `current`

`current`, `superseded`, `revoked`, `unknown`. A statement that does not mention
an artifact says nothing about it, and that is `unknown`. Reporting `unknown` as
`current` would make silence into evidence, which is the whole failure this layer
exists to prevent. `superseded` is policy; `revoked` is a security event.

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
  `unknown`. That is correct and must be loud.
- Rollback resistance is unavailable at first contact and after state loss. Said
  plainly rather than implied away.
- Statement validity is measured in hours, not a week. Short windows require
  reachable infrastructure; that trade is now made deliberately and in the right
  direction.

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

**Transparency as the base layer.** Correct for the evidenced profile and wrong
as a hard dependency of every query, because it adds availability requirements
most deployments do not need to defeat their actual adversary.
