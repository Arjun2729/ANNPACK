# ANNPack compatibility and release policy

This policy exists so an external reviewer has a stable target and a clear answer
to the question every reviewer asks: *if I find a format defect, what happens to
it?*

## Tags are immutable

Every published tag is immutable. A tag is never moved, re-pointed, or deleted.
If a tagged release is wrong, it is superseded or withdrawn — never edited.

## Release candidates absorb format changes

A **format-changing review finding before final release produces a new release
candidate**, even when it changes artifact roots again.

`v0.4.0-rc1` → `v0.4.0-rc2` → `v0.4.0-rc3` → … → `v0.4.0`

Reviewers should therefore report format defects freely during the RC period.
Root churn between candidates is expected and is not a reason to soften, defer,
or re-scope a finding. An RC exists precisely to absorb this.

## v0.4.0 final freezes the format

Once `v0.4.0` is tagged:

- the wire format is frozen
- the manifest schema is frozen
- artifact roots for a given source and build configuration are frozen

Any later breaking change goes to **v0.5.0** with its own RC cycle.

## Correctness outranks root stability

A **critical security finding can withdraw a release candidate at any time.**
Preserving a candidate's roots never overrides correctness. If the choice is
between a stable root and a correct one, the root loses.

This applies after `v0.4.0` final as well: a critical security defect is grounds
for an out-of-band release, and the freeze above is not a commitment to ship
known-broken bytes.

## Reader compatibility

Independently of the above:

- A reader MUST accept every manifest section format version it implements, and
  MUST refuse an unknown one with an explicit version error at the container
  boundary rather than failing inside deserialization (FORMAT-v3 §4.2).
- A reader MUST ignore unknown *optional* sections and unknown manifest fields.
- A reader MUST reject unknown *required* sections and unknown required codecs.
- Previously published artifacts keep their original artifact roots forever. A
  root reset changes what a *builder emits*, never what a *reader computes* for
  an existing artifact. `tests/compatibility.rs` pins this.

## Normativity

The specification is normative. Where an implementation and the reference
implementation disagree, and the specification permits the implementation's
reading, the **reference implementation** is what changes. The specification is
not retroactively edited to match reference behaviour.

## History of root changes

Recorded because two resets in four days is the kind of thing a reviewer should
see acknowledged rather than discover.

| Version | Root change | Why |
|---|---|---|
| v0.3.1 | yes | Removed the builder identifier from the manifest. Correct change, but shipped as a patch without a schema-version bump, which broke old readers one-way. |
| v0.4.0-rc1 | yes | Made that boundary explicit (manifest section format 2) and added the logical content root required for evidence receipts. |
| v0.4.0 final | frozen | — |
