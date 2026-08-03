# Google OKF → ANNPack interoperability fixture

This fixture compiles three public OKF bundles from Google's
`knowledge-catalog` repository into deterministic, content-addressed ANNPack
artifacts and verifies the expected roots. It is **not** an assertion that Google
publishes, endorses, or has reviewed ANNPack. Google publishes the OKF source and
specification; the ANNPack artifacts and expected ANNPack roots are produced by
this project.

## Pinned source

| | |
|---|---|
| Repository | `github.com/GoogleCloudPlatform/knowledge-catalog` |
| Revision | `3fcbb9f` (pinned by [`reproduce.sh`](reproduce.sh)) |
| Bundles | `okf/bundles/{ga4, crypto_bitcoin, stackoverflow}` |
| Input | OKF v0.2 |
| Source license | Apache-2.0 |
| Reference compiler | `annpack-reference/0.5.0` |

## Reproduce

```bash
cargo build --release
./examples/okf-reproduction/reproduce.sh
```

The script clones the pinned revision, compiles all three bundles, and compares
the artifact roots with [`expected-roots.json`](expected-roots.json). Generated
packs and build reports are written under `target/google-okf-reproduction/`. Any
root mismatch fails the run.

An **artifact root** commits to the non-signature section-directory entries and
the stored section bytes they reference, which pins the compression output and
layout produced by this builder. It is not a whole-file hash and not a
cross-implementation identity. The manifest's `passage_merkle_root` is the
layout-independent commitment used to compare the authenticated passage records
produced by independent builders.

## Expected roots

These roots compile the pinned OKF v0.2 source with `annpack-reference/0.5.0`. They identify this builder's exact artifact bytes; the reproduction script and CI fail on any unreviewed drift.

| bundle | artifact root |
|---|---|
| ga4 | `7ae75a2da13d50fbffdbd810441c59074d4e649c06e4c547ac013dc46504b2a9` |
| crypto-bitcoin | `8301570579afff4f349f8b35bd7ee4af759d8e7604a97a7328f8b76984e116b4` |
| stackoverflow | `45aa3600f1c82284c98d26c290405c420a6525c943dad0311bfa49e0c5f405ae` |

## Live browser fixture

The GA4 artifact can be opened through the zero-server range reader:

```text
https://arjun2729.github.io/annpackv2/?pack=./packs/google-okf-ga4.annpack&root=7ae75a2da13d50fbffdbd810441c59074d4e649c06e4c547ac013dc46504b2a9&q=what%20does%20the%20user_properties%20field%20contain
```

The browser fetches strict byte ranges, checks the expected artifact root, and
returns the `user_properties` passage with an evidence envelope bound to the
opened artifact. GitHub Pages currently serves the pack as
`application/octet-stream`; the client does not depend on the media type, but a
production origin should use `application/vnd.annpack.v3` with correct CORS,
immutable caching, stable ETags, and Range behavior.

A valid demo signature proves only that one key signed the root. The included
demo key is not an external identity binding and must not be described as Google
or publisher authentication.

## OKF v0.2 implementation result

Building Google's `acme_retail` v0.2 exemplar at `3fcbb9f` produced 17 documents
and 47 passages while preserving `generated`, `verified`, `status`,
`stale_after`, and `tags` in document metadata.

That exercise exposed two defects in ANNPack's OKF reader, both subsequently
fixed:

1. it incorrectly rejected frontmatter in `log.md`;
2. it treated an absent optional `okf_version` as `0.1` instead of undeclared.

The result is an implementation finding, not an endorsement or an independent
conformance certification.

## Open interoperability questions

1. **Authoring versus packaging.** Is treating OKF as the authoring/interchange
   interface and ANNPack as one compiled, signed, range-queryable packaging layer
   consistent with OKF's stated non-goals?
2. **Actor identity versus signing keys.** `verified.by` identifies an actor, not
   a cryptographic key. Is actor-to-key binding an OKF concern, deliberately out
   of scope, or a packaging-layer responsibility?
3. **Freshness and revocation.** `stale_after` and `status` express producer intent
   inside the content. An immutable old artifact can still verify indefinitely.
   Should adversarial freshness be provided by a separately distributed,
   publisher-signed revocation/current-root statement while OKF retains the
   declared lifecycle metadata?
4. **Independent reproduction.** Does a fresh run reproduce the three expected
   roots, and does an independent reader agree on the logical passage root?

## Optional deployment fixture

```bash
./examples/okf-reproduction/deploy-gcs.sh <bucket-name> \
  target/google-okf-reproduction/ga4.annpack
```

The script uploads a content-addressed artifact and configures browser-visible
range headers and immutable caching. It does not alter bucket IAM.

## Gemini CLI demonstration

```bash
./examples/okf-reproduction/gemini-demo.sh \
  target/google-okf-reproduction/ga4.annpack
```

The demonstration asks the client to return the ANNPack artifact root, exact
passage hash, pinned source revision, and canonical URL. That proves the retrieval
provenance exposed by ANNPack; it does not prove that a generated answer follows
from the passage or that Google endorses the integration.
