# Public launch gates

> **Project checklist, not protocol specification.** Nothing in this file
> constrains the ANNPack format or a conformant implementation. Normative
> documents live under [`../spec/`](../spec/).

These gates prevent local engineering evidence from being presented as external
validation. Current release-candidate evidence lives in
[`RELEASE-READINESS.md`](RELEASE-READINESS.md). Closing an internal evidence gate
does not automatically close a stricter public-claims gate.

- [ ] Run the Core browser reader against a signed pack on a real remote CDN origin with cold browser cache, correct CORS, immutable caching, ETag stability, and production Range behavior. Record a network trace and artifact root.
- [ ] Complete the independent parser and OCI review in [`../spec/SECURITY-REVIEW.md`](../spec/SECURITY-REVIEW.md).
- [ ] Complete at least one six-hour-per-target deep fuzz run on the release commit; preserve corpus and crash artifacts. Longer continuous fuzzing remains desirable.
- [ ] Pin a license-compatible real documentation corpus and publish 50–100 human-adjudicated relevance queries using [`../evals/evaluate.py`](../evals/evaluate.py).
- [ ] Publish BM25, vector, and hybrid macro recall@5, hit-rate@5, and MRR@5 tied to an exact pack root. Do not hide a losing mode.
- [ ] Promote the candidate embedding profile to default only if the real-corpus table supports it and cold model download/inference are acceptable in target browsers.
- [ ] Replace the modeled crawl baseline with a reproducible crawl of an identified real site revision before using transfer reduction in a headline.
- [ ] Obtain a second Core reader implemented from the specification and golden corpus without importing or translating the Rust parser. Until then, call ANNPack a candidate format and reference implementation, not an adopted protocol or standard.
- [ ] For every public third-party pack, record source URL, immutable revision, source license, redistribution basis, build command, pack root, signing key, and refresh owner.
- [ ] Test GHCR push/pull against the actual public registry, then publish signed immutable digests and a catalog. Mock-registry tests do not satisfy this gate.

Public announcements, outreach, and headline performance claims follow these
gates; they are not substitutes for them.
