# OKF team outreach — draft

Framing: technical-peer / interop feedback, **not** a product pitch. You built
something on their format that reproduces their own published bundles. Do not
overclaim: no "independently security-reviewed", no headline retrieval-quality
percentages, no "adopted standard". Keep the humility line.

Two artifacts back every version below:

1. **Self-verifying reproduction** (they run it, trust nothing):
   ```bash
   git clone https://github.com/<you>/annpackv2 && cd annpackv2
   cargo build --release
   ./launch/google-okf/reproduce.sh
   ```
   Clones `GoogleCloudPlatform/knowledge-catalog` at `d44368c`, compiles the three
   OKF 0.1 bundles, and verifies all three artifact roots against
   `launch/google-okf/expected-roots.json`.

2. **Live, zero-server browser demo** (clickable; range-fetches + verifies in-page):
   `https://<your-pages-origin>/?pack=./packs/google-okf-ga4.annpack&root=5381831ae89f9de25dcc9cf4ec49958cce783460ee772dc840714e0432b31e3d&q=what%20does%20the%20user_properties%20field%20contain`

---

## Version A — email to maintainers (short)

**Subject:** OKF 0.1 interop: reproducing your three bundles as verifiable, content-addressed artifacts

Hi <name>,

I've been building ANNPack, an open (Apache-2.0) format that packages knowledge
into a deterministic, content-addressed, range-queryable artifact whose search
results cite the exact immutable passage that produced them.

It consumes OKF 0.1 directly. As an interop check I compiled the three public
bundles in `knowledge-catalog` (at `d44368c`) into ANNPack artifacts — the build
is deterministic, so an independent run reproduces all three artifact roots
bit-for-bit. Two commands reproduce it end to end, cloning your repo and verifying
against pinned expected roots:

    cargo build --release && ./launch/google-okf/reproduce.sh

There's also a zero-server browser demo that range-fetches one reproduced bundle
off a CDN and verifies its root client-side, then answers a question with a
passage-level evidence envelope: <live URL>

The way I think about the relationship: **OKF is the source/authoring format;
ANNPack is a compiled, content-addressed, verifiable, range-servable artifact +
retrieval/evidence layer on top of it** — roughly OKF : ANNPack :: source (or a
Dockerfile) : a signed container image. Which raises the one question I'd most
value your read on: **is that complementary to where you're taking OKF, or is
verification/packaging/serving heading into OKF's own scope?** I'd rather hear
"that overlaps our roadmap" now than build a redundant layer. Either answer is
useful to me.

(And more concretely: does the OKF 0.1 → verifiable-artifact mapping preserve
what matters to you, and is there anything an interop layer should expose that I'm
dropping?)

(To be clear, this is an independent interop experiment — not a claim that Google
publishes or endorses ANNPack.)

Thanks for the format — it made a clean deterministic target to build against.

<you>
<repo link>

---

## Version B — GitHub issue / discussion on `knowledge-catalog` (public, warm)

**Title:** Interop: reproducing the three OKF 0.1 bundles as verifiable content-addressed artifacts

Hi — I built an open, deterministic packaging + retrieval format (ANNPack) that
consumes OKF 0.1, and wanted to share an interop result and ask for feedback.

Compiling the three bundles under `okf/bundles/` at `d44368c` produces ANNPack
artifacts whose roots are reproducible by the same builder across environments. (The artifact root commits to compression and layout, so it is an identity for one artifact, not a cross-implementation identity; the manifest also carries a layout-independent `passage_merkle_root` for that purpose.) A two-command
reproduction clones this repo at the pinned commit and verifies all three roots:

```bash
cargo build --release
./launch/google-okf/reproduce.sh
```

Expected roots (also checked in as `expected-roots.json`):

| bundle | root |
|---|---|
| ga4 | `5381831ae89f9de25dcc9cf4ec49958cce783460ee772dc840714e0432b31e3d` |
| crypto-bitcoin | `92632e4d4936e964e575882a117741e95fb5830a1467edc87470bbc424d1d31a` |
| stackoverflow | `f324253c8e0376aeca97f7bb42f50d91d542c6969191bc12b10dce21904733d3` |

What I'd love feedback on:
- Does the OKF 0.1 → verifiable-artifact mapping preserve what matters to you?
- Anything an interop layer should expose that I'm dropping?

Independent experiment; not implying any endorsement. Repro + a live browser demo
in the links.

---

## Do NOT include (until the gates close)

- Any retrieval quality percentage (Gate 4 human-label independence unconfirmed).
- "Independently security-reviewed" (Gate 2 external review outstanding).
- "Standard" / "adopted" language (needs external uptake).

---

## The ask, stated precisely

Do **not** ask for endorsement, a partnership, or a quote. Ask three technical
questions that a maintainer can answer from the artifacts alone:

1. **Semantics.** Does the ANNPack compilation preserve the OKF semantics that
   matter to you — `type`, `concept_id`, version, the index/log/concept
   distinction, and YAML metadata? We surface all of these in the pack, and we
   validate OKF conformance during ingestion (a concept missing a non-empty
   `type` is rejected). We would rather learn now if we are preserving the wrong
   things.

2. **Layering.** Is "OKF is the authoring/interchange source, ANNPack is a
   compiled, verifiable, range-queryable artifact" a reasonable division? We are
   deliberately not proposing a change to OKF. We think of ANNPack as a
   compiler's output format, not a competitor.

3. **Reproduction.** Would you run `launch/google-okf/reproduce.sh` and either
   confirm the three artifact roots or tell us where they diverge? A discrepancy
   is a more useful result to us than a match.

### What counts as success

A public issue reply, a merged interop fixture, a reproduced root, or a written
technical objection. All four are real validation. "Google likes it" is not a
goal and should not be pursued.

### What we must never imply

Google publishes the OKF **source** bundles. We produce the ANNPack artifacts and
publish the expected **ANNPack** roots of our own reproduction. Those roots are
ours. Google neither publishes ANNPack artifacts nor endorses this project. The
demo output field is named `root_matches_expected_annpack_reproduction`
specifically so a screenshot cannot be misread as a Google-published root.

### Send only after

- [ ] the live CDN demo works from a clean clone (Gate 9)
- [ ] `launch/google-okf/reproduce.sh` passes on a machine that is not the author's
- [ ] no retrieval-quality numbers appear anywhere in the message
