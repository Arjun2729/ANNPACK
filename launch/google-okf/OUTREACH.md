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
   `https://<your-pages-origin>/?pack=./packs/google-okf-ga4.annpack&root=b45a93d8145cb993d9025c40956318339c948d8109061574e8dc3b6174281fc4&q=what%20does%20the%20user_properties%20field%20contain`

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
artifacts whose roots are stable across independent builds. A two-command
reproduction clones this repo at the pinned commit and verifies all three roots:

```bash
cargo build --release
./launch/google-okf/reproduce.sh
```

Expected roots (also checked in as `expected-roots.json`):

| bundle | root |
|---|---|
| ga4 | `b45a93d8145cb993d9025c40956318339c948d8109061574e8dc3b6174281fc4` |
| crypto-bitcoin | `1a6c4e6d906ea75161ddb14be2d4094323fdf944c54992e5c49f1e1b20849d56` |
| stackoverflow | `fd8500d9a86f35f3bc7ab32c4932eed35d56954a4dd43c579cdc43a9dd0e8556` |

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
