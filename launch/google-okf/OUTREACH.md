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
   `https://<your-pages-origin>/?pack=./packs/google-okf-ga4.annpack&root=f9256b569f574d8a9068be6372a6e8d7f2b76d0afd2f0650e305218669e73b35&q=what%20does%20the%20user_properties%20field%20contain`

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

I'd genuinely value your feedback on the OKF-consumption path — whether the
mapping I'm making from OKF 0.1 into a verifiable artifact matches how you think
about the format, and anything you'd want an interop layer to preserve or expose.

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
| ga4 | `f9256b569f574d8a9068be6372a6e8d7f2b76d0afd2f0650e305218669e73b35` |
| crypto-bitcoin | `3978663b7e7cedaafeb97460f0bbab4643e21ca854271b21f03eff96c5214f97` |
| stackoverflow | `3acbe05d8c78f8c75dedcfb25843049d0cd4059b89d30c1b07c60e063a1b2d86` |

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
