# Google OKF → ANNPack reproduction

This experiment compiles the three public OKF 0.1 bundles checked into Google's
`knowledge-catalog` repository into deterministic ANNPack artifacts. It pins the
source commit, compiler version, input format, redistribution license, build
arguments, source digest, and expected artifact roots.

```bash
cargo build --release
./launch/google-okf/reproduce.sh
```

The build is successful only if an independent run produces all three roots in
[`expected-roots.json`](expected-roots.json). The resulting packs and JSON build
reports are written under `target/google-okf-reproduction/`.

This is an interoperability fixture, not an assertion that Google publishes or
endorses ANNPack.

## Deploy the static proof to Google Cloud Storage

After making the destination bucket publicly readable, deploy any reproduced
pack together with the zero-server browser runtime:

```bash
./launch/google-okf/deploy-gcs.sh <bucket-name> \
  target/google-okf-reproduction/ga4.annpack
```

The script configures browser-visible range headers, uploads the immutable pack
under its content root, applies immutable caching to the pack, and prints the
one-bucket demonstration URL. It intentionally does not mutate bucket IAM.

## Close the loop through Gemini CLI

With Gemini CLI authenticated, configure the pack-backed read-only MCP tools and
run the scripted grounding demonstration:

```bash
./launch/google-okf/gemini-demo.sh \
  target/google-okf-reproduction/ga4.annpack
```

The prompt requires Gemini to return the pack root, exact passage hash, pinned
source revision, and canonical URL supplied by ANNPack rather than an unscoped
claim that the answer was merely “grounded.”
