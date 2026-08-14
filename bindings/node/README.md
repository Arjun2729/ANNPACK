# @adyar/node

Node.js binding for the [Adyar](https://github.com/Arjun2729/ANNPACK)
reference runtime: verifiable knowledge artifacts and portable retrieval
evidence.

## This package requires the `annpack` CLI

It is a thin process binding. Every method spawns the `annpack` binary and
parses its JSON, so **installing this package alone is not enough** — the CLI
has to be on your `PATH`, or named explicitly.

Parsing untrusted artifact bytes stays in the Rust runtime rather than being
reimplemented here, which is the reason for the split: a bounds or codec bug in
a hand-written JavaScript parser would be a security defect in every consumer.

```bash
npm install @adyar/node
```

Then install the runtime — a release binary for your platform from
[the releases page](https://github.com/Arjun2729/ANNPACK/releases), or from
source:

```bash
cargo install --git https://github.com/Arjun2729/ANNPACK --tag v0.7.0 annpack
```

Verify it is visible:

```bash
annpack --version    # annpack 0.7.0
```

## Use

```js
import { Client } from '@adyar/node';

const annpack = new Client();                  // or: new Client({ binary: '/path/to/annpack' })

annpack.inspect('corpus.annpack');             // manifest, sections, conformance
annpack.verify('corpus.annpack');              // artifact integrity
annpack.search('corpus.annpack', 'AP-104');    // ranked results with evidence
```

`ANNPACK_BINARY` overrides the binary path if you would rather not pass it in
code.

Every call throws `AdyarError` when the CLI exits non-zero, so a failed
verification is an exception rather than a falsy return you might not check.

## Versioning

This package tracks the CLI it drives. A binding whose version lagged the
runtime would describe a command surface different from the one it invokes, so
they are released together and are expected to match.

## License

Apache-2.0. The specification, wire format, and conformance packet live in
[the main repository](https://github.com/Arjun2729/ANNPACK).
