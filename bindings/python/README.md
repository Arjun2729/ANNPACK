# adyar

Python binding for the Adyar reference runtime.

Adyar compiles a documentation tree into a single immutable, content-addressed
artifact and issues offline-verifiable evidence for every passage it returns. A
third party can check that a cited passage existed unmodified in a named
artifact at a named source revision — without the artifact, without network
access, and without trusting the service that produced the citation.

## Relationship to versions below 0.5

Releases in the `0.1.x` line published a different system: an approximate
nearest-neighbour index built on FAISS, searched in the browser through a WASM
runtime. That architecture is not carried forward. The current line is a Rust
implementation of the Adyar v3 container format, exact BM25 retrieval with
normative tokenization, and the EVIDENCE-v1 receipt chain. The format, the CLI
surface, and the artifact bytes all differ. `0.1.x` artifacts cannot be read by
this runtime, and no migration path exists between them.

## This package requires the `adyar` CLI

`pip install adyar` is not a complete installation. The binding drives the
`adyar` binary as a subprocess and does not parse artifact bytes in Python, so
untrusted input stays in the Rust runtime — but the binary has to exist.

Install it from [the releases page](https://github.com/Arjun2729/ANNPACK/releases),
or from source with Rust 1.88 or newer:

```bash
cargo install --git https://github.com/Arjun2729/ANNPACK --tag v0.7.0 adyar
```

Then put it on `PATH`, set `ANNPACK_BINARY`, or pass `binary=` to the client:

```bash
adyar --version    # adyar 0.7.0
```

## Usage

```python
from adyar import Client

client = Client()

report = client.build("docs/", "knowledge.adyar", name="vendor-docs", version="1.0.0")
print(report["root_hash"])

results = client.search("knowledge.adyar", "rotate the signing key", limit=5)
for hit in results["results"]:
    print(hit["score"], hit["citation"]["canonical_url"])
```

### Retrieval evidence

A run bundle collects one agent run's retrieval evidence into a single portable
file: a standalone receipt per retrieved passage, plus the metadata needed to
locate the run in an application's own logs.

```python
bundle = client.bundle(
    "knowledge.adyar",
    "rotate the signing key",
    "run.json",
    limit=5,
    application="support-agent/2.1",
    model="claude-opus-5",
)

report = client.verify_run("run.json")
assert report["attested"]
print(report["pack_roots"], report["source_revisions"])
```

`attested` means every receipt proved its passage existed unmodified in the
named artifact. The query, application, model, and answer travel with the
receipts and are attested by nothing — the report keeps the two separate.

### OpenTelemetry attributes

`telemetry()` returns span and event attributes that bind a retrieval to the
immutable artifact it read, so a trace remains checkable after the corpus moves.

```python
attributes = client.telemetry("knowledge.adyar", "rotate the signing key")
span = attributes["span"]  # annpack.root, annpack.pack, annpack.source_revision, ...
```

## Verification without this package

Receipt verification is implemented independently in Rust, Python, and
JavaScript, and all three are held to the same conformance suite. Verifying a
receipt does not require this binding, the artifact, or network access.

## License

Apache-2.0.
