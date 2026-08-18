# ANN-1: Vector retrieval

Status: implemented draft. Requires ANNPack Core v1.0-draft.

ANN-1 consists of optional section types 7, 8, and 9. A pack is ANN-1 conformant only when all three are present and mutually consistent. Their formats, bounds, exact passage-order requirement, `ivf-flat-v1` validation, dot scoring, and hybrid fusion behavior are defined in [the v3 wire format](../FORMAT-v3.md#7-vector-profiles-and-data).

An embedding profile identifies the model and exact revision, output dimensions and dtype, pooling, normalization, query/document prefixes, and—when a portable runtime is claimed—the runtime library version, weight dtype, and input-token bound. Clients reject a provider whose declared descriptor differs from the pack profile. CPU, WASM, and WebGPU are deployment providers for the same pinned ONNX space; their numerical/ranking parity must be measured, not assumed or encoded as different model identities.

The first golden-path candidate is [`../examples/default-embedding-profile.json`](../examples/default-embedding-profile.json). It pins `mixedbread-ai/mxbai-embed-xsmall-v1` to a specific revision and Transformers.js runtime. It does not become the release default until the real-corpus retrieval evaluation shows that it improves or preserves hybrid recall.

## Measured: quantized embeddings are not architecture-independent

The parity above has now been measured rather than assumed, and it fails. On an
identical corpus — same content digest, same passage order — macOS arm64 and
Linux x64 produce different vectors under every 8-bit variant tested, and
agreeing vectors under fp32:

| weights | min self-cosine | max dimension delta |
|---|---|---|
| q8 (`model_quantized.onnx`, the pinned profile) | 0.995439 | 1.57e-02 |
| int8 (`model_int8.onnx`, byte-identical output to q8) | 0.995439 | 1.57e-02 |
| uint8 (`model_uint8.onnx`) | 0.993729 | 2.00e-02 |
| fp32 (`model.onnx`) | 0.999999 | 1.10e-07 |

ONNX Runtime documents U8S8 saturation via `VPMADDUBSW` on x86-64 with
AVX2/AVX512 but without VNNI, and the Linux host used here reports exactly that
capability. That is **not** the mechanism: U8U8 avoids the instruction entirely
and diverges slightly more. What the data supports is the broader statement —
architecture-specific 8-bit kernels do not reproduce a vector across
architectures, and no tested 8-bit variant does.

The consequence for this extension is that a profile naming model, revision,
dimensions, dtype, pooling, normalization, prefixes, library and library version
**does not determine a vector**. The effective inputs also include the execution
provider, the CPU instruction set, and the kernel implementation selected from
it. On a 63-query corpus the divergence changed 21 top-5 orderings under q8 and
30 under uint8; q8 changed no hit/miss verdict and uint8 changed two, but that
is a property of one corpus and not a bound.

Not established, and deliberately not acted on here: that fp32 should become the
default. It is portable and scored best on that corpus (vector recall@5 48/63
against q8's 46/63), but `model.onnx` is 91.5 MB against 23.3 MB, which is the
cost that motivated a browser-sized model to begin with. Sixty-three
machine-authored queries do not justify a fourfold artifact-size change, and the
rule above still applies: a candidate becomes the default when a real-corpus
evaluation says so.

Open, and unresolved: whether canonical vectors and local acceleration should be
separated — an fp32 canonical path that a pack's vectors are defined against,
with quantized execution permitted for interactive query-time use and reported
as non-canonical. That would keep determinism on the path that claims it without
forcing every client to carry a 91.5 MB model, but it has not been designed, and
clients that generate their own query vectors still pay the download.

Alternative vector indexes require new ANN-1 section-format versions or later numbered extensions. Core and ANN-1 readers must not infer HNSW, quantization, sparse retrieval, or another metric from `ivf-flat-v1`.
