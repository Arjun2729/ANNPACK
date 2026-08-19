// Canonical-execution probe: run the pinned q8 model through onnxruntime-web's
// WebAssembly runtime instead of native ONNX Runtime.
//
// Native ORT selects kernels from the host ISA -- separate AVX/AVX2/AVX512,
// NEON and WASM implementations exist in MLAS -- so the same model produces
// different vectors on arm64 and x64. The question here is whether one
// compiled .wasm binary, executed single-threaded, removes that variable:
// fixed-width WASM SIMD is deterministic by specification, and it is relaxed
// SIMD, which ORT's wasm_simd kernels do not use, that is not.
//
// If this reproduces across hosts, ANN-1's canonical computation can be a
// pinned execution machine rather than a model plus a hope about the CPU.
import { readFile, writeFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import * as ort from 'onnxruntime-web';
import { AutoTokenizer } from '@huggingface/transformers';
import { DEFAULT_EMBEDDING_PROFILE as P } from '../web/annpack-transformers.js';

// One portable execution machine: same .wasm binary, single-threaded, no proxy.
ort.env.wasm.numThreads = 1;
ort.env.wasm.proxy = false;
ort.env.wasm.wasmPaths = pathToFileURL(resolve('node_modules/onnxruntime-web/dist/')).href + '/';

const MODEL = `node_modules/@huggingface/transformers/.cache/${P.model}/${P.revision}/onnx/model_quantized.onnx`;
const wasmFiles = ['ort-wasm-simd-threaded.wasm', 'ort-wasm-simd-threaded.jsep.wasm'];
for (const f of wasmFiles) {
  try {
    const b = await readFile(`node_modules/onnxruntime-web/dist/${f}`);
    console.log(`  ${f} sha256 ${createHash('sha256').update(b).digest('hex').slice(0,16)}`);
  } catch {}
}
console.log(`  model sha256 ${createHash('sha256').update(await readFile(MODEL)).digest('hex').slice(0,16)}`);
console.log(`  ort-web ${ort.env.versions.web ?? 'n/a'}  threads ${ort.env.wasm.numThreads}`);

const session = await ort.InferenceSession.create(MODEL, {
  executionProviders: ['wasm'], graphOptimizationLevel: 'all',
});
const tk = await AutoTokenizer.from_pretrained(P.model, { revision: P.revision });
const kind = process.argv.includes('--queries') ? 'queries' : 'passages';
const items = kind === 'queries'
  ? (await readFile('corpora/okf-hard-negatives.jsonl', 'utf8')).split(/\r?\n/u)
      .filter((l) => l.trim()).map((l) => JSON.parse(l))
  : JSON.parse(await readFile('../target/okf-eval/passages.json', 'utf8'));
const passages = items;

const t0 = Date.now();
const vectors = [];
for (const p of passages) {
  const enc = await tk(kind === 'queries' ? p.query : p.text);
  const feeds = {};
  for (const name of session.inputNames) {
    if (enc[name]) feeds[name] = new ort.Tensor('int64', BigInt64Array.from(Array.from(enc[name].data).map(BigInt)), enc[name].dims);
  }
  const out = await session.run(feeds);
  const hidden = out[session.outputNames[0]];
  const [, T, D] = hidden.dims;
  const mask = Array.from(enc.attention_mask.data).map(Number);
  const v = new Array(D).fill(0);
  let n = 0;
  for (let t = 0; t < T; t++) {
    if (!mask[t]) continue;
    n++;
    for (let d = 0; d < D; d++) v[d] += hidden.data[t * D + d];
  }
  for (let d = 0; d < D; d++) v[d] /= n;
  const norm = Math.sqrt(v.reduce((s, x) => s + x * x, 0));
  vectors.push(v.map((x) => x / norm));
}
console.log(`  elapsed ${((Date.now()-t0)/1000).toFixed(1)}s`);
console.log('wasm q8 vectors sha256 ' + createHash('sha256').update(JSON.stringify(vectors)).digest('hex'));
const out = process.argv[2] || '/tmp/wasm-q8.json';
if (kind === 'queries') {
  await writeFile(out, items.map((r, i) => JSON.stringify({ ...r, query_vector: vectors[i] })).join('\n') + '\n');
} else {
  await writeFile(out, JSON.stringify({ profile: P, vectors, passage_ids: passages.map((p) => p.id) }, null, 2));
}
