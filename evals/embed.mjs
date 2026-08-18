import { createHash } from 'node:crypto';
import { readFile, writeFile } from 'node:fs/promises';
import { createRequire } from 'node:module';
import { arch, platform, version as nodeVersion } from 'node:process';
import { readdirSync, readFileSync, statSync } from 'node:fs';
import { join } from 'node:path';
import { pipeline } from '@huggingface/transformers';
import { DEFAULT_EMBEDDING_PROFILE } from '../web/annpack-transformers.js';

function argument(name, fallback = null) {
  const index = process.argv.indexOf(name);
  return index < 0 ? fallback : process.argv[index + 1];
}

const input = argument('--input');
const output = argument('--output');
const kind = argument('--kind');
// The pinned profile is a candidate, not a conclusion: KP-1 says it does not
// become the release default until a real-corpus evaluation justifies it. That
// evaluation is impossible while the model is a constant, so allow an
// alternative profile to be supplied. The profile still travels into the
// vectors file, so a measurement always records which encoder produced it.
const profilePath = argument('--profile');
if (!input || !output || !['passages', 'queries'].includes(kind)) {
  throw new Error(
    'usage: node evals/embed.mjs --kind passages|queries --input FILE --output FILE [--profile FILE]',
  );
}
const PROFILE = profilePath
  ? JSON.parse(await readFile(profilePath, 'utf8'))
  : DEFAULT_EMBEDDING_PROFILE;
process.stderr.write(`embedding with ${PROFILE.model} @ ${PROFILE.runtime.weights_dtype}, ${PROFILE.dimensions}d\n`);

const extractor = await pipeline(
  'feature-extraction',
  PROFILE.model,
  {
    revision: PROFILE.revision,
    dtype: PROFILE.runtime.weights_dtype,
    device: 'cpu',
  },
);

const require_ = createRequire(import.meta.url);
// Some packages block package.json in their exports map, so fall back to
// reading it from node_modules directly rather than recording null.
const pkgVersion = (name) => {
  try { return require_(`${name}/package.json`).version; } catch { /* fall through */ }
  try {
    return JSON.parse(readFileSync(`node_modules/${name}/package.json`, 'utf8')).version;
  } catch { return null; }
};

/** SHA-256 of every weight file the profile's model actually resolved to. */
function modelFileDigests(model, revision) {
  const root = 'node_modules/@huggingface/transformers/.cache';
  const digests = {};
  const walk = (dir) => {
    let entries;
    try { entries = readdirSync(dir); } catch { return; }
    for (const entry of entries) {
      const full = join(dir, entry);
      if (statSync(full).isDirectory()) walk(full);
      else if (/\.onnx(_data)?$/u.test(entry)) {
        digests[full.slice(root.length + 1)] =
          createHash('sha256').update(readFileSync(full)).digest('hex');
      }
    }
  };
  walk(join(root, model));
  return digests;
}

// A retrieval measurement is a claim about a computation, so the artifact has
// to identify the computation. The pinned profile names the model; it does not
// name the machine, the runtime build, or the bytes that came out. Without
// those, a pack root that fails to reproduce is archaeology: this records
// enough to answer "same vectors?" before anyone asks "same builder?".
const ENVIRONMENT = {
  node: nodeVersion,
  platform,
  arch,
  transformers_js: pkgVersion('@huggingface/transformers'),
  onnxruntime_node: pkgVersion('onnxruntime-node'),
  onnxruntime_common: pkgVersion('onnxruntime-common'),
  model_files: modelFileDigests(PROFILE.model, PROFILE.revision),
};

async function embed(texts) {
  const values = [];
  for (let offset = 0; offset < texts.length; offset += 32) {
    const tensor = await extractor(texts.slice(offset, offset + 32), {
      pooling: PROFILE.pooling,
      normalize: PROFILE.normalized,
    });
    values.push(...tensor.tolist());
  }
  return values;
}

if (kind === 'passages') {
  const passages = JSON.parse(await readFile(input, 'utf8'));
  const prefix = PROFILE.document_prefix || '';
  const vectors = await embed(passages.map((passage) => `${prefix}${passage.text}`));
  const body = {
    profile: PROFILE,
    environment: ENVIRONMENT,
    vectors,
    passage_ids: passages.map((passage) => passage.id),
  };
  // Hash the vector payload on its own, not the enclosing file. The file also
  // carries the environment block, so a whole-file digest would differ between
  // two hosts even when they computed identical vectors -- which is precisely
  // the comparison this is for: same vector digest and a differing pack root
  // means the builder or its arguments moved, not the encoder.
  body.vectors_sha256 = createHash('sha256')
    .update(JSON.stringify(vectors))
    .digest('hex');
  await writeFile(output, `${JSON.stringify(body, null, 2)}\n`);
  process.stderr.write(`vectors sha256 ${body.vectors_sha256}\n`);
} else {
  const lines = (await readFile(input, 'utf8')).split(/\r?\n/u).filter((line) => line.trim());
  const records = lines.map((line) => JSON.parse(line));
  const prefix = PROFILE.query_prefix || '';
  const vectors = await embed(records.map((record) => `${prefix}${record.query}`));
  const encoded = records.map((record, index) => JSON.stringify({
    ...record,
    query_vector: vectors[index],
  })).join('\n');
  await writeFile(output, `${encoded}\n`);

  // A similarity is computed from two sides, so both need an identity. The
  // query file is JSONL that the evaluator parses line by line, and a header
  // record would break it -- hence a sidecar rather than an inline block.
  // Printing this to stderr only, as an earlier revision did, is not
  // provenance: it does not survive the run that produced it.
  const queryDigest = createHash('sha256').update(JSON.stringify(vectors)).digest('hex');
  const sidecar = `${output}.provenance.json`;
  await writeFile(sidecar, `${JSON.stringify({
    kind: 'queries',
    profile: PROFILE,
    environment: ENVIRONMENT,
    queries_sha256: createHash('sha256').update(await readFile(input)).digest('hex'),
    query_vectors_sha256: queryDigest,
    query_count: records.length,
  }, null, 2)}\n`);
  process.stderr.write(`query vectors sha256 ${queryDigest}\n`);
  process.stderr.write(`provenance ${sidecar}\n`);
}
