import { readFile, writeFile } from 'node:fs/promises';
import { pipeline } from '@huggingface/transformers';
import { DEFAULT_EMBEDDING_PROFILE } from '../web/adyar-transformers.js';

function argument(name, fallback = null) {
  const index = process.argv.indexOf(name);
  return index < 0 ? fallback : process.argv[index + 1];
}

const input = argument('--input');
const output = argument('--output');
const kind = argument('--kind');
if (!input || !output || !['passages', 'queries'].includes(kind)) {
  throw new Error('usage: node evals/embed.mjs --kind passages|queries --input FILE --output FILE');
}

const extractor = await pipeline(
  'feature-extraction',
  DEFAULT_EMBEDDING_PROFILE.model,
  {
    revision: DEFAULT_EMBEDDING_PROFILE.revision,
    dtype: DEFAULT_EMBEDDING_PROFILE.runtime.weights_dtype,
    device: 'cpu',
  },
);

async function embed(texts) {
  const values = [];
  for (let offset = 0; offset < texts.length; offset += 32) {
    const tensor = await extractor(texts.slice(offset, offset + 32), {
      pooling: DEFAULT_EMBEDDING_PROFILE.pooling,
      normalize: DEFAULT_EMBEDDING_PROFILE.normalized,
    });
    values.push(...tensor.tolist());
  }
  return values;
}

if (kind === 'passages') {
  const passages = JSON.parse(await readFile(input, 'utf8'));
  const prefix = DEFAULT_EMBEDDING_PROFILE.document_prefix || '';
  const vectors = await embed(passages.map((passage) => `${prefix}${passage.text}`));
  await writeFile(output, `${JSON.stringify({
    profile: DEFAULT_EMBEDDING_PROFILE,
    vectors,
    passage_ids: passages.map((passage) => passage.id),
  }, null, 2)}\n`);
} else {
  const lines = (await readFile(input, 'utf8')).split(/\r?\n/u).filter((line) => line.trim());
  const records = lines.map((line) => JSON.parse(line));
  const prefix = DEFAULT_EMBEDDING_PROFILE.query_prefix || '';
  const vectors = await embed(records.map((record) => `${prefix}${record.query}`));
  const encoded = records.map((record, index) => JSON.stringify({
    ...record,
    query_vector: vectors[index],
  })).join('\n');
  await writeFile(output, `${encoded}\n`);
}
