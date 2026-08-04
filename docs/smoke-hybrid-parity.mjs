// Cross-runtime hybrid parity.
//
// The Rust reader and the browser reader implement ranking twice, in two
// languages, and nothing forced them to agree on fusion. They silently
// diverged: when the Rust fusion changed from reciprocal-rank to absolute-scale
// scoring, every existing smoke still passed while the two runtimes returned
// different hybrid rankings for the same query against the same pack. A user
// would have seen one order in the CLI and another in the browser.
//
// This asserts they agree, in the mode where it is hardest — hybrid, where both
// retrievers contribute and a fusion difference actually reorders results.
import { spawn, spawnSync } from 'node:child_process';
import { readFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

import init, { blake3_hex as blake3, inflate_zlib as inflate } from './pkg/annpack.js';
import { ANNPackBrowser } from './annpack-browser.js';

const wasm = await readFile(new URL('./pkg/annpack_bg.wasm', import.meta.url));
await init(wasm);

const webDirectory = dirname(fileURLToPath(import.meta.url));
const root = resolve(webDirectory, '..');
const binary = process.env.ANNPACK_BINARY || resolve(root, 'target/release/annpack');
const packPath = resolve(webDirectory, 'hybrid-parity.annpack');
const { writeFileSync, mkdirSync, rmSync } = await import('node:fs');

// A fixture where the two fusions provably disagree, because a fixture where
// they agree proves nothing. The first version of this test used the 7-passage
// docs-v1 pack, and reciprocal-rank fusion and absolute-scale fusion returned
// the same order on it — the test passed against a deliberately reverted
// browser and caught nothing.
//
// The shape that separates them is the one measured on the hard-negative
// corpus: a decoy that lexical ranks first and vectors rank mid-list, against a
// target that only vectors find. RRF rewards the decoy for being in both lists;
// absolute-scale fusion does not.
const corpus = resolve(root, 'target/hybrid-parity-corpus');
rmSync(corpus, { recursive: true, force: true });
mkdirSync(corpus, { recursive: true });
// The decoy repeats the query terms; the target contains neither of them.
writeFileSync(`${corpus}/decoy.md`, '# Decoy\n\nalpha beta alpha beta alpha beta filler words here.\n');
writeFileSync(`${corpus}/target.md`, '# Target\n\nthe passage a reader actually wants returned.\n');
for (let i = 0; i < 4; i += 1) {
  writeFileSync(`${corpus}/other${i}.md`, `# Other ${i}\n\nunrelated padding content number ${i}.\n`);
}

const probe = spawnSync(binary, [
  'build', corpus, '--output', `${corpus}/probe.annpack`,
  '--name', 'hybrid-parity', '--version', '1.0.0', '--json',
], { encoding: 'utf8' });
if (probe.status !== 0) throw new Error(`probe build failed: ${probe.stderr}`);
const exported = spawnSync(binary, [
  'export-passages', `${corpus}/probe.annpack`, '--output', `${corpus}/passages.json`,
], { encoding: 'utf8' });
if (exported.status !== 0) throw new Error(`export failed: ${exported.stderr}`);
const passages = JSON.parse(await readFile(`${corpus}/passages.json`, 'utf8'));

// Query points at the target. The decoy is deliberately given a small but
// non-zero cosine so it lands mid-list in the vector ranking — that is what RRF
// rewards and what absolute-scale fusion correctly discounts.
const QUERY_VECTOR = [1, 0, 0];
const vectors = passages.map((p) => {
  if (p.text.includes('actually wants')) return [1, 0, 0];
  if (p.text.includes('alpha beta')) return [0.2, 0.9797958971132712, 0];
  return [0, 1, 0];
});
writeFileSync(`${corpus}/vectors.json`, JSON.stringify({
  profile: {
    id: 'fixture-v1', model: 'deterministic-browser-fixture',
    revision: 'sha256:fixture-v1', dimensions: 3, dtype: 'float32',
    pooling: 'fixture', normalized: true,
    query_prefix: null, document_prefix: null, runtime: null,
  },
  vectors,
  passage_ids: passages.map((p) => p.id),
}));

const built = spawnSync(binary, [
  'build', corpus, '--output', packPath,
  '--name', 'hybrid-parity', '--version', '1.0.0',
  '--vectors', `${corpus}/vectors.json`, '--json',
], { encoding: 'utf8' });
if (built.status !== 0) throw new Error(`build failed: ${built.stderr}`);

const QUERY = 'alpha beta';
const vectorFile = resolve(root, 'target/hybrid-parity-query.json');
writeFileSync(vectorFile, JSON.stringify(QUERY_VECTOR));

const native = spawnSync(binary, [
  'search', packPath, QUERY,
  '--mode', 'hybrid', '--limit', '5', '--json',
  '--query-vector', vectorFile,
  '--vector-profile', 'fixture-v1',
], { encoding: 'utf8' });
if (native.status !== 0) throw new Error(`native search failed: ${native.stderr}`);
const nativeIds = JSON.parse(native.stdout).results.map((r) => r.passage_id);

const server = spawn('python3', ['-u', 'serve.py', '--port', '0'], {
  cwd: webDirectory, stdio: ['ignore', 'pipe', 'inherit'],
});
try {
  const address = await new Promise((resolveAddress, reject) => {
    let output = '';
    server.stdout.on('data', (chunk) => {
      output += chunk.toString();
      const match = output.match(/Serving (http:\/\/[^\s]+)/u);
      if (match) resolveAddress(match[1].replace(/\/$/u, ''));
    });
    server.once('exit', (code) => reject(new Error(`server exited early with ${code}`)));
  });

  const pack = await ANNPackBrowser.open(`${address}/hybrid-parity.annpack`, { blake3, inflate });
  const response = await pack.search(QUERY, {
    mode: 'hybrid',
    queryVector: QUERY_VECTOR,
    vectorProfile: 'fixture-v1',
    vectorProbes: 4,
    limit: 5,
  });
  const browserIds = response.results.map((r) => r.passage_id);

  const agree = nativeIds.length === browserIds.length
    && nativeIds.every((id, i) => id === browserIds[i]);
  console.log(JSON.stringify({
    smoke: 'hybrid-parity',
    query: QUERY,
    native_order: nativeIds.map((id) => id.slice(0, 10)),
    browser_order: browserIds.map((id) => id.slice(0, 10)),
    agree,
    result: agree ? 'PASS' : 'FAIL',
  }, null, 2));
  if (!agree) {
    throw new Error('native and browser hybrid rankings differ; the two fusions have diverged');
  }
} finally {
  server.kill();
  rmSync(vectorFile, { force: true });
  rmSync(packPath, { force: true });
  rmSync(corpus, { recursive: true, force: true });
}
