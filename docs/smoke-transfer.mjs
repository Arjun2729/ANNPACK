// Transfer-budget gate: proves that searching a remote artifact moves
// materially less than the artifact.
//
// This is the claim the project makes about range access, so it is asserted as
// a byte fraction rather than as "a Range header was sent". An earlier version
// of that check asserted only `rangeRequests > 0`, which passed while moving
// 97.8% of the file.
//
// The corpus is generated and the pack is built here rather than tracked,
// because the property is only meaningful at a size where the index is not
// most of the artifact. On a 4 KB pack every number is noise.
import { spawn, spawnSync } from 'node:child_process';
import { mkdirSync, writeFileSync, rmSync, statSync } from 'node:fs';
import { readFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

import init, { blake3_hex as blake3, inflate_zlib as inflate } from './pkg/adyar.js';
import { AdyarBrowser } from './adyar-browser.js';
import { envVar } from '../integrations/shared/compat.mjs';

const wasm = await readFile(new URL('./pkg/adyar_bg.wasm', import.meta.url));
await init(wasm);

const webDirectory = dirname(fileURLToPath(import.meta.url));
const root = resolve(webDirectory, '..');
const binary = envVar('BINARY') || resolve(root, 'target/release/adyar');
const corpus = resolve(root, 'target/transfer-corpus');
const packPath = resolve(webDirectory, 'transfer-fixture.annpack');
const QUERY = 'rotation cache token001234';
const BUDGET = 0.45;

// Deterministic corpus: fixed vocabulary, fixed layout, no randomness, so the
// measured fraction is reproducible across machines and runs.
rmSync(corpus, { recursive: true, force: true });
mkdirSync(corpus, { recursive: true });
for (let document = 0; document < 300; document += 1) {
  const sections = [];
  for (let section = 0; section < 5; section += 1) {
    const words = [];
    for (let word = 0; word < 60; word += 1) {
      words.push(`token${String((document * 300 + section * 60 + word) % 9000).padStart(6, '0')}`);
    }
    words.push('rotation', 'cache');
    sections.push(`## Section ${section}\n\n${words.join(' ')}\n`);
  }
  writeFileSync(`${corpus}/doc${String(document).padStart(4, '0')}.md`,
    `# Document ${document}\n\n${sections.join('\n')}`);
}

const built = spawnSync(binary, [
  'build', corpus, '--output', packPath,
  '--name', 'transfer-fixture', '--version', '1.0.0', '--json',
], { encoding: 'utf8' });
if (built.status !== 0) throw new Error(`build failed: ${built.stderr}`);

const server = spawn('python3', ['-u', 'serve.py', '--port', '0'], { cwd: webDirectory });
try {
  const address = await new Promise((resolveAddr, reject) => {
    const onData = (chunk) => {
      const match = chunk.toString().match(/Serving (http:\/\/[^\s]+)/u);
      if (match) resolveAddr(match[1].replace(/\/$/u, ''));
    };
    server.stdout.on('data', onData);
    server.stderr.on('data', onData);
    setTimeout(() => reject(new Error('server did not start')), 5000);
  });

  const packBytes = statSync(packPath).size;
  const pack = await AdyarBrowser.open(`${address}/transfer-fixture.annpack`, { blake3, inflate });
  const openBytes = pack.stats.bytes;
  const openRequests = pack.stats.rangeRequests;

  const response = await pack.search(QUERY, { limit: 5, debug: true });
  const hit = response.results[0];

  const transferred = pack.stats.bytes;
  const fraction = transferred / packBytes;

  const checks = {
    core_conformant: response.pack.conformance.core_conformant === true,
    returned_a_result: Boolean(hit?.passage_id),
    evidence_root_bound: hit?.evidence?.pack_root === pack.header.rootHash,
    within_transfer_budget: fraction <= BUDGET,
  };
  const passed = Object.values(checks).every(Boolean);

  console.log(JSON.stringify({
    smoke: 'transfer-budget',
    query: QUERY,
    pack_bytes: packBytes,
    // Split out because they fail for different reasons: open cost is the index
    // floor, paid once per session regardless of query; query cost scales with
    // how many blocks the terms and results actually touch.
    open_bytes: openBytes,
    open_fraction: Number((openBytes / packBytes).toFixed(4)),
    open_requests: openRequests,
    total_bytes: transferred,
    total_requests: pack.stats.rangeRequests,
    transferred_fraction: Number(fraction.toFixed(4)),
    transfer_budget: BUDGET,
    checks,
    result: passed ? 'PASS' : 'FAIL',
  }, null, 2));
  if (!passed) process.exitCode = 1;
} finally {
  server.kill();
  rmSync(packPath, { force: true });
  rmSync(corpus, { recursive: true, force: true });
}
