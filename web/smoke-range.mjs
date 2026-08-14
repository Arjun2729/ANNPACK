import { spawn, spawnSync } from 'node:child_process';
import { readFile } from 'node:fs/promises';
import { once } from 'node:events';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import init, { blake3_hex as blake3, inflate_zlib as inflate } from './pkg/adyar.js';
import { AdyarBrowser } from './adyar-browser.js';

const wasm = await readFile(new URL('./pkg/adyar_bg.wasm', import.meta.url));
await init({ module_or_path: wasm });
const webDirectory = dirname(fileURLToPath(import.meta.url));
const root = resolve(webDirectory, '..');

const server = spawn('python3', ['-u', 'serve.py', '--port', '0'], {
  cwd: new URL('.', import.meta.url),
  stdio: ['ignore', 'pipe', 'inherit'],
});

try {
  const address = await new Promise((resolve, reject) => {
    let output = '';
    server.stdout.on('data', (chunk) => {
      output += chunk.toString();
      const match = output.match(/Serving (http:\/\/[^\s]+)/u);
      if (match) resolve(match[1]);
    });
    server.once('exit', (code) => reject(new Error(`range server exited early with ${code}`)));
  });
  const packUrl = `${address}/docs-v1.adyar`;
  const packBytes = Number((await fetch(packUrl, { method: 'HEAD' })).headers.get('content-length'));
  const pack = await AdyarBrowser.open(packUrl, { blake3, inflate });
  const response = await pack.search('AP-104', { limit: 1, debug: true });
  if (!response.results[0]?.text.includes('API key has expired')) {
    throw new Error('Browser range search returned the wrong passage');
  }
  if (response.results[0].evidence?.schema !== 'annpack-evidence-v1'
    || response.results[0].evidence.pack_root !== pack.header.rootHash
    || response.results[0].evidence.passage_hash === response.results[0].passage_id
    || !response.pack.conformance.core_conformant) {
    throw new Error('Browser evidence or Core conformance output is invalid');
  }
  const native = spawnSync(resolve(root, 'target/release/adyar'), [
    'search', resolve(webDirectory, 'docs-v1.adyar'), 'AP-104',
    '--mode', 'lexical', '--limit', '1', '--json',
  ], { encoding: 'utf8' });
  if (native.status !== 0) throw new Error(`Native evidence check failed: ${native.stderr}`);
  const nativeResponse = JSON.parse(native.stdout);
  if (nativeResponse.results[0].evidence.passage_hash
    !== response.results[0].evidence.passage_hash) {
    throw new Error('Native and browser passage evidence hashes differ');
  }
  // Request count is a round-trip budget, not an efficiency claim: resolving a
  // term through the block-addressable index costs one more request than
  // downloading the whole index did, and far fewer bytes. Bytes are the point,
  // so that bound is strict; the request bound is a loose ceiling that catches
  // a runaway fetch loop. Mirrors tests/http_range.rs.
  if (pack.stats.rangeRequests < 1 || pack.stats.rangeRequests > 12) {
    throw new Error(`Expected a small bounded number of range requests, observed ${pack.stats.rangeRequests}`);
  }
  if (pack.stats.bytes >= packBytes) {
    throw new Error(`A ranged search must transfer less than the artifact: ${pack.stats.bytes} of ${packBytes}`);
  }
  console.log(JSON.stringify({
    browser_range: true,
    range_requests: pack.stats.rangeRequests,
    transferred_bytes: pack.stats.bytes,
    pack_bytes: packBytes,
    root_hash: pack.header.rootHash,
  }, null, 2));
} finally {
  server.kill('SIGTERM');
  await once(server, 'exit').catch(() => {});
}
