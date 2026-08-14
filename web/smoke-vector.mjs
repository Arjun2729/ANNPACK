import { spawn, spawnSync } from 'node:child_process';
import { readFile, unlink } from 'node:fs/promises';
import { once } from 'node:events';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import init, { blake3_hex as blake3, inflate_zlib as inflate } from './pkg/annpack.js';
import { AdyarBrowser, createEmbeddingAdapter } from './adyar-browser.js';

const webDirectory = dirname(fileURLToPath(import.meta.url));
const root = resolve(webDirectory, '..');
const packPath = resolve(webDirectory, 'docs-vector.annpack');
const build = spawnSync(resolve(root, 'target/release/annpack'), [
  'build', resolve(root, 'fixtures/docs-v1'),
  '--output', packPath,
  '--name', 'vendor-docs',
  '--version', '1.0.0',
  '--base-url', 'https://vendor.example/docs/v1',
  '--vectors', resolve(root, 'fixtures/vectors-v1.json'),
  '--json',
], { encoding: 'utf8' });
if (build.status !== 0) throw new Error(`Vector pack build failed: ${build.stderr}`);

const wasm = await readFile(new URL('./pkg/annpack_bg.wasm', import.meta.url));
await init({ module_or_path: wasm });
const server = spawn('python3', ['-u', 'serve.py', '--port', '0'], {
  cwd: webDirectory,
  stdio: ['ignore', 'pipe', 'inherit'],
});

try {
  const address = await new Promise((resolveAddress, reject) => {
    let output = '';
    server.stdout.on('data', (chunk) => {
      output += chunk.toString();
      const match = output.match(/Serving (http:\/\/[^\s]+)/u);
      if (match) resolveAddress(match[1]);
    });
    server.once('exit', (code) => reject(new Error(`range server exited early with ${code}`)));
  });
  const pack = await AdyarBrowser.open(`${address}/docs-vector.annpack`, { blake3, inflate });
  const embed = createEmbeddingAdapter(
    async (text) => {
      if (!text.startsWith('query: ')) throw new Error('Profile query prefix was not applied');
      return [1.0, 0.0, 0.0];
    },
    {
      id: 'fixture-v1',
      model: 'deterministic-browser-fixture',
      revision: 'sha256:fixture-v1',
      dimensions: 3,
    },
  );
  const response = await pack.search('expired credential', {
    mode: 'vector',
    embed,
    vectorProfile: 'fixture-v1',
    vectorProbes: 1,
    limit: 1,
    debug: true,
  });
  if (!response.results[0]?.text.includes('API key has expired')) {
    throw new Error('Browser IVF vector search returned the wrong passage');
  }
  if (!response.pack.conformance.extensions.includes('AN-1')
    || response.results[0].evidence?.schema !== 'annpack-evidence-v1') {
    throw new Error('Browser vector response omitted AN-1 conformance or evidence');
  }
  console.log(JSON.stringify({
    browser_vector: true,
    effective_mode: response.effective_mode,
    range_requests: pack.stats.rangeRequests,
    root_hash: pack.header.rootHash,
  }, null, 2));
} finally {
  server.kill('SIGTERM');
  await once(server, 'exit').catch(() => {});
  await unlink(packPath).catch(() => {});
}
