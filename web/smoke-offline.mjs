import { spawn } from 'node:child_process';
import { readFile } from 'node:fs/promises';
import { once } from 'node:events';
import init, { blake3_hex as blake3, inflate_zlib as inflate } from './pkg/annpack.js';
import { AdyarBrowser } from './adyar-browser.js';

const wasm = await readFile(new URL('./pkg/annpack_bg.wasm', import.meta.url));
await init({ module_or_path: wasm });

const server = spawn('python3', ['-u', 'serve.py', '--port', '0'], {
  cwd: new URL('.', import.meta.url),
  stdio: ['ignore', 'pipe', 'inherit'],
});
let stopped = false;

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
  const events = [];
  const remote = await AdyarBrowser.open(`${address}/docs-v1.adyar`, {
    blake3,
    inflate,
    onRequest: (event) => events.push(event),
  });
  const offline = await remote.installOffline();
  if (offline.mode !== 'offline-memory' || offline.header.rootHash !== remote.header.rootHash) {
    throw new Error('Offline installation did not preserve the verified pack identity');
  }
  server.kill('SIGTERM');
  await once(server, 'exit');
  stopped = true;

  const response = await offline.search('Which key should rotate first?', {
    mode: 'lexical',
    limit: 1,
  });
  if (!response.results[0]?.text.includes('simple synchronous rotation API')) {
    throw new Error('Offline search returned the wrong passage after the server stopped');
  }
  if (events.some((event) => event.mode === 'offline-memory' && event.kind === 'range')) {
    throw new Error('Offline runtime attempted a network range request');
  }
  console.log(JSON.stringify({
    offline: true,
    server_stopped_before_query: true,
    root_hash: offline.header.rootHash,
    installed_bytes: offline.stats.installedBytes,
    memory_reads: offline.stats.memoryReads,
  }, null, 2));
} finally {
  if (!stopped) {
    server.kill('SIGTERM');
    await once(server, 'exit').catch(() => {});
  }
}
