// OKF demo validator: proves the zero-server browser runtime range-fetches and
// verifies an ANNPack artifact compiled from Google's published OKF (GA4)
// bundle, then returns a cited answer with an evidence envelope. Mirrors
// smoke-range.mjs but targets the OKF interop pack.
//
// Attribution, precisely: Google publishes the OKF *source* bundle. This project
// compiles it into an ANNPack artifact and publishes the *expected ANNPack root*
// of that reproduction. The root is ours, not Google's, and Google neither
// publishes ANNPack artifacts nor endorses this project.
import { spawn } from 'node:child_process';
import { readFile } from 'node:fs/promises';
import { dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

import init, { blake3_hex as blake3, inflate_zlib as inflate } from './pkg/annpack.js';
import { ANNPackBrowser } from './annpack-browser.js';

const wasm = await readFile(new URL('./pkg/annpack_bg.wasm', import.meta.url));
await init(wasm);

const webDirectory = dirname(fileURLToPath(import.meta.url));
const expectedRoots = JSON.parse(
  await readFile(new URL('../launch/google-okf/expected-roots.json', import.meta.url), 'utf8'),
);
const EXPECTED_ROOT = expectedRoots.artifacts.ga4;
const PINNED_REVISION = expectedRoots.source.revision;
const QUERY = 'what does the user_properties field contain';

const server = spawn('python3', ['-u', 'serve.py', '--port', '0'], { cwd: webDirectory });
try {
  const address = await new Promise((resolveAddr, reject) => {
    server.stdout.on('data', (chunk) => {
      const match = chunk.toString().match(/Serving (http:\/\/[^\s]+)/u);
      if (match) resolveAddr(match[1].replace(/\/$/u, ''));
    });
    server.stderr.on('data', (chunk) => {
      const match = chunk.toString().match(/Serving (http:\/\/[^\s]+)/u);
      if (match) resolveAddr(match[1].replace(/\/$/u, ''));
    });
    setTimeout(() => reject(new Error('server did not start')), 5000);
  });

  const pack = await ANNPackBrowser.open(`${address}/packs/google-okf-ga4.annpack`, { blake3, inflate });
  const response = await pack.search(QUERY, { limit: 1, debug: true });
  const hit = response.results[0];

  const checks = {
    core_conformant: response.pack.conformance.core_conformant === true,
    root_matches_expected_annpack_reproduction: pack.header.rootHash === EXPECTED_ROOT,
    source_revision_matches_pinned_okf: pack.manifest.source_revision === `git:${PINNED_REVISION}`,
    evidence_schema: hit?.evidence?.schema === 'annpack-evidence-v1',
    evidence_root_bound: hit?.evidence?.pack_root === pack.header.rootHash,
    ranged_not_full_download: pack.stats.rangeRequests > 0,
  };
  const passed = Object.values(checks).every(Boolean);

  console.log(JSON.stringify({
    demo: 'annpack-reproduction-of-google-okf-ga4',
    query: QUERY,
    pinned_source_revision: PINNED_REVISION,
    root_hash: pack.header.rootHash,
    expected_root: EXPECTED_ROOT,
    answer: hit?.text?.slice(0, 120),
    passage_id: hit?.passage_id,
    range_requests: pack.stats.rangeRequests,
    transferred_bytes: pack.stats.bytes,
    checks,
    result: passed ? 'PASS' : 'FAIL',
  }, null, 2));
  if (!passed) process.exitCode = 1;
} finally {
  server.kill();
}
