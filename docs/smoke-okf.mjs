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

import init, { blake3_hex as blake3, inflate_zlib as inflate } from './pkg/adyar.js';
import { AdyarBrowser } from './adyar-browser.js';

const wasm = await readFile(new URL('./pkg/adyar_bg.wasm', import.meta.url));
await init(wasm);

const webDirectory = dirname(fileURLToPath(import.meta.url));
const expectedRoots = JSON.parse(
  await readFile(new URL('../examples/okf-reproduction/expected-roots.json', import.meta.url), 'utf8'),
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

  const packUrl = `${address}/packs/google-okf-ga4.annpack`;
  const packBytes = Number((await fetch(packUrl, { method: 'HEAD' })).headers.get('content-length'));

  const pack = await AdyarBrowser.open(packUrl, { blake3, inflate });
  const response = await pack.search(QUERY, { limit: 1, debug: true });
  const hit = response.results[0];

  // This demo proves provenance and interop: that the artifact reproduces
  // Google's pinned OKF bundle and that a cited answer binds to its root.
  //
  // It deliberately does NOT gate on transfer fraction. At 23 KB this pack is
  // smaller than one CDN response is worth optimizing, and its indexes are
  // most of it, so any range-efficiency number here is noise. The previous
  // `rangeRequests > 0` check claimed otherwise while moving 97.8% of the
  // file. The transfer budget is enforced in smoke-transfer.mjs, against a
  // pack large enough for it to mean something.
  const transferred = pack.stats.bytes;
  const fraction = transferred / packBytes;

  const checks = {
    core_conformant: response.pack.conformance.core_conformant === true,
    root_matches_expected_annpack_reproduction: pack.header.rootHash === EXPECTED_ROOT,
    source_revision_matches_pinned_okf: pack.manifest.source_revision === `git:${PINNED_REVISION}`,
    evidence_schema: hit?.evidence?.schema === 'annpack-evidence-v1',
    evidence_root_bound: hit?.evidence?.pack_root === pack.header.rootHash,
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
    transferred_bytes: transferred,
    pack_bytes: packBytes,
    transferred_fraction: Number(fraction.toFixed(4)),
    checks,
    result: passed ? 'PASS' : 'FAIL',
  }, null, 2));
  if (!passed) process.exitCode = 1;
} finally {
  server.kill();
}
