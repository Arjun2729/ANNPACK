// Cross-runtime run-bundle parity.
//
// A bundle is an envelope over receipts, so the danger is not a broken proof --
// it is two verifiers that disagree about what a file proves, or a verifier that
// reports success for a bundle proving nothing. The native CLI and the browser
// implement the aggregate independently; this asserts they reach the same
// verdict on the same file, and that both still reject a tampered one.
import { spawnSync } from 'node:child_process';
import { mkdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { readFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

import init, { blake3_hex as blake3, inflate_zlib as inflate } from './pkg/annpack.js';
import { verifyRunBundle } from './adyar-browser.js';
import { envVar } from '../integrations/shared/compat.mjs';

await init(await readFile(new URL('./pkg/annpack_bg.wasm', import.meta.url)));

const webDirectory = dirname(fileURLToPath(import.meta.url));
const root = resolve(webDirectory, '..');
const binary = envVar('BINARY') || resolve(root, 'target/release/adyar');
const work = resolve(root, 'target/bundle-parity');

function run(...args) {
  const result = spawnSync(binary, args, { encoding: 'utf8' });
  if (result.status !== 0) throw new Error(`${args[0]} failed: ${result.stderr}`);
  return result;
}

function verifyNative(path, trustedPublicKey) {
  const args = ['verify-run', path, '--json'];
  if (trustedPublicKey) args.push('--trusted-public-key', trustedPublicKey);
  // Exit status is part of the contract, so a failure is captured rather than
  // thrown: a tampered bundle must exit non-zero and still emit its report.
  const result = spawnSync(binary, args, { encoding: 'utf8' });
  return { ok: result.status === 0, report: JSON.parse(result.stdout) };
}

// Fields both implementations compute independently. Comparing the whole report
// would compare field order and per-receipt issue strings, which are not part of
// the contract; these are.
const COMPARED = [
  'attested', 'receipts_total', 'receipts_verified',
  'all_receipts_signed', 'all_signers_trusted', 'answer_hash_consistent',
];

function compare(label, native, browser) {
  for (const field of COMPARED) {
    if (JSON.stringify(native[field]) !== JSON.stringify(browser[field])) {
      throw new Error(
        `${label}: native and browser disagree on ${field}: `
        + `${JSON.stringify(native[field])} vs ${JSON.stringify(browser[field])}`,
      );
    }
  }
  for (const field of ['pack_roots', 'source_revisions']) {
    if (native[field].join(',') !== browser[field].join(',')) {
      throw new Error(`${label}: native and browser disagree on ${field}`);
    }
  }
}

rmSync(work, { recursive: true, force: true });
mkdirSync(`${work}/corpus`, { recursive: true });
// Both documents carry the query token, so the run retrieves more than one
// passage and the per-receipt isolation check below has something to compare.
writeFileSync(`${work}/corpus/policy.md`,
  '---\ntitle: Refund policy\nurl: https://acme.example/policy\n---\n'
  + '# Refund policy\n\nA refund is issued within fourteen days of purchase.\n');
writeFileSync(`${work}/corpus/support.md`,
  '---\ntitle: Support hours\nurl: https://acme.example/support\n---\n'
  + '# Support hours\n\nSupport answers refund questions on business days.\n');

run('build', `${work}/corpus`, '--output', `${work}/unsigned.annpack`,
  '--name', 'acme-policy', '--version', '1.0.0',
  '--source-revision', 'git:deadbeef', '--base-url', 'https://acme.example', '--json');

const keys = JSON.parse(
  run('keygen', '--output', `${work}/signing.key`, '--json').stdout,
);
run('sign', `${work}/unsigned.annpack`, '--output', `${work}/pack.annpack`,
  '--key', `${work}/signing.key`);

const answer = 'Refunds are issued within fourteen days.';
writeFileSync(`${work}/answer.txt`, answer);
run('bundle', `${work}/pack.annpack`, 'refund',
  '--limit', '2', '--application', 'support-agent/1.0', '--model', 'test-model',
  '--answer', `${work}/answer.txt`, '--output', `${work}/run.json`);

const bundle = JSON.parse(readFileSync(`${work}/run.json`, 'utf8'));
if (bundle.receipts.length < 2) {
  // With one receipt the per-receipt isolation check below compares nothing.
  throw new Error('fixture must retrieve at least two passages');
}

const results = [];
try {
  // 1. An intact, signed bundle, with and without the identity assertion.
  for (const trusted of [null, keys.public_key]) {
    const label = trusted ? 'signed + trusted key' : 'signed';
    const native = verifyNative(`${work}/run.json`, trusted);
    const browser = await verifyRunBundle(bundle, { blake3, inflate, trustedPublicKey: trusted });
    compare(label, native.report, browser);
    if (!native.ok || !browser.attested) throw new Error(`${label}: intact bundle did not attest`);
    if (browser.pack_roots.length !== 1) throw new Error(`${label}: expected exactly one artifact`);
    if (browser.answer_hash_consistent !== true) throw new Error(`${label}: answer digest not checked`);
    results.push({ case: label, attested: browser.attested, agree: true });
  }

  // 2. A valid signature from a key the caller did not authorise. Integrity
  //    still holds; the identity assertion must not.
  const other = 'a'.repeat(64);
  const nativeOther = verifyNative(`${work}/run.json`, other);
  const browserOther = await verifyRunBundle(bundle, { blake3, inflate, trustedPublicKey: other });
  compare('wrong trusted key', nativeOther.report, browserOther);
  if (nativeOther.ok) throw new Error('an unmet identity assertion must exit non-zero');
  if (!browserOther.attested || browserOther.all_signers_trusted) {
    throw new Error('wrong trusted key: expected attested integrity with untrusted identity');
  }
  results.push({ case: 'wrong trusted key', attested: true, trusted: false, agree: true });

  // 3. A rewritten passage with every hash left alone. Both runtimes must
  //    reject it, and both must still credit the receipts that survived.
  const tampered = JSON.parse(JSON.stringify(bundle));
  const record = JSON.parse(Buffer.from(tampered.receipts[0].passage_record_b64, 'base64').toString());
  record.text = 'Refunds are never issued.';
  tampered.receipts[0].passage_record_b64 = Buffer.from(JSON.stringify(record)).toString('base64');
  writeFileSync(`${work}/tampered.json`, JSON.stringify(tampered));

  const nativeTampered = verifyNative(`${work}/tampered.json`, null);
  const browserTampered = await verifyRunBundle(tampered, { blake3, inflate });
  compare('tampered', nativeTampered.report, browserTampered);
  if (nativeTampered.ok) throw new Error('a tampered bundle must exit non-zero');
  if (browserTampered.attested) throw new Error('browser attested a tampered bundle');
  if (browserTampered.receipts_verified !== bundle.receipts.length - 1) {
    throw new Error('one bad receipt should fail itself, not the whole bundle');
  }
  results.push({ case: 'tampered', attested: false, agree: true });

  // 4. A bundle stripped of its receipts proves nothing and must not read as
  //    fully signed or fully trusted just because no receipt failed.
  const emptied = { ...bundle, receipts: [] };
  writeFileSync(`${work}/emptied.json`, JSON.stringify(emptied));
  const nativeEmpty = verifyNative(`${work}/emptied.json`, keys.public_key);
  const browserEmpty = await verifyRunBundle(emptied, {
    blake3, inflate, trustedPublicKey: keys.public_key,
  });
  compare('emptied', nativeEmpty.report, browserEmpty);
  if (nativeEmpty.ok) throw new Error('an empty bundle must exit non-zero');
  if (browserEmpty.attested || browserEmpty.all_receipts_signed || browserEmpty.all_signers_trusted) {
    throw new Error('an empty bundle must not report as attested, signed, or trusted');
  }
  results.push({ case: 'emptied', attested: false, agree: true });

  console.log(JSON.stringify({
    smoke: 'bundle-parity',
    artifact: bundle.receipts[0].pack_root.slice(0, 16),
    receipts: bundle.receipts.length,
    cases: results,
    result: 'PASS',
  }, null, 2));
} finally {
  rmSync(work, { recursive: true, force: true });
}
