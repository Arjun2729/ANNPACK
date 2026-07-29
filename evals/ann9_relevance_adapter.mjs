// ANN-9 cross-model RELEVANCE test + anchor-supervised linear adapter.
//
// Fixes the two flaws in the self-identity kill-switch:
//   Flaw 1 (wrong task): that harness measured cross-model DUPLICATE matching
//     (passage i under B retrieves passage i under A), which is easy for any two
//     English models and saturated the baseline. The product task is RELEVANCE:
//     query Q (one model) retrieves a relevant but DIFFERENT doc D (another
//     model), Q != D, with hard negatives. This harness uses real FastAPI qrels
//     (142 queries, relevance judged by source path) over a real doc corpus.
//   Flaw 2 (wrong tool): that harness only tried isometric alignment (orthogonal
//     Procrustes). Independently-trained models are NOT isometric. Here we add a
//     ridge-regression linear ADAPTER fit on the shared anchor pairs — the fair,
//     correct tool — that maps the query model's space into the doc model's space
//     (obviating lossy relative coordinates entirely).
//
// Configs compared (metric: nDCG@10 / success@10 / recall@10 / MRR):
//   (a) same-model     — query embedded with model A vs corpus A  ........ CEILING
//   (b) raw cross      — query embedded with model B vs corpus A (dims must match)
//   (c) anchor-relative— ANN-9 as specified, centered (both sides -> anchor coords)
//   (d) ridge adapter  — map query_B into A-space via anchor-supervised linear map
//
// A=doc/corpus model (also the "old" model). B=query/"new" model. If (d) recovers
// most of the (a) ceiling while (b) fails, the adapter is the real migration path.
//
// Usage: node ann9_relevance_adapter.mjs
//   env: MODEL_A, MODEL_B, N_ANCHOR, RIDGE, MAX_PASSAGES, SEED

import { pipeline } from '@huggingface/transformers';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const MODEL_A = process.env.MODEL_A || 'Xenova/all-MiniLM-L6-v2'; // doc/corpus + ceiling model
const MODEL_B = process.env.MODEL_B || 'Xenova/gte-base';        // query / "new" model
const N_ANCHOR = Number(process.env.N_ANCHOR || 1500); // in-domain first, generic fill
const DH = Number(process.env.DH || 1024);             // random-feature dim for the non-linear adapter
const RIDGE_GRID = (process.env.RIDGE_GRID || '0.003,0.01,0.03,0.1,0.3').split(',').map(Number); // lambda CV grid (fraction of mean gram diag)
const MAX_PASSAGES = Number(process.env.MAX_PASSAGES || 4000);
const K = 10;
const SEED = Number(process.env.SEED || 1234);

const REPO_ROOT = process.env.ANNPACK_REPO_ROOT
  || path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const FASTAPI_EVAL_ROOT = process.env.FASTAPI_EVAL_ROOT
  || path.join(REPO_ROOT, 'target/fastapi-eval');
const DOCS_ROOT = process.env.FASTAPI_DOCS_ROOT
  || path.join(FASTAPI_EVAL_ROOT, 'fastapi-src/docs/en/docs');
const QREL_FILES = process.env.FASTAPI_QREL_FILES
  ? process.env.FASTAPI_QREL_FILES.split(path.delimiter)
  : [
      path.join(FASTAPI_EVAL_ROOT, 'qrels-labeled.jsonl'),
      path.join(REPO_ROOT, 'launch/evidence/2026-07-20/workstream3-evals/fastapi-candidate-qrels.jsonl'),
    ];
// anchors: repo prose DISJOINT from the English eval docs corpus (public-anchor-
// set analogue). Broad roots for volume; the whole en docs tree (DOCS_ROOT) is
// skipped inside proseSentences so no eval passage can leak in as an anchor.
const ANCHOR_ROOTS = process.env.ANNPACK_ANCHOR_ROOTS
  ? process.env.ANNPACK_ANCHOR_ROOTS.split(path.delimiter)
  : [
      path.join(REPO_ROOT, 'spec'),
      path.join(REPO_ROOT, 'launch'),
      path.join(REPO_ROOT, 'rust/src'),
      path.join(REPO_ROOT, 'bindings'),
      path.join(REPO_ROOT, 'README.md'),
      path.join(FASTAPI_EVAL_ROOT, 'fastapi-src/fastapi'),
      path.join(FASTAPI_EVAL_ROOT, 'fastapi-src/tests'),
    ];

function mulberry32(a) {
  return function () {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
const rng = mulberry32(SEED);
function shuffle(arr) {
  for (let i = arr.length - 1; i > 0; i--) { const j = Math.floor(rng() * (i + 1)); [arr[i], arr[j]] = [arr[j], arr[i]]; }
  return arr;
}

// prose sentence extractor (for anchors) ------------------------------------
function proseSentences(roots) {
  const seen = new Set(); const out = [];
  const walk = (dir, depth) => {
    if (depth > 6) return;
    let entries; try { entries = fs.readdirSync(dir, { withFileTypes: true }); } catch { return; }
    for (const e of entries) {
      if (e.name.startsWith('.') || e.name === 'node_modules') continue;
      const p = path.join(dir, e.name);
      if (p === DOCS_ROOT) continue; // never let an English eval passage become an anchor
      if (e.isDirectory()) walk(p, depth + 1);
      else if (/\.(md|rs|py|txt)$/.test(e.name)) {
        let txt; try { txt = fs.readFileSync(p, 'utf8'); } catch { continue; }
        for (const raw of txt.split(/(?<=[.!?])\s+|\n/)) {
          const s = raw.trim().replace(/\s+/g, ' ');
          if (s.length < 45 || s.length > 260) continue;
          if (s.split(' ').length < 8) continue;
          const alpha = (s.match(/[A-Za-z]/g) || []).length;
          if (alpha / s.length < 0.72) continue;
          if (/[{}<>=|#*`\[\]]/.test(s)) continue;
          if (!/^[A-Z"']/.test(s)) continue;
          const key = s.toLowerCase(); if (seen.has(key)) continue; seen.add(key); out.push(s);
        }
      }
    }
  };
  for (const r of roots) { try { fs.statSync(r).isDirectory() ? walk(r, 0) : null; } catch {} }
  return out;
}

// corpus: chunk the FastAPI docs, tag each passage with its source path --------
function buildCorpus() {
  const passages = [];
  const walk = (dir) => {
    let entries; try { entries = fs.readdirSync(dir, { withFileTypes: true }); } catch { return; }
    for (const e of entries) {
      const p = path.join(dir, e.name);
      if (e.isDirectory()) walk(p);
      else if (e.name.endsWith('.md')) {
        const rel = path.relative(DOCS_ROOT, p); // matches qrels relevant_source_paths
        let txt; try { txt = fs.readFileSync(p, 'utf8'); } catch { continue; }
        txt = txt.replace(/```[\s\S]*?```/g, ' ')      // drop fenced code
                 .replace(/<!--[\s\S]*?-->/g, ' ')      // drop html comments
                 .replace(/^---[\s\S]*?---/m, ' ');     // drop frontmatter
        for (let block of txt.split(/\n\s*\n/)) {
          block = block.replace(/[#>*`]/g, ' ')
                       .replace(/\[([^\]]*)\]\([^)]*\)/g, '$1') // md links -> text
                       .replace(/\s+/g, ' ').trim();
          if (block.length < 80) continue;
          // split overlong blocks on sentence boundaries into ~800-char passages
          if (block.length <= 1200) { passages.push({ text: block, path: rel }); continue; }
          let cur = '';
          for (const sent of block.split(/(?<=[.!?])\s+/)) {
            if ((cur + ' ' + sent).length > 900 && cur.length >= 80) { passages.push({ text: cur.trim(), path: rel }); cur = sent; }
            else cur += ' ' + sent;
          }
          if (cur.trim().length >= 80) passages.push({ text: cur.trim(), path: rel });
        }
      }
    }
  };
  walk(DOCS_ROOT);
  return passages.slice(0, MAX_PASSAGES);
}

function loadQueries() {
  const out = [];
  for (const f of QREL_FILES) {
    let txt; try { txt = fs.readFileSync(f, 'utf8'); } catch { continue; }
    for (const line of txt.split('\n')) {
      if (!line.trim()) continue;
      const o = JSON.parse(line);
      const rel = new Set(o.relevant_source_paths || []);
      if (o.query && rel.size) out.push({ query: o.query, rel });
    }
  }
  return out;
}

// embedding ------------------------------------------------------------------
async function embedAll(modelId, texts, label) {
  const opts = {}; if (modelId.includes('mxbai')) opts.dtype = 'q8';
  const extractor = await pipeline('feature-extraction', modelId, opts);
  const vecs = []; const BATCH = 32;
  for (let i = 0; i < texts.length; i += BATCH) {
    const t = await extractor(texts.slice(i, i + BATCH), { pooling: 'mean', normalize: true });
    const d = t.dims[t.dims.length - 1];
    for (let r = 0; r < Math.min(BATCH, texts.length - i); r++) vecs.push(Float32Array.from(t.data.slice(r * d, (r + 1) * d)));
    process.stdout.write(`\r  ${label} [${modelId}]: ${Math.min(i + BATCH, texts.length)}/${texts.length}   `);
  }
  process.stdout.write('\n');
  return vecs;
}

// linear algebra -------------------------------------------------------------
function dot(a, b) { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; }
function normalizeRow(row) {
  let n = 0; for (let i = 0; i < row.length; i++) n += row[i] * row[i]; n = Math.sqrt(n) || 1;
  const o = new Float32Array(row.length); for (let i = 0; i < row.length; i++) o[i] = row[i] / n; return o;
}
function meanVec(rows) { const d = rows[0].length, m = new Float64Array(d); for (const r of rows) for (let j = 0; j < d; j++) m[j] += r[j]; for (let j = 0; j < d; j++) m[j] /= rows.length; return m; }
function subMean(row, m) { const o = new Float32Array(row.length); for (let j = 0; j < row.length; j++) o[j] = row[j] - m[j]; return o; }
function relRow(vec, anchors) { const r = new Float32Array(anchors.length); for (let k = 0; k < anchors.length; k++) r[k] = dot(vec, anchors[k]); return r; }

// ridge adapter: fit W (dB x dA) minimizing ||Xc W - Yc||^2 + lambda||W||^2
// on centered anchor pairs (Xc = B-anchors, Yc = A-anchors). Solve SPD system
// (Xc^T Xc + lambda I) W = Xc^T Yc via Cholesky.
function fitRidge(Xrows, Yrows, ridgeFrac) {
  const S = Xrows.length, dB = Xrows[0].length, dA = Yrows[0].length;
  const G = Array.from({ length: dB }, () => new Float64Array(dB));
  for (let s = 0; s < S; s++) { const x = Xrows[s]; for (let i = 0; i < dB; i++) { const xi = x[i]; if (!xi) continue; const gi = G[i]; for (let j = 0; j < dB; j++) gi[j] += xi * x[j]; } }
  let diag = 0; for (let i = 0; i < dB; i++) diag += G[i][i]; const lambda = ridgeFrac * (diag / dB);
  for (let i = 0; i < dB; i++) G[i][i] += lambda;
  const B = Array.from({ length: dB }, () => new Float64Array(dA));
  for (let s = 0; s < S; s++) { const x = Xrows[s], y = Yrows[s]; for (let i = 0; i < dB; i++) { const xi = x[i]; if (!xi) continue; const bi = B[i]; for (let j = 0; j < dA; j++) bi[j] += xi * y[j]; } }
  // Cholesky G = L L^T
  const L = Array.from({ length: dB }, () => new Float64Array(dB));
  for (let j = 0; j < dB; j++) {
    let s = G[j][j]; for (let k = 0; k < j; k++) s -= L[j][k] * L[j][k];
    L[j][j] = Math.sqrt(Math.max(s, 1e-12));
    for (let i = j + 1; i < dB; i++) { let t = G[i][j]; for (let k = 0; k < j; k++) t -= L[i][k] * L[j][k]; L[i][j] = t / L[j][j]; }
  }
  // solve for each of dA columns: L y = b ; L^T w = y
  const W = Array.from({ length: dB }, () => new Float64Array(dA));
  const y = new Float64Array(dB);
  for (let c = 0; c < dA; c++) {
    for (let i = 0; i < dB; i++) { let s = B[i][c]; for (let k = 0; k < i; k++) s -= L[i][k] * y[k]; y[i] = s / L[i][i]; }
    for (let i = dB - 1; i >= 0; i--) { let s = y[i]; for (let k = i + 1; k < dB; k++) s -= L[k][i] * W[k][c]; W[i][c] = s / L[i][i]; }
  }
  return W;
}
function applyMap(centeredRow, W, dA) { const o = new Float32Array(dA); for (let i = 0; i < centeredRow.length; i++) { const v = centeredRow[i]; if (!v) continue; const wi = W[i]; for (let j = 0; j < dA; j++) o[j] += v * wi[j]; } return o; }
function cosine(a, b) { let s = 0, na = 0, nb = 0; for (let i = 0; i < a.length; i++) { s += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; } return s / (Math.sqrt(na * nb) || 1); }

// non-linear adapter via random ReLU features (kernel-ish, no gradient descent):
// phi(b) = ReLU(R b + bias), then linear ridge phi -> A. Fixed random R/bias make
// this a single stable ridge solve while capturing non-linear structure.
function gaussian(rng) { let u = 0, v = 0; while (!u) u = rng(); while (!v) v = rng(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); }
function makeRandomFeatureMap(dB, dh, rng) {
  const scale = Math.sqrt(2 / dB);
  const R = Array.from({ length: dh }, () => Float32Array.from({ length: dB }, () => gaussian(rng) * scale));
  const bias = Float32Array.from({ length: dh }, () => gaussian(rng) * 0.1);
  return (row) => { const o = new Float32Array(dh); for (let h = 0; h < dh; h++) { let s = bias[h]; const rh = R[h]; for (let i = 0; i < row.length; i++) s += rh[i] * row[i]; o[h] = s > 0 ? s : 0; } return o; };
}

// fit ridge with lambda selected by held-out cosine on an 80/20 anchor split
function fitRidgeCV(featRows, Yc, grid) {
  const n = featRows.length, dA = Yc[0].length;
  const idx = shuffle([...Array(n).keys()]);
  const nVal = Math.max(1, Math.floor(n * 0.2));
  const val = idx.slice(0, nVal), tr = idx.slice(nVal);
  const Xtr = tr.map((i) => featRows[i]), Ytr = tr.map((i) => Yc[i]);
  let best = null;
  for (const lam of grid) {
    const W = fitRidge(Xtr, Ytr, lam);
    let sc = 0; for (const i of val) sc += cosine(applyMap(featRows[i], W, dA), Yc[i]);
    sc /= val.length;
    if (!best || sc > best.sc) best = { sc, lam };
  }
  return { W: fitRidge(featRows, Yc, best.lam), lam: best.lam, cv: best.sc };
}

// retrieval + metrics --------------------------------------------------------
function evalQueries(queryVecs, corpusVecs, corpusPaths, queries, relCountByQuery) {
  let nDCG = 0, succ = 0, recall = 0, mrr = 0;
  const n = corpusVecs.length;
  for (let qi = 0; qi < queries.length; qi++) {
    const q = queryVecs[qi], rel = queries[qi].rel;
    const scored = new Array(n);
    for (let j = 0; j < n; j++) scored[j] = [j, dot(q, corpusVecs[j])];
    scored.sort((a, b) => b[1] - a[1]);
    let dcg = 0, firstHit = 0; const pathsHit = new Set();
    for (let r = 0; r < K; r++) {
      const p = corpusPaths[scored[r][0]];
      if (rel.has(p)) { dcg += 1 / Math.log2(r + 2); if (!firstHit) firstHit = r + 1; pathsHit.add(p); }
    }
    const R = Math.min(K, relCountByQuery[qi]);
    let idcg = 0; for (let r = 0; r < R; r++) idcg += 1 / Math.log2(r + 2);
    nDCG += idcg > 0 ? dcg / idcg : 0;
    if (firstHit) { succ += 1; mrr += 1 / firstHit; }
    recall += pathsHit.size / rel.size;
  }
  const m = queries.length;
  return { nDCG: nDCG / m, succ: succ / m, recall: recall / m, mrr: mrr / m };
}

function pct(x) { return (100 * x).toFixed(1).padStart(5) + '%'; }
function fmt(r) { return `nDCG@10 ${pct(r.nDCG)} | succ@10 ${pct(r.succ)} | recall@10 ${pct(r.recall)} | MRR ${pct(r.mrr)}`; }

async function main() {
  console.log('ANN-9 cross-model RELEVANCE + adapter');
  console.log(`  A (doc/ceiling): ${MODEL_A}`);
  console.log(`  B (query/new):   ${MODEL_B}`);
  console.log(`  ridge grid: [${RIDGE_GRID}]   anchors: ${N_ANCHOR}   feat-dim: ${DH}   seed: ${SEED}\n`);

  const corpus = buildCorpus();
  const corpusTexts = corpus.map((c) => c.text), corpusPaths = corpus.map((c) => c.path);
  const queries = loadQueries();
  // in-domain anchors: corpus passages whose source path is NEVER relevant to any
  // query (zero relevance leakage) -> fill remainder with generic repo prose.
  const relevantPaths = new Set(); for (const q of queries) for (const p of q.rel) relevantPaths.add(p);
  const inDomainPool = shuffle(corpus.filter((c) => !relevantPaths.has(c.path)).map((c) => c.text));
  const genericPool = shuffle(proseSentences(ANCHOR_ROOTS));
  const nIn = Math.min(inDomainPool.length, N_ANCHOR);
  let anchors = inDomainPool.slice(0, nIn).concat(genericPool.slice(0, Math.max(0, N_ANCHOR - nIn)));
  console.log(`corpus passages: ${corpus.length}   queries: ${queries.length}   anchors: ${anchors.length} (in-domain ${nIn}, generic ${anchors.length - nIn})`);
  // sanity: are relevant paths actually present in the corpus?
  const corpusPathSet = new Set(corpusPaths);
  let reachable = 0; const relCountByQuery = queries.map((q) => { let c = 0; for (const p of corpusPaths) if (q.rel.has(p)) c++; if ([...q.rel].some((p) => corpusPathSet.has(p))) reachable++; return c; });
  console.log(`queries whose relevant docs exist in corpus: ${reachable}/${queries.length}\n`);
  if (anchors.length < N_ANCHOR) console.log(`(note: only ${anchors.length} anchors available)`);
  if (process.env.DRY_RUN) { console.log('DRY_RUN: data pipeline OK, exiting before embedding.'); return; }

  const t0 = Date.now();
  const corpusA = (await embedAll(MODEL_A, corpusTexts, 'corpus')).map(normalizeRow);
  const qA = (await embedAll(MODEL_A, queries.map((q) => q.query), 'query-A')).map(normalizeRow);
  const qB = (await embedAll(MODEL_B, queries.map((q) => q.query), 'query-B')).map(normalizeRow);
  const anchA = (await embedAll(MODEL_A, anchors, 'anchor-A')).map(normalizeRow);
  const anchB = (await embedAll(MODEL_B, anchors, 'anchor-B')).map(normalizeRow);
  console.log(`embedding done in ${((Date.now() - t0) / 1000).toFixed(1)}s (dA=${corpusA[0].length} dB=${qB[0].length})\n`);
  const dimsMatch = corpusA[0].length === qB[0].length;

  // (a) ceiling: query A vs corpus A
  const ceiling = evalQueries(qA, corpusA, corpusPaths, queries, relCountByQuery);

  // (b) raw cross: query B vs corpus A (only if dims match)
  const rawCross = dimsMatch ? evalQueries(qB, corpusA, corpusPaths, queries, relCountByQuery) : null;

  // (c) anchor-relative (centered): corpus & query in cosine-to-anchor space
  const meanRelA = meanVec(anchA.map((a) => relRow(a, anchA)));
  const meanRelB = meanVec(anchB.map((b) => relRow(b, anchB)));
  const corpusRel = corpusA.map((v) => normalizeRow(subMean(relRow(v, anchA), meanRelA)));
  const qRel = qB.map((v) => normalizeRow(subMean(relRow(v, anchB), meanRelB)));
  const anchorRel = evalQueries(qRel, corpusRel, corpusPaths, queries, relCountByQuery);

  // (d) adapters: map query_B -> A space using centered anchor pairs
  const mA = meanVec(anchA), mB = meanVec(anchB);
  const dA = corpusA[0].length, dB = anchB[0].length;
  const XcB = anchB.map((b) => subMean(b, mB));   // centered B anchors (adapter input)
  const Yc = anchA.map((a) => subMean(a, mA));     // centered A anchors (adapter target)
  // (d1) linear ridge, lambda by CV
  const lin = fitRidgeCV(XcB, Yc, RIDGE_GRID);
  const qLin = qB.map((b) => { const m = applyMap(subMean(b, mB), lin.W, dA); for (let j = 0; j < dA; j++) m[j] += mA[j]; return normalizeRow(m); });
  const adapterLin = evalQueries(qLin, corpusA, corpusPaths, queries, relCountByQuery);
  // (d2) non-linear random-feature ridge, lambda by CV
  const featMap = makeRandomFeatureMap(dB, DH, rng);
  const XfB = XcB.map(featMap);
  const nl = fitRidgeCV(XfB, Yc, RIDGE_GRID);
  const qNL = qB.map((b) => { const m = applyMap(featMap(subMean(b, mB)), nl.W, dA); for (let j = 0; j < dA; j++) m[j] += mA[j]; return normalizeRow(m); });
  const adapterNL = evalQueries(qNL, corpusA, corpusPaths, queries, relCountByQuery);

  console.log('RESULTS (query model B retrieving against corpus embedded by model A)');
  console.log(`  (a) same-model CEILING (query A)   : ${fmt(ceiling)}`);
  console.log(`  (b) raw cross-model (query B)      : ${dimsMatch ? fmt(rawCross) : 'N/A (dim mismatch — raw comparison impossible)'}`);
  console.log(`  (c) ANN-9 anchor-relative, centered : ${fmt(anchorRel)}`);
  console.log(`  (d1) LINEAR adapter (lambda=${lin.lam}, cv=${pct(lin.cv)}) : ${fmt(adapterLin)}`);
  console.log(`  (d2) NON-LINEAR adapter (lambda=${nl.lam}, cv=${pct(nl.cv)}): ${fmt(adapterNL)}`);

  const recov = (r) => ceiling.nDCG > 0 ? (r.nDCG / ceiling.nDCG) : 0;
  console.log('\nVERDICT (nDCG@10 as fraction of same-model ceiling)');
  if (dimsMatch) console.log(`  raw cross-model recovers ...... ${pct(recov(rawCross))} of ceiling`);
  console.log(`  anchor-relative recovers ...... ${pct(recov(anchorRel))} of ceiling`);
  console.log(`  LINEAR adapter recovers ....... ${pct(recov(adapterLin))} of ceiling`);
  console.log(`  NON-LINEAR adapter recovers ... ${pct(recov(adapterNL))} of ceiling`);
  const best = Math.max(recov(anchorRel), recov(adapterLin), recov(adapterNL), dimsMatch ? recov(rawCross) : 0);
  if (best >= 0.85) console.log(`\n  => PROMISING. A cross-model method recovers >=85% of ceiling on the REAL task.`);
  else if (best >= 0.6) console.log(`\n  => PARTIAL. Best cross-model method recovers ${pct(best)} of ceiling — useful for reranking, not authoritative retrieval.`);
  else console.log(`\n  => WEAK. No cross-model method recovers even 60% of ceiling on the real task.`);
}

main().catch((e) => { console.error(e); process.exit(1); });
