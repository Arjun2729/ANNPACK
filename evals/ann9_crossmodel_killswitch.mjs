// ANN-9 cross-model kill-switch.
//
// Question this answers (go/no-go): on TWO independently-trained REAL embedding
// models, do anchor-relative coordinates recover the cross-model nearest-neighbor
// structure that raw cross-model comparison destroys?
//
// This is the label-free geometric core of the ANN-9 thesis ("any model can embed
// the anchors and compute comparable coordinates"). It deliberately mirrors the
// earlier SYNTHETIC anchor sweep (16/32/64/128/256) so the numbers are directly
// comparable — but here both spaces come from real models, not a rotation matrix.
//
// Metric: cross-model retrieval identity. Passage i embedded by model B must, in
// anchor-relative space, retrieve the SAME passage i embedded by model A, against
// a database of all eval passages' model-A relative coords. top-1 and hit@5.
//
// Variants, in increasing strength of calibration:
//   raw        — cosine over uncentered relative coords (what the ANN-9 spec/
//                reader implement today).
//   centered   — per-model mean-centering before comparison (the calibration the
//                reviewers flagged: two models' coordinate distributions are not
//                identically distributed). Mean is fit on a held-out SUPERVISION
//                set, not the eval set, so this is inductive, not transductive.
//   procrustes — the anchor texts are the SAME strings embedded by both models,
//                so a disjoint supervision set gives paired points in both
//                relative spaces. Fit the closest orthogonal rotation W (K x K,
//                Kabsch/orthogonal-Procrustes via SVD) mapping model B's centered
//                relative coords onto model A's, on the supervision set, then
//                apply W to the eval set. This is the strongest calibration that
//                doesn't require labels — if it can't clear the bar, no amount of
//                anchor-selection tuning will save the cross-model bridge.
//
// Baselines: same-model relative ceiling, naive raw cross-model (impossible when
// dims differ), and a shuffled control that must collapse to ~1/N (sanity that
// the metric itself isn't buggy).
//
// Usage:  node ann9_crossmodel_killswitch.mjs
//   env:  MODEL_A, MODEL_B, N_EVAL, N_POOL, N_SUPER, SEED

import { pipeline } from '@huggingface/transformers';
import fs from 'node:fs';
import path from 'node:path';

// --- config -----------------------------------------------------------------
// Default to an INCOMPATIBLE pair: different family AND different dimension, so
// raw cross-model comparison is impossible and the anchor bridge is actually
// exercised. (A compatible same-dim pair saturates every metric at ~100% and
// tests nothing — confirmed in an earlier run with MiniLM/mxbai.)
const MODEL_A = process.env.MODEL_A || 'Xenova/all-MiniLM-L6-v2'; // 384d, MiniLM / sentence-transformers
const MODEL_B = process.env.MODEL_B || 'Xenova/gte-base';        // 768d, Alibaba GTE (different family + dim)
const N_EVAL = Number(process.env.N_EVAL || 600);
const N_POOL = Number(process.env.N_POOL || 320);   // anchor pool, disjoint from eval; must be >= max K
const N_SUPER = Number(process.env.N_SUPER || 400); // Procrustes supervision pairs, disjoint from eval+pool; must be > max K for a well-conditioned fit
const ANCHOR_KS = [16, 32, 64, 128, 256];
const SEED = Number(process.env.SEED || 1234);
// GO if anchor-relative cross-model top-1 reaches this at a practical K (<=128):
const GO_TOP1 = 0.80;
const GO_K = 128;

// --- deterministic rng ------------------------------------------------------
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
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
  return arr;
}

// --- corpus: real technical prose from the repo -----------------------------
function gatherTexts() {
  const roots = [
    '/Users/anika/annpackv2/target/fastapi-eval/fastapi-src/docs/en/docs',
    '/Users/anika/annpackv2/target/fastapi-eval/fastapi-src',
    '/Users/anika/annpackv2/spec',
    '/Users/anika/annpackv2',
  ];
  const seen = new Set();
  const out = [];
  const walk = (dir, depth) => {
    if (depth > 6 || out.length > 40000) return;
    let entries;
    try { entries = fs.readdirSync(dir, { withFileTypes: true }); } catch { return; }
    for (const e of entries) {
      if (e.name.startsWith('.') || e.name === 'node_modules' || e.name === 'target') continue;
      const p = path.join(dir, e.name);
      if (e.isDirectory()) walk(p, depth + 1);
      else if (/\.(md|py|rs|txt)$/.test(e.name)) {
        let txt;
        try { txt = fs.readFileSync(p, 'utf8'); } catch { continue; }
        for (const raw of txt.split(/(?<=[.!?])\s+|\n/)) {
          const s = raw.trim().replace(/\s+/g, ' ');
          // prose filter: real sentences, not code / markup / boilerplate
          if (s.length < 45 || s.length > 260) continue;
          const words = s.split(' ');
          if (words.length < 8) continue;
          const alpha = (s.match(/[A-Za-z]/g) || []).length;
          if (alpha / s.length < 0.72) continue;           // reject code-heavy lines
          if (/[{}<>=|#*`\[\]]/.test(s)) continue;          // reject markup/code punctuation
          if (!/^[A-Z"']/.test(s)) continue;                // sentence-like start
          const key = s.toLowerCase();
          if (seen.has(key)) continue;
          seen.add(key);
          out.push(s);
        }
      }
    }
  };
  for (const r of roots) walk(r, 0);
  return out;
}

// --- embedding --------------------------------------------------------------
async function embedAll(modelId, texts) {
  const opts = {};
  if (modelId.includes('mxbai')) opts.dtype = 'q8';
  const extractor = await pipeline('feature-extraction', modelId, opts);
  const vecs = [];
  const BATCH = 32;
  for (let i = 0; i < texts.length; i += BATCH) {
    const batch = texts.slice(i, i + BATCH);
    const t = await extractor(batch, { pooling: 'mean', normalize: true });
    const d = t.dims[t.dims.length - 1];
    for (let r = 0; r < batch.length; r++) {
      vecs.push(Float32Array.from(t.data.slice(r * d, (r + 1) * d)));
    }
    process.stdout.write(`\r  ${modelId}: embedded ${Math.min(i + BATCH, texts.length)}/${texts.length}   `);
  }
  process.stdout.write('\n');
  return vecs; // array of unit-norm Float32Array
}

// --- linear algebra -----------------------------------------------------------
function dot(a, b) { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; }
function relRow(vec, anchors) { // cosine to each anchor (all unit-norm) => dot
  const r = new Float32Array(anchors.length);
  for (let k = 0; k < anchors.length; k++) r[k] = dot(vec, anchors[k]);
  return r;
}
function computeMean(rows) {
  const n = rows.length, d = rows[0].length;
  const mean = new Float64Array(d);
  for (const row of rows) for (let j = 0; j < d; j++) mean[j] += row[j];
  for (let j = 0; j < d; j++) mean[j] /= n;
  return mean;
}
function applyCenter(rows, mean) { // subtract given mean, then unit-normalize
  return rows.map((row) => {
    const d = row.length;
    const c = new Float32Array(d);
    let nrm = 0;
    for (let j = 0; j < d; j++) { c[j] = row[j] - mean[j]; nrm += c[j] * c[j]; }
    nrm = Math.sqrt(nrm) || 1;
    for (let j = 0; j < d; j++) c[j] /= nrm;
    return c;
  });
}
function unit(rows) {
  return rows.map((row) => {
    let nrm = 0; for (let j = 0; j < row.length; j++) nrm += row[j] * row[j];
    nrm = Math.sqrt(nrm) || 1;
    const u = new Float32Array(row.length);
    for (let j = 0; j < row.length; j++) u[j] = row[j] / nrm;
    return u;
  });
}

// retrieval identity: query rows Q (model B) vs database rows D (model A); success = i retrieves i
function identity(Q, D) {
  const n = Q.length;
  let top1 = 0, hit5 = 0;
  for (let i = 0; i < n; i++) {
    const scores = new Array(n);
    for (let j = 0; j < n; j++) scores[j] = [j, dot(Q[i], D[j])];
    scores.sort((x, y) => y[1] - x[1]);
    if (scores[0][0] === i) top1++;
    for (let r = 0; r < 5; r++) if (scores[r][0] === i) { hit5++; break; }
  }
  return { top1: top1 / n, hit5: hit5 / n };
}

// farthest-point sampling in model-A anchor space for a "diverse" anchor set
function diverseIdx(poolVecs, k) {
  const n = poolVecs.length;
  const picked = [Math.floor(rng() * n)];
  const minDist = new Float64Array(n).fill(Infinity);
  while (picked.length < k) {
    const last = poolVecs[picked[picked.length - 1]];
    let best = -1, bestD = -Infinity;
    for (let i = 0; i < n; i++) {
      const d = 1 - dot(poolVecs[i], last); // cosine distance
      if (d < minDist[i]) minDist[i] = d;
      if (!picked.includes(i) && minDist[i] > bestD) { bestD = minDist[i]; best = i; }
    }
    picked.push(best);
  }
  return picked;
}

// --- orthogonal Procrustes (Kabsch) via Jacobi eigendecomposition -----------
// Solves: given paired rows X (S x K), Y (S x K), find orthogonal W (K x K)
// minimizing ||X W - Y||_F. Closed form: M = X^T Y, SVD M = U Sigma V^T,
// W = U V^T. No off-the-shelf linalg here, so SVD is derived from a symmetric
// eigendecomposition of M^T M via cyclic Jacobi rotations (standard, stable
// for the K <= 256 sizes used here).
function matTMat(X, Y, K) { // X^T Y for X, Y: arrays of S row-vectors, each length K -> K x K
  const out = Array.from({ length: K }, () => new Float64Array(K));
  for (let s = 0; s < X.length; s++) {
    const xr = X[s], yr = Y[s];
    for (let a = 0; a < K; a++) {
      const xa = xr[a];
      if (xa === 0) continue;
      const row = out[a];
      for (let b = 0; b < K; b++) row[b] += xa * yr[b];
    }
  }
  return out;
}
function jacobiEigenSymmetric(Ain, K, maxSweeps = 60, tol = 1e-10) {
  const A = Ain.map((row) => Float64Array.from(row));
  const V = Array.from({ length: K }, (_, i) => {
    const r = new Float64Array(K); r[i] = 1; return r;
  });
  for (let sweep = 0; sweep < maxSweeps; sweep++) {
    let off = 0;
    for (let p = 0; p < K; p++) for (let q = p + 1; q < K; q++) off += A[p][q] * A[p][q];
    if (off < tol) break;
    for (let p = 0; p < K; p++) {
      for (let q = p + 1; q < K; q++) {
        const apq = A[p][q];
        if (Math.abs(apq) < 1e-14) continue;
        const theta = (A[q][q] - A[p][p]) / (2 * apq);
        const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
        const c = 1 / Math.sqrt(t * t + 1);
        const s = t * c;
        const app = A[p][p], aqq = A[q][q];
        A[p][p] = app - t * apq;
        A[q][q] = aqq + t * apq;
        A[p][q] = 0; A[q][p] = 0;
        for (let k = 0; k < K; k++) {
          if (k === p || k === q) continue;
          const akp = A[k][p], akq = A[k][q];
          A[k][p] = c * akp - s * akq; A[p][k] = A[k][p];
          A[k][q] = s * akp + c * akq; A[q][k] = A[k][q];
        }
        for (let k = 0; k < K; k++) {
          const vkp = V[k][p], vkq = V[k][q];
          V[k][p] = c * vkp - s * vkq;
          V[k][q] = s * vkp + c * vkq;
        }
      }
    }
  }
  const values = new Float64Array(K);
  for (let i = 0; i < K; i++) values[i] = A[i][i];
  // columns of V are eigenvectors; extract as array of column vectors
  const vectors = Array.from({ length: K }, (_, i) => Float64Array.from({ length: K }, (_, k) => V[k][i]));
  return { values, vectors };
}
function fitProcrustes(X, Y, K) { // X, Y: paired S-row arrays, length K each -> W (K x K, array of rows)
  const M = matTMat(X, Y, K);          // K x K
  const MtM = matTMat(M, M, K);        // K x K, symmetric PSD
  const { values, vectors } = jacobiEigenSymmetric(MtM, K);
  const order = [...values.keys()].sort((a, b) => values[b] - values[a]);
  const sigmas = order.map((i) => Math.sqrt(Math.max(values[i], 0)));
  const V = order.map((i) => vectors[i]); // right singular vectors, descending sigma
  const U = V.map((v, idx) => {
    const sigma = sigmas[idx];
    const u = new Float64Array(K);
    for (let a = 0; a < K; a++) {
      let sum = 0;
      const row = M[a];
      for (let b = 0; b < K; b++) sum += row[b] * v[b];
      u[a] = sigma > 1e-8 ? sum / sigma : v[a]; // degenerate fallback for ~0 singular values
    }
    return u;
  });
  // W = U V^T
  const W = Array.from({ length: K }, () => new Float64Array(K));
  for (let i = 0; i < K; i++) {
    const ui = U[i], vi = V[i];
    for (let a = 0; a < K; a++) {
      const ua = ui[a];
      if (ua === 0) continue;
      const row = W[a];
      for (let b = 0; b < K; b++) row[b] += ua * vi[b];
    }
  }
  return W;
}
function applyW(rows, W, K) {
  return rows.map((row) => {
    const out = new Float32Array(K);
    for (let b = 0; b < K; b++) {
      let sum = 0;
      for (let a = 0; a < K; a++) sum += row[a] * W[a][b];
      out[b] = sum;
    }
    return out;
  });
}

// --- main -------------------------------------------------------------------
function pct(x) { return (100 * x).toFixed(1).padStart(5) + '%'; }

async function main() {
  console.log('ANN-9 cross-model kill-switch');
  console.log(`  model A: ${MODEL_A}`);
  console.log(`  model B: ${MODEL_B}`);
  console.log(`  eval passages: ${N_EVAL}   anchor pool: ${N_POOL}   supervision (procrustes): ${N_SUPER}   seed: ${SEED}\n`);

  let texts = gatherTexts();
  console.log(`gathered ${texts.length} real prose sentences from repo`);
  if (texts.length < N_EVAL + N_POOL + N_SUPER) {
    console.error(`FATAL: need ${N_EVAL + N_POOL + N_SUPER} texts, only found ${texts.length}`);
    process.exit(2);
  }
  shuffle(texts);
  const evalTexts = texts.slice(0, N_EVAL);
  const poolTexts = texts.slice(N_EVAL, N_EVAL + N_POOL);
  const superTexts = texts.slice(N_EVAL + N_POOL, N_EVAL + N_POOL + N_SUPER); // disjoint from eval + anchor pool

  const t0 = Date.now();
  console.log('\nembedding eval / anchor-pool / supervision sets with both models...');
  const A_eval = await embedAll(MODEL_A, evalTexts);
  const B_eval = await embedAll(MODEL_B, evalTexts);
  const A_pool = await embedAll(MODEL_A, poolTexts);
  const B_pool = await embedAll(MODEL_B, poolTexts);
  const A_sup = await embedAll(MODEL_A, superTexts);
  const B_sup = await embedAll(MODEL_B, superTexts);
  const embedMs = Date.now() - t0;
  console.log(`embedding done in ${(embedMs / 1000).toFixed(1)}s (dims A=${A_eval[0].length} B=${B_eval[0].length})\n`);

  // baselines --------------------------------------------------------------
  const dimsMatch = A_eval[0].length === B_eval[0].length;
  const rawCross = dimsMatch ? identity(B_eval, A_eval) : null;
  const permA = shuffle([...Array(A_eval.length).keys()]);
  const ctrl = identity(B_eval, dimsMatch ? permA.map((k) => A_eval[k]) : B_eval.slice().sort(() => rng() - 0.5));
  console.log('BASELINES');
  if (dimsMatch) {
    console.log(`  naive raw cross-model:        top1 ${pct(rawCross.top1)}   hit@5 ${pct(rawCross.hit5)}   <- what anchors must beat`);
  } else {
    console.log(`  naive raw cross-model:        N/A (dim ${A_eval[0].length} vs ${B_eval[0].length}) <- raw comparison IMPOSSIBLE; anchors are the only bridge`);
  }
  console.log(`  shuffled control (~1/N sanity): top1 ${pct(ctrl.top1)}   (expect ~${pct(1 / N_EVAL)}; high => metric bug)`);

  const diverseOrder = diverseIdx(A_pool, N_POOL);
  const randomOrder = shuffle([...Array(N_POOL).keys()]);

  const rows = [];
  for (const sel of ['random', 'diverse']) {
    const order = sel === 'random' ? randomOrder : diverseOrder;
    for (const K of ANCHOR_KS) {
      const idx = order.slice(0, K);
      const A_anch = idx.map((i) => A_pool[i]);
      const B_anch = idx.map((i) => B_pool[i]);

      const RA = evalTexts.map((_, i) => relRow(A_eval[i], A_anch));
      const RB = evalTexts.map((_, i) => relRow(B_eval[i], B_anch));
      const SA = superTexts.map((_, i) => relRow(A_sup[i], A_anch));
      const SB = superTexts.map((_, i) => relRow(B_sup[i], B_anch));

      const rawX = identity(unit(RB), unit(RA));
      const sameModel = identity(unit(RB), unit(RB));

      // centering: mean fit on the held-out supervision set only (inductive,
      // not the eval set itself — fixes the transductive caveat from run 1).
      const meanA = computeMean(SA), meanB = computeMean(SB);
      const CRA = applyCenter(RA, meanA), CRB = applyCenter(RB, meanB);
      const CSA = applyCenter(SA, meanA), CSB = applyCenter(SB, meanB);
      const cX = identity(CRB, CRA);

      // procrustes: fit rotation on supervision pairs, apply to eval.
      const W = fitProcrustes(CSB, CSA, K);
      const PRB = unit(applyW(CRB, W, K));
      const pX = identity(PRB, unit(CRA));

      rows.push({
        sel, K,
        rawTop1: rawX.top1, rawHit5: rawX.hit5,
        cTop1: cX.top1, cHit5: cX.hit5,
        pTop1: pX.top1, pHit5: pX.hit5,
        ceil: sameModel.top1,
      });
      process.stdout.write(`  fit+scored sel=${sel} K=${K}\n`);
    }
  }

  console.log('\nANCHOR-RELATIVE CROSS-MODEL RETRIEVAL (real model pair)');
  console.log('  sel      K   | raw top1  h@5  | ctr top1  h@5  | proc top1  h@5  | same-model ceil');
  console.log('  ---------------+------------------+------------------+-------------------+----------------');
  for (const r of rows) {
    console.log(
      `  ${r.sel.padEnd(7)} ${String(r.K).padStart(3)} |  ${pct(r.rawTop1)} ${pct(r.rawHit5)} |  ${pct(r.cTop1)} ${pct(r.cHit5)} |  ${pct(r.pTop1)}  ${pct(r.pHit5)} |  ${pct(r.ceil)}`,
    );
  }

  // verdict ----------------------------------------------------------------
  const bestAt = (maxK) => rows.filter((r) => r.K <= maxK).reduce(
    (m, r) => Math.max(m, r.rawTop1, r.cTop1, r.pTop1), 0,
  );
  const best = bestAt(GO_K);
  const best256 = bestAt(256);
  const bestProc = rows.reduce((m, r) => Math.max(m, r.pTop1), 0);
  console.log('\nVERDICT');
  console.log(`  raw cross-model baseline ......... ${dimsMatch ? pct(rawCross.top1) + ' top1' : 'N/A (dim mismatch — raw impossible)'}`);
  console.log(`  shuffled-control sanity .......... ${pct(ctrl.top1)} top1 (must be ~${pct(1 / N_EVAL)})`);
  console.log(`  best (raw/centered/procrustes) @K<=${GO_K} .. ${pct(best)} top1`);
  console.log(`  best (raw/centered/procrustes) @K<=256 ... ${pct(best256)} top1`);
  console.log(`  best procrustes (any K) .......... ${pct(bestProc)} top1`);
  console.log(`  GO threshold ..................... ${pct(GO_TOP1)} top1 at K<=${GO_K}`);
  if (best >= GO_TOP1) {
    console.log(`\n  => GO. Anchor calibration recovers cross-model structure on a real pair at practical K.`);
  } else if (best256 >= GO_TOP1) {
    console.log(`\n  => MARGINAL. Only reaches threshold at high K (>${GO_K}). Needs anchor/calibration work before infra.`);
  } else {
    console.log(`\n  => NO-GO on this pair. Even Procrustes-calibrated anchor-relative does not recover cross-model retrieval; do not promote to infrastructure.`);
  }
}

main().catch((e) => { console.error(e); process.exit(1); });
