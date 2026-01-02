#!/usr/bin/env python3
"""
Compare ANNPACK search against exact FAISS ground truth on the same embeddings.
Usage:
  python tools/fidelity_gate.py --manifest data/fiqa/fiqa.manifest.json --queries fiqa --sample 200 --k 10 --seed 42
"""
import argparse
import json
import os
import random
import subprocess
import hashlib
import shutil
import sys
from pathlib import Path
import struct

import faiss  # type: ignore
import numpy as np
from beir import util as beir_util  # type: ignore
from beir.datasets.data_loader import GenericDataLoader  # type: ignore
from scipy.stats import kendalltau  # type: ignore
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import ctypes

ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = ROOT / "eval"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True, help="Path to manifest JSON")
    p.add_argument("--queries", default="fiqa", help="Query dataset name (fiqa)")
    p.add_argument("--queries-path", help="Path to pre-downloaded BEIR dataset (skip download)")
    p.add_argument("--sample", type=int, default=200, help="Number of queries to sample")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--lib", help="Path to native annpack dylib (optional, will build if missing)")
    p.add_argument("--probe", type=int, help="nprobe setting for ANN search")
    p.add_argument("--max-scan", type=int, help="max_scan setting (0 = all)")
    p.add_argument("--mode", choices=["default", "exact"], default="default", help="Convenience presets: exact sets probe=n_lists and max_scan=total_vectors")
    p.add_argument("--gt", choices=["doc_embeds", "annpack"], default="doc_embeds", help="Ground truth vectors: doc_embeds (fp32) or annpack (fp16 roundtrip)")
    p.add_argument("--debug-one", type=int, help="Run a single query (index); -1 picks a random query using --seed")
    return p.parse_args()


def compute_native_hash(srcs, flags):
    h = hashlib.sha256()
    for p in srcs:
        h.update(p.read_bytes())
    h.update(" ".join(flags).encode("utf-8"))
    return h.hexdigest()


def build_native(lib_path: Path):
    srcs = [ROOT / "src" / "ann_engine.c", ROOT / "tests" / "io_posix.c"]
    cc = shutil.which("cc") or "cc"
    flags = ["-std=c11", "-O3", "-shared", "-fPIC", "-I", str(ROOT / "include")]
    out_flag = ["-o", str(lib_path)]
    hash_path = lib_path.with_suffix(".dylib.hash")
    src_hash = compute_native_hash(srcs, [cc, *flags, *out_flag])
    if lib_path.exists() and hash_path.exists():
        existing = hash_path.read_text().strip()
        if existing == src_hash:
            print(f"[native] up-to-date {lib_path}")
            return
    cmd = [cc, *flags, *map(str, srcs), *out_flag]
    print(f"[native] rebuilding {lib_path} (hash changed)")
    subprocess.run(cmd, check=True)
    hash_path.write_text(src_hash)


class AnnLib:
    def __init__(self, lib_path: Path):
        build_native(lib_path)
        self.lib = ctypes.CDLL(str(lib_path))
        self.lib.ann_load_index.restype = ctypes.c_void_p
        self.lib.ann_load_index.argtypes = [ctypes.c_char_p]
        self.lib.ann_free_index.restype = None
        self.lib.ann_free_index.argtypes = [ctypes.c_void_p]
        self.lib.ann_search.restype = ctypes.c_int
        self.lib.ann_search.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_void_p,
            ctypes.c_int,
        ]
        self.lib.ann_result_size_bytes.restype = ctypes.c_int
        self.lib.ann_last_error.restype = ctypes.c_char_p
        self.lib.ann_set_probe.argtypes = [ctypes.c_int]
        self.lib.ann_set_max_k.argtypes = [ctypes.c_int]
        if hasattr(self.lib, "ann_set_max_scan"):
            self.lib.ann_set_max_scan.argtypes = [ctypes.c_int]
        else:
            self.lib.ann_set_max_scan = None

        class AnnResult(ctypes.Structure):
            _pack_ = 1
            _fields_ = [("id", ctypes.c_uint64), ("score", ctypes.c_float)]

        self.AnnResult = AnnResult
        self.result_size = self.lib.ann_result_size_bytes()

    def load(self, path: Path):
        return self.lib.ann_load_index(str(path).encode("utf-8"))

    def free(self, ctx):
        if ctx:
            self.lib.ann_free_index(ctx)

    def search(self, ctx, vec: np.ndarray, k: int):
        arr = np.asarray(vec, dtype=np.float32, order="C")
        ptr = arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        out_arr = (self.AnnResult * k)()
        n = self.lib.ann_search(ctx, ptr, ctypes.byref(out_arr), k)
        if n < 0:
            err = self.lib.ann_last_error()
            raise RuntimeError(f"ann_search failed: {err}")
        results = [{"id": out_arr[i].id, "score": out_arr[i].score} for i in range(n)]
        return results


def load_queries(name, seed, sample, path_hint=None):
    if path_hint:
        data_path = Path(path_hint).resolve()
    else:
        cache = ROOT / "data" / name / "beir_cache"
        url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{name}.zip"
        data_path = beir_util.download_and_unzip(url, str(cache))
    if not data_path.exists():
        raise SystemExit(f"Query path not found: {data_path}")
    _, queries, _ = GenericDataLoader(data_folder=str(data_path)).load(split="test")
    ids = sorted(queries.keys())
    random.seed(seed)
    random.shuffle(ids)
    ids = ids[:sample]
    return ids, [queries[i] for i in ids]


def load_annpack_vectors(path: Path):
    with path.open("rb") as f:
        header = f.read(72)
    magic, version, endian, header_size, dim, metric, n_lists, n_vecs, offpos = struct.unpack("<QIIIIIIIQ", header[:44])
    if magic != 0x504E4E41 or header_size != 72:
        raise RuntimeError(f"Bad header in {path}")
    with path.open("rb") as f:
        f.seek(offpos)
        table = f.read(n_lists * 16)
    ids_all = []
    vecs_all = []
    with path.open("rb") as f:
        for i in range(n_lists):
            off, ln = struct.unpack_from("<QQ", table, i * 16)
            if ln < 4:
                continue
            f.seek(off)
            count = struct.unpack("<I", f.read(4))[0]
            if count == 0:
                continue
            ids = np.frombuffer(f.read(count * 8), dtype=np.uint64)
            raw = f.read(count * dim * 2)
            vecs = np.frombuffer(raw, dtype=np.float16).astype(np.float32).reshape(count, dim)
            ids_all.append(ids)
            vecs_all.append(vecs)
    if not ids_all:
        raise RuntimeError(f"No vectors read from {path}")
    ids_out = np.concatenate(ids_all).astype(np.int64, copy=False)
    vecs_out = np.vstack(vecs_all).astype(np.float32, copy=False)
    return ids_out, vecs_out


def overlap(a, b, k):
    a_set = set(a[:k])
    b_set = set(b[:k])
    if not a_set:
        return 0.0
    return len(a_set & b_set) / float(k)


def jaccard(a, b, k):
    a_set = set(a[:k])
    b_set = set(b[:k])
    if not a_set and not b_set:
        return 1.0
    return len(a_set & b_set) / float(len(a_set | b_set) or 1)


def tau_on_intersection(a_ids, b_ids):
    common = [i for i in a_ids if i in b_ids]
    if len(common) < 2:
        return 1.0
    a_pos = {v: idx for idx, v in enumerate(a_ids)}
    b_pos = {v: idx for idx, v in enumerate(b_ids)}
    a_rank = [a_pos[v] for v in common]
    b_rank = [b_pos[v] for v in common]
    val, _ = kendalltau(a_rank, b_rank)
    return float(val) if val == val else 0.0


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    manifest_path = Path(args.manifest).resolve()
    if not manifest_path.exists():
        print(f"Manifest missing: {manifest_path}", file=sys.stderr)
        sys.exit(2)
    manifest = json.loads(manifest_path.read_text())
    model_name = manifest.get("model", "sentence-transformers/all-MiniLM-L6-v2")
    metric = "dot" if manifest.get("metric", 1) == 1 else "cosine"
    normalized = manifest.get("normalized", metric == "cosine")
    n_lists = int(manifest.get("n_lists") or 0)
    total_vectors = int(manifest.get("total_vectors") or 0)
    if args.gt == "annpack":
        all_ids = []
        all_vecs = []
        for shard in manifest["shards"]:
            ids_s, vecs_s = load_annpack_vectors(manifest_path.parent / shard["annpack"])
            all_ids.append(ids_s)
            all_vecs.append(vecs_s)
        doc_ids = np.concatenate(all_ids).astype(np.int64, copy=False)
        doc_emb = np.vstack(all_vecs).astype(np.float32, copy=False)
        if total_vectors <= 0:
            total_vectors = int(doc_emb.shape[0])
    else:
        doc_ids = np.load(manifest_path.parent / "doc_ids.npy").astype(np.int64, copy=False)
        doc_emb = np.load(manifest_path.parent / "doc_embeds.f32.npy")
        doc_emb = np.asarray(doc_emb, dtype=np.float32, order="C")
        if normalized:
            faiss.normalize_L2(doc_emb)
        if total_vectors <= 0:
            total_vectors = int(doc_emb.shape[0])

    faiss_index = faiss.IndexIDMap(faiss.IndexFlatIP(doc_emb.shape[1]))
    faiss_index.add_with_ids(doc_emb, doc_ids)

    lib_path = Path(args.lib) if args.lib else Path("/tmp/annpack_native_eval.dylib")
    annlib = AnnLib(lib_path)
    ctxs = []
    effective_probe = args.probe if args.probe is not None else None
    effective_max_scan = args.max_scan if args.max_scan is not None else None
    if args.mode == "exact":
        effective_probe = n_lists or effective_probe
        effective_max_scan = total_vectors or effective_max_scan
    # Clamp probe/max_scan to safe ranges if provided
    if effective_probe is not None and n_lists > 0:
        effective_probe = max(1, min(int(effective_probe), n_lists))
    if effective_max_scan is not None and total_vectors > 0:
        if int(effective_max_scan) <= 0:
            effective_max_scan = total_vectors
        effective_max_scan = max(1, min(int(effective_max_scan), total_vectors))

    for shard in manifest["shards"]:
        ctx = annlib.load(manifest_path.parent / shard["annpack"])
        if not ctx:
            print(f"Failed to load shard {shard}", file=sys.stderr)
            sys.exit(3)
        ctxs.append(ctx)
    if effective_probe is not None:
        annlib.lib.ann_set_probe(int(effective_probe))
    annlib.lib.ann_set_max_k(args.k)
    if annlib.lib.ann_set_max_scan and effective_max_scan is not None:
        annlib.lib.ann_set_max_scan(int(effective_max_scan))

    default_qpath = (manifest_path.parent / "beir_cache" / args.queries)
    q_path = args.queries_path if args.queries_path else (default_qpath if default_qpath.exists() else None)
    q_ids, q_texts = load_queries(args.queries, args.seed, args.sample, q_path)
    debug_one = args.debug_one
    if debug_one is not None:
        if debug_one < 0:
            random.seed(args.seed)
            debug_one = random.randrange(len(q_ids))
        debug_one = max(0, min(int(debug_one), len(q_ids) - 1))
        q_ids = [q_ids[debug_one]]
        q_texts = [q_texts[debug_one]]
    model = SentenceTransformer(model_name)
    q_vecs = model.encode(q_texts, batch_size=32, convert_to_numpy=True, normalize_embeddings=True)
    q_vecs = np.asarray(q_vecs, dtype=np.float32)
    if normalized:
        faiss.normalize_L2(q_vecs)

    metrics = {"overlap10": [], "overlap20": [], "jaccard10": [], "tau": []}
    debug_payload = None
    for qi, qv in tqdm(list(zip(q_ids, q_vecs)), desc="queries", total=len(q_ids)):
        # FAISS
        faiss_scores, faiss_idxs = faiss_index.search(qv[None, :], args.k * 2)
        faiss_ids = [int(i) for i in faiss_idxs[0]]

        # ANNPACK merged
        shard_results = []
        for ctx in ctxs:
            res = annlib.search(ctx, qv, args.k)
            shard_results.extend(res)
        shard_results.sort(key=lambda r: r["score"], reverse=True)
        ann_ids = [int(r["id"]) for r in shard_results[: args.k]]
        if debug_one is not None:
            debug_payload = {
                "query_id": qi,
                "faiss_topk": faiss_ids[: args.k],
                "annpack_topk": ann_ids[: args.k],
                "probe": effective_probe,
                "max_scan": effective_max_scan,
            }
        metrics["overlap10"].append(overlap(faiss_ids, ann_ids, min(args.k, 10)))
        metrics["overlap20"].append(overlap(faiss_ids, ann_ids, min(args.k, 20)))
        metrics["jaccard10"].append(jaccard(faiss_ids, ann_ids, min(args.k, 10)))
        metrics["tau"].append(tau_on_intersection(faiss_ids[: args.k], ann_ids[: args.k]))

    for ctx in ctxs:
        annlib.free(ctx)

    def avg(xs): return float(np.mean(xs)) if xs else 0.0
    summary = {
        "avg_overlap10": avg(metrics["overlap10"]),
        "min_overlap10": float(np.min(metrics["overlap10"])) if metrics["overlap10"] else 0.0,
        "avg_overlap20": avg(metrics["overlap20"]),
        "avg_jaccard10": avg(metrics["jaccard10"]),
        "avg_tau": avg(metrics["tau"]),
        "n_queries": len(q_ids),
        "k": args.k,
        "metric": metric,
        "normalized": normalized,
        "model": model_name,
        "dataset": args.queries,
        "probe": effective_probe,
        "max_scan": effective_max_scan,
        "mode": args.mode,
        "gt": args.gt,
    }
    thresholds = {"avg_overlap10": 0.90, "min_overlap10": 0.70, "avg_tau": 0.80}
    failures = []
    for key, th in thresholds.items():
        if summary[key] < th:
            failures.append(f"{key} {summary[key]:.3f} < {th}")
    summary["pass"] = len(failures) == 0
    summary["failures"] = failures

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    out_json = EVAL_DIR / f"{args.queries}_fidelity.json"
    out_json.write_text(json.dumps({"summary": summary, "metrics": metrics}, indent=2), encoding="utf-8")
    md = [
        f"# Fidelity Report ({args.queries})",
        "",
        f"- queries: {len(q_ids)} sampled, k={args.k}, seed={args.seed}",
        f"- metric: {metric}, normalized={normalized}",
        f"- model: {model_name}",
        f"- probe: {summary['probe']} | max_scan: {summary['max_scan']} | mode: {args.mode}",
        f"- gt: {args.gt} | vecs: {doc_emb.shape}",
        f"- avg overlap@10: {summary['avg_overlap10']:.3f}",
        f"- min overlap@10: {summary['min_overlap10']:.3f}",
        f"- avg overlap@20: {summary['avg_overlap20']:.3f}",
        f"- avg jaccard@10: {summary['avg_jaccard10']:.3f}",
        f"- avg tau: {summary['avg_tau']:.3f}",
        f"- PASS: {summary['pass']} ({'; '.join(failures) if failures else 'all thresholds met'})",
    ]
    out_md = EVAL_DIR / f"{args.queries}_fidelity.md"
    out_md.write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    if debug_payload:
        print("[debug-one]")
        print(f"query_id={debug_payload['query_id']}")
        print(f"probe={debug_payload['probe']} max_scan={debug_payload['max_scan']}")
        print(f"faiss_topk={debug_payload['faiss_topk']}")
        print(f"annpack_topk={debug_payload['annpack_topk']}")
        print(f"overlap@{args.k}={overlap(debug_payload['faiss_topk'], debug_payload['annpack_topk'], args.k):.3f}")
        sys.exit(0)
    sys.exit(0 if summary["pass"] else 1)


if __name__ == "__main__":
    main()
