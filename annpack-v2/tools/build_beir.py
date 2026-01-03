#!/usr/bin/env python3
"""
Build a BEIR corpus (default: FiQA) into ANNPACK shards plus a doc embedding dump.
Deterministic seeds and snippet-only metadata. Designed for macOS with modest RAM.
"""

import argparse
import json
import random
import sys
import struct
from datetime import datetime, timezone
from pathlib import Path

import faiss  # type: ignore
import numpy as np
from beir import util as beir_util  # type: ignore
from beir.datasets.data_loader import GenericDataLoader  # type: ignore
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="fiqa", help="BEIR dataset name (fiqa)")
    p.add_argument("--dataset-path", help="Pre-downloaded BEIR folder (skips download)")
    p.add_argument("--outdir", default="data/fiqa", help="Output directory")
    p.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--shard-size", type=int, default=50000)
    p.add_argument("--lists", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--snippet-chars", type=int, default=240)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-docs", type=int, help="Cap documents for quick runs")
    p.add_argument(
        "--offline-dummy",
        action="store_true",
        help="Deterministic hash embeddings (no model download)",
    )
    return p.parse_args()


def embed_texts(texts, model_name, batch_size, offline_dummy=False):
    if offline_dummy:
        dim = 384
        rng = np.random.default_rng(12345)
        vecs = np.zeros((len(texts), dim), dtype=np.float32)
        for i, t in enumerate(texts):
            h = abs(hash(t))
            rng.bit_generator.state["state"]["state"] = h & ((1 << 64) - 1)
            v = rng.standard_normal(dim)
            v /= np.linalg.norm(v) + 1e-9
            vecs[i] = v.astype(np.float32)
        return vecs
    device = None
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            device = "cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = "mps"
    except Exception:
        device = None
    model = SentenceTransformer(model_name, device=device)
    vecs = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    vecs = np.asarray(vecs, dtype=np.float32)
    faiss.normalize_L2(vecs)
    return vecs


def train_ivf(vectors, n_lists):
    dim = vectors.shape[1]
    kmeans = faiss.Kmeans(dim, n_lists, niter=20, verbose=True)
    kmeans.train(vectors)
    _, list_ids = kmeans.index.search(vectors, 1)
    return kmeans.centroids.astype(np.float32), list_ids.flatten()


def write_annpack(prefix_path: Path, dim, n_lists, vectors, ids, centroids, list_ids):
    fn = prefix_path.with_suffix(".annpack")
    print(f"[write] {fn}")
    with fn.open("wb") as f:
        magic = 0x504E4E41
        header_size = 72
        version = 1
        endian = 1
        metric = 1  # dot
        f.write(
            struct.pack(
                "<QIIIIIIIQ",
                magic,
                version,
                endian,
                header_size,
                dim,
                metric,
                n_lists,
                vectors.shape[0],
                0,
            )
        )
        if f.tell() < header_size:
            f.write(b"\x00" * (header_size - f.tell()))
        f.write(centroids.tobytes())
        order = np.argsort(list_ids)
        vecs_sorted = vectors[order]
        ids_sorted = ids[order]
        counts = np.bincount(list_ids, minlength=n_lists)
        starts = np.concatenate(([0], np.cumsum(counts[:-1])))
        list_offsets = []
        list_lengths = []
        for i in range(n_lists):
            count = int(counts[i])
            start = int(starts[i])
            offset = f.tell()
            f.write(struct.pack("<I", count))
            if count > 0:
                sl = slice(start, start + count)
                f.write(ids_sorted[sl].tobytes())
                f.write(vecs_sorted[sl].astype(np.float16).tobytes())
            list_offsets.append(offset)
            list_lengths.append(f.tell() - offset)
        table_pos = f.tell()
        for off, length in zip(list_offsets, list_lengths):
            f.write(struct.pack("<QQ", off, length))
        f.seek(36)
        f.write(struct.pack("<Q", table_pos))
    return fn


def write_meta(path_prefix: Path, ids, texts, snippet_chars):
    meta_fn = path_prefix.with_suffix(".meta.jsonl")
    with meta_fn.open("w", encoding="utf-8") as w:
        for doc_id, text in zip(ids.tolist(), texts):
            obj = {"id": int(doc_id), "snippet": text[:snippet_chars]}
            w.write(json.dumps(obj, ensure_ascii=False) + "\n")
    return meta_fn


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    if args.dataset not in ("fiqa",):
        print("Only fiqa supported now.", file=sys.stderr)
        sys.exit(1)

    if args.dataset_path:
        data_path = Path(args.dataset_path).resolve()
        if not data_path.exists():
            print(f"dataset-path not found: {data_path}", file=sys.stderr)
            sys.exit(2)
    else:
        cache_dir = outdir / "beir_cache"
        url = (
            f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{args.dataset}.zip"
        )
        print(f"[download] {url}")
        data_path = beir_util.download_and_unzip(url, str(cache_dir))
    corpus, _, _ = GenericDataLoader(data_folder=data_path).load(split="train")

    doc_ids = list(corpus.keys())
    if args.max_docs:
        doc_ids = doc_ids[: args.max_docs]
    texts = [corpus[i]["text"] for i in doc_ids]
    print(f"[info] docs={len(doc_ids)}")
    vectors = embed_texts(texts, args.model, args.batch_size, offline_dummy=args.offline_dummy)
    ids_np = np.asarray(
        [int(i) if str(i).isdigit() else idx for idx, i in enumerate(doc_ids)], dtype=np.int64
    )
    if vectors.shape[0] != ids_np.shape[0]:
        print("Embedding/doc ID mismatch", file=sys.stderr)
        sys.exit(2)
    shards = []
    total = len(doc_ids)
    offset = 0
    shard_idx = 0
    MIN_POINTS_PER_CENTROID = 39
    while offset < total:
        end = min(offset + args.shard_size, total)
        sl = slice(offset, end)
        shard_name = f"{args.dataset}_shard{shard_idx:03d}"
        prefix = outdir / shard_name
        n_points = int(end - offset)
        max_lists_by_points = max(1, n_points // MIN_POINTS_PER_CENTROID)
        n_lists = min(args.lists, max_lists_by_points, n_points)
        if n_lists != args.lists:
            print(
                f"[warn] shard {shard_idx} has {n_points} points; using n_lists={n_lists} (requested {args.lists})"
            )
            if n_points < args.lists:
                print(
                    f"[warn] last shard smaller than n_lists: rows={n_points}, n_lists={args.lists}"
                )
        centroids, list_ids = train_ivf(vectors[sl], n_lists)
        ann_fn = write_annpack(
            prefix, vectors.shape[1], n_lists, vectors[sl], ids_np[sl], centroids, list_ids
        )
        meta_fn = write_meta(prefix, ids_np[sl], texts[sl], args.snippet_chars)
        shards.append(
            {
                "name": shard_name,
                "annpack": ann_fn.name,
                "meta": meta_fn.name,
                "n_vectors": int(end - offset),
                "id_min": int(ids_np[sl].min()),
                "id_max": int(ids_np[sl].max()),
                "n_lists": int(n_lists),
            }
        )
        offset = end
        shard_idx += 1

    manifest = {
        "version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "dim": int(vectors.shape[1]),
        "metric": 1,
        "n_lists": max(s["n_lists"] for s in shards) if shards else args.lists,
        "total_vectors": int(len(doc_ids)),
        "shard_size": args.shard_size,
        "normalized": True,
        "shards": shards,
    }
    man_path = outdir / f"{args.dataset}.manifest.json"
    man_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    np.save(outdir / "doc_ids.npy", ids_np)
    np.save(outdir / "doc_embeds.f32.npy", vectors.astype(np.float32))
    (outdir / "fidelity_meta.json").write_text(
        json.dumps({"metric": "dot", "normalized": True, "dataset": args.dataset}, indent=2),
        encoding="utf-8",
    )
    print(f"[done] manifest={man_path} shards={len(shards)} embeds={vectors.shape}")


if __name__ == "__main__":
    main()
