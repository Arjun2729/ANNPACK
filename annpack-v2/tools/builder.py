import argparse
import json
import os
import struct
from datetime import datetime, timezone
import numpy as np
import polars as pl
import faiss
from sentence_transformers import SentenceTransformer


def parse_args():
    p = argparse.ArgumentParser(description="ANNPack builder (generic CSV/Parquet/JSON)")
    p.add_argument("--input", required=True, help="Input file (.parquet/.csv/.json)")
    p.add_argument("--text-col", required=True, help="Column to embed")
    p.add_argument("--id-col", help="ID column (int64). Auto-generate if missing.")
    p.add_argument(
        "--meta-cols", nargs="*", default=[], help="Columns to include in metadata (JSONL)"
    )
    p.add_argument("--model", default="all-MiniLM-L6-v2")
    p.add_argument("--output", default="generic_index", help="Output prefix")
    p.add_argument("--output-dir", default=".", help="Output directory (default: current)")
    p.add_argument(
        "--lists", type=int, default=256, help="IVF clusters (use 256-1024 for small shards)"
    )
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-rows", type=int, help="Limit rows for small shards (e.g., 50000)")
    p.add_argument("--shard-size", type=int, default=0, help="Shard size; 0 means single shard")
    p.add_argument("--snippet-chars", type=int, default=240, help="Snippet length for metadata")
    p.add_argument("--include-text", action="store_true", help="Include full text in metadata")
    p.add_argument(
        "--offline-dummy",
        action="store_true",
        help="Use deterministic hash embeddings (no model download)",
    )
    return p.parse_args()


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def load_df(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in (".parquet", ".parq"):
        return pl.read_parquet(path)
    if ext == ".csv":
        return pl.read_csv(path)
    if ext == ".json":
        return pl.read_json(path)
    raise ValueError(f"Unsupported extension: {ext}")


def embed_texts(texts, model_name, batch_size, offline_dummy=False):
    if offline_dummy:
        dim = 384
        vecs = np.zeros((len(texts), dim), dtype=np.float32)
        for i, t in enumerate(texts):
            h = abs(hash(t))
            for d in range(dim):
                h = (h * 1664525 + 1013904223) & 0xFFFFFFFF
                vecs[i, d] = (h / 0xFFFFFFFF) - 0.5
            # L2 normalize
            norm = np.linalg.norm(vecs[i])
            if norm > 0:
                vecs[i] /= norm
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
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    vectors = np.asarray(vectors, dtype=np.float32)
    faiss.normalize_L2(vectors)
    return vectors


def train_ivf(vectors, n_lists):
    dim = vectors.shape[1]
    kmeans = faiss.Kmeans(dim, n_lists, niter=20, verbose=True)
    kmeans.train(vectors)
    _, list_ids = kmeans.index.search(vectors, 1)
    return kmeans.centroids.astype(np.float32), list_ids.flatten()


def write_annpack(prefix, dim, n_lists, vectors, ids, centroids, list_ids):
    fn = f"{prefix}.annpack"
    print(f"[write] {fn}")
    with open(fn, "wb") as f:
        magic = 0x504E4E41
        header_size = 72
        version = 1
        endian = 1
        metric = 1
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

    print(f"[done] {fn} ({os.path.getsize(fn) / (1024 * 1024):.2f} MB)")


def write_meta(prefix, df, ids, text_col, meta_cols):
    fn = f"{prefix}.meta.jsonl"
    print(f"[meta] {fn}")
    cols = meta_cols + [text_col]
    with open(fn, "w", encoding="utf-8") as w:
        for i, row in zip(ids.tolist(), df.select(cols).to_dicts()):
            row["id"] = int(i)
            w.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    args = parse_args()
    df = load_df(args.input)
    if args.text_col not in df.columns:
        raise SystemExit(f"text col '{args.text_col}' not found. cols={df.columns}")
    if args.max_rows:
        df = df.head(args.max_rows)

    out_dir = os.path.abspath(args.output_dir)
    os.makedirs(out_dir, exist_ok=True)

    def slice_df(start, end):
        return df.slice(offset=start, length=end - start)

    def compute_ids(chunk, offset_base=0):
        if args.id_col:
            return chunk[args.id_col].to_numpy().astype(np.int64, copy=False)
        return np.arange(chunk.height, dtype=np.int64) + offset_base

    def write_meta_file(path_prefix, chunk_df, ids):
        meta_cols = list(args.meta_cols)
        snippet_len = max(0, args.snippet_chars)
        with open(path_prefix + ".meta.jsonl", "w", encoding="utf-8") as w:
            for row_id, row in zip(ids.tolist(), chunk_df.to_dicts()):
                out = {"id": int(row_id)}
                for c in meta_cols:
                    if c in row:
                        out[c] = row[c]
                txt = str(row[args.text_col])
                out["snippet"] = txt[:snippet_len]
                if args.include_text:
                    out[args.text_col] = txt
                w.write(json.dumps(out, ensure_ascii=False) + "\n")

    def build_shard(chunk_df, shard_idx, id_offset):
        ids = compute_ids(chunk_df, offset_base=id_offset)
        texts = [str(t) for t in chunk_df[args.text_col].to_list()]
        print(f"[embed] shard {shard_idx} rows={len(texts)}")
        vectors = embed_texts(texts, args.model, args.batch_size, offline_dummy=args.offline_dummy)
        if vectors.shape[0] != len(ids):
            raise SystemExit("ids length mismatch vectors")
        centroids, list_ids = train_ivf(vectors, args.lists)
        dim = vectors.shape[1]
        shard_name = f"{args.output}.shard{shard_idx:03d}"
        prefix_path = os.path.join(out_dir, shard_name)
        write_annpack(prefix_path, dim, args.lists, vectors, ids, centroids, list_ids)
        write_meta_file(prefix_path, chunk_df, ids)
        return {
            "name": shard_name,
            "annpack": os.path.basename(prefix_path + ".annpack"),
            "meta": os.path.basename(prefix_path + ".meta.jsonl"),
            "n_vectors": int(vectors.shape[0]),
            "id_min": int(ids.min()) if len(ids) > 0 else None,
            "id_max": int(ids.max()) if len(ids) > 0 else None,
            "dim": int(dim),
            "metric": 1,
            "n_lists": int(args.lists),
        }

    if args.shard_size and args.shard_size > 0:
        shard_entries = []
        total = df.height
        offset = 0
        shard_idx = 0
        while offset < total:
            end = min(offset + args.shard_size, total)
            chunk = slice_df(offset, end)
            shard_entry = build_shard(chunk, shard_idx, offset if not args.id_col else 0)
            shard_entries.append(shard_entry)
            shard_idx += 1
            offset = end
        manifest = {
            "version": 1,
            "created_at": now_iso(),
            "model": args.model,
            "dim": shard_entries[0]["dim"] if shard_entries else None,
            "metric": shard_entries[0]["metric"] if shard_entries else 1,
            "n_lists": shard_entries[0]["n_lists"] if shard_entries else args.lists,
            "total_vectors": int(sum(se["n_vectors"] for se in shard_entries)),
            "shard_size": args.shard_size,
            "shards": shard_entries,
        }
        manifest_path = os.path.join(out_dir, f"{args.output}.manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as mf:
            json.dump(manifest, mf, ensure_ascii=False, indent=2)
        print(f"[done] Wrote manifest with {len(shard_entries)} shards to {manifest_path}")
    else:
        ids = compute_ids(df, offset_base=0)
        texts = [str(t) for t in df[args.text_col].to_list()]
        print("[embed] generating embeddings...")
        vectors = embed_texts(texts, args.model, args.batch_size, offline_dummy=args.offline_dummy)
        if vectors.shape[0] != len(ids):
            raise SystemExit("ids length mismatch vectors")
        centroids, list_ids = train_ivf(vectors, args.lists)
        dim = vectors.shape[1]
        out_prefix = os.path.join(out_dir, args.output)
        write_annpack(out_prefix, dim, args.lists, vectors, ids, centroids, list_ids)
        write_meta_file(out_prefix, df, ids)
        # optional manifest single shard
        manifest = {
            "version": 1,
            "created_at": now_iso(),
            "model": args.model,
            "dim": int(dim),
            "metric": 1,
            "n_lists": args.lists,
            "total_vectors": int(vectors.shape[0]),
            "shard_size": 0,
            "shards": [
                {
                    "name": f"{args.output}.shard000",
                    "annpack": os.path.basename(out_prefix + ".annpack"),
                    "meta": os.path.basename(out_prefix + ".meta.jsonl"),
                    "n_vectors": int(vectors.shape[0]),
                }
            ],
        }
        manifest_path = os.path.join(out_dir, f"{args.output}.manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as mf:
            json.dump(manifest, mf, ensure_ascii=False, indent=2)
        print("[done] ANNPack build complete (single shard)")


if __name__ == "__main__":
    main()
