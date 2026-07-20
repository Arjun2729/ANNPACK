#!/usr/bin/env python3
import argparse
import json
import os
import struct
import numpy as np
import polars as pl
import faiss


def load_df(path: str) -> pl.DataFrame:
    if path.endswith('.parquet'):
        return pl.read_parquet(path)
    if path.endswith('.csv'):
        return pl.read_csv(path)
    if path.endswith('.json'):
        return pl.read_json(path)
    raise ValueError(f"Unsupported file type: {path}")


def embed_texts(texts, model_name: str, batch_size: int):
    from sentence_transformers import SentenceTransformer
    device = 'mps' if hasattr(__import__('torch'), 'backends') and getattr(__import__('torch').backends, 'mps', None) and __import__('torch').backends.mps.is_available() else 'cpu'
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


def train_ivf(vectors: np.ndarray, n_lists: int):
    dim = vectors.shape[1]
    kmeans = faiss.Kmeans(dim, n_lists, niter=20, verbose=True)
    kmeans.train(vectors)
    _, list_ids = kmeans.index.search(vectors, 1)
    return np.asarray(kmeans.centroids, dtype=np.float32), list_ids.reshape(-1)


def write_annpack(prefix: str, dim: int, n_lists: int, vectors: np.ndarray, ids: np.ndarray, centroids: np.ndarray, list_ids: np.ndarray):
    magic = 0x504E4E41
    version = 1
    endian = 1
    header_size = 72
    metric = 1
    n_vectors = vectors.shape[0]

    path = f"{prefix}.annpack"
    with open(path, 'wb') as f:
        header = struct.pack('<QIIIIIIIQ', magic, version, endian, header_size, dim, metric, n_lists, n_vectors, 0)
        f.write(header)
        f.write(b'\x00' * (header_size - len(header)))

        f.write(centroids.astype(np.float32).tobytes())

        counts = np.bincount(list_ids, minlength=n_lists)
        starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
        order = np.argsort(list_ids)
        vecs_sorted = vectors[order]
        ids_sorted = ids[order]

        offsets = []
        lengths = []
        for i in range(n_lists):
            count = int(counts[i])
            offsets.append(f.tell())
            f.write(struct.pack('<I', count))
            if count > 0:
                sl = slice(int(starts[i]), int(starts[i] + count))
                cur_ids = ids_sorted[sl].astype(np.int64)
                cur_vecs = vecs_sorted[sl].astype(np.float16)
                f.write(cur_ids.tobytes())
                f.write(cur_vecs.tobytes())
            lengths.append(f.tell() - offsets[-1])

        table_pos = f.tell()
        for off, ln in zip(offsets, lengths):
            f.write(struct.pack('<QQ', int(off), int(ln)))

        f.seek(36)
        f.write(struct.pack('<Q', table_pos))


def write_meta(prefix: str, df: pl.DataFrame, ids: np.ndarray, text_col: str, meta_cols):
    meta_cols = list(dict.fromkeys((meta_cols or []) + [text_col]))
    path = f"{prefix}.meta.jsonl"
    with open(path, 'w', encoding='utf-8') as f:
        for idx, row in enumerate(df.iter_rows(named=True)):
            meta = {k: row.get(k) for k in meta_cols if k in row}
            meta = {k: (v if v is not None else '') for k, v in meta.items()}
            meta['id'] = int(ids[idx])
            # heuristics
            meta.setdefault('title', meta.get('title') or meta.get('name') or meta.get(text_col, '')[:80])
            meta.setdefault('url', meta.get('url') or '')
            meta.setdefault('text', str(meta.get(text_col, '')))
            f.write(json.dumps(meta, ensure_ascii=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description='ANNPack v2 builder')
    parser.add_argument('--input', required=True, help='Path to .parquet / .csv / .json')
    parser.add_argument('--text-col', required=True, help='Column to embed')
    parser.add_argument('--id-col', help='Optional ID column')
    parser.add_argument('--meta-cols', nargs='*', help='Additional metadata columns')
    parser.add_argument('--model', default='all-MiniLM-L6-v2', help='SentenceTransformer model name')
    parser.add_argument('--output', default='generic_index', help='Output prefix')
    parser.add_argument('--lists', type=int, default=4096, help='IVF cluster count')
    parser.add_argument('--batch-size', type=int, default=512, help='Embedding batch size')
    parser.add_argument('--emb-col', help='Precomputed embedding column (list[float])')
    args = parser.parse_args()

    df = load_df(args.input)

    if args.id_col and args.id_col in df.columns:
        ids = np.asarray(df.select(args.id_col).to_numpy().reshape(-1), dtype=np.int64)
    else:
        ids = np.arange(df.height, dtype=np.int64)

    if args.emb_col:
        if args.emb_col not in df.columns:
            raise ValueError(f"Embedding column {args.emb_col} not found")
        emb_series = df.select(args.emb_col).to_series()
        emb_list = emb_series.to_list()
        vectors = np.asarray([np.asarray(v, dtype=np.float32) for v in emb_list], dtype=np.float32)
        faiss.normalize_L2(vectors)
    else:
        texts = df.select(args.text_col).to_series().to_list()
        vectors = embed_texts(texts, args.model, args.batch_size)

    dim = vectors.shape[1]
    centroids, list_ids = train_ivf(vectors, args.lists)

    write_annpack(args.output, dim, args.lists, vectors, ids, centroids, list_ids)
    write_meta(args.output, df, ids, args.text_col, args.meta_cols)

    print('[done] wrote', f"{args.output}.annpack", 'and', f"{args.output}.meta.jsonl")


if __name__ == '__main__':
    main()
