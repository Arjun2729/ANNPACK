# LEGACY: Cohere-specific Wikipedia builder. For new indexes, use annpack_build.py (annpack-build CLI).
import time
import numpy as np
import struct
import os
import polars as pl
import faiss
from huggingface_hub import hf_hub_download

# --- CONFIGURATION ---
OUTPUT_FILE = "wiki_1M.annpack"
N_LISTS = 4096
DIM = 768
TARGET_ROWS = 1_000_000
# Dataset and shard discovery
REPO_ID = "Cohere/wikipedia-22-12-en-embeddings"


def write_annpack(filename, centroids, list_ids, vectors, doc_ids):
    print(f"[write] Writing {filename}...")
    with open(filename, "wb") as f:
        magic = 0x504E4E41
        # Header: Magic(8) | Ver(4) | Endian(4) | HSize(4) | Dim(4) | Metric(4) | Lists(4) | Vecs(4) | OffsetPtr(8)
        f.write(struct.pack("<QIIIIIIIQ",
                            magic, 1, 1, 72, DIM, 1, N_LISTS, len(vectors), 0))
        f.write(b"\x00" * (72 - f.tell()))

        print("[write] Centroids...")
        f.write(centroids.tobytes())

        print("[write] Sorting/order for sequential write...")
        order = np.argsort(list_ids)
        vecs_sorted = vectors[order]
        ids_sorted = doc_ids[order]
        counts = np.bincount(list_ids, minlength=N_LISTS)
        starts = np.concatenate(([0], np.cumsum(counts[:-1])))

        list_offsets, list_lengths = [], []
        print("[write] Streaming lists...")
        for i in range(N_LISTS):
            if i % 512 == 0:
                print(f"[write] List {i}/{N_LISTS}")
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
        print(f"[write] Offset table at {table_pos}...")
        for off, length in zip(list_offsets, list_lengths):
            f.write(struct.pack("<QQ", off, length))

        print("[write] Patching header pointer...")
        f.seek(36)
        f.write(struct.pack("<Q", table_pos))

    size_mb = os.path.getsize(filename) / (1024 * 1024)
    print(f"[write] Done: {filename} ({size_mb:.2f} MB)")


def main():
    start_global = time.time()
    print("[discover] Listing shards from HuggingFace...")
    from huggingface_hub import list_repo_files

    all_files = list_repo_files(REPO_ID, repo_type="dataset")
    shard_files = sorted(
        f for f in all_files if f.startswith("data/train-") and f.endswith(".parquet")
    )
    if not shard_files:
        raise RuntimeError("No parquet shards found in dataset")

    print(f"[discover] Found {len(shard_files)} shards; will load until {TARGET_ROWS} rows")
    vectors = np.zeros((TARGET_ROWS, DIM), dtype=np.float32)
    doc_ids = np.zeros(TARGET_ROWS, dtype=np.int64)

    total_rows = 0
    for idx, shard in enumerate(shard_files):
        if total_rows >= TARGET_ROWS:
            break
        print(f"[download] Fetching shard {idx+1}/{len(shard_files)}: {shard}")
        parquet_file = hf_hub_download(
            repo_id=REPO_ID,
            filename=shard,
            repo_type="dataset",
        )

        print(f"[load] Reading {shard} with Polars (columns id, emb)...")
        df = pl.read_parquet(parquet_file, columns=["id", "emb"])
        shard_rows = df.height
        if shard_rows == 0:
            continue

        print(f"[load] Shard rows: {shard_rows}")
        shard_vectors = np.stack(df["emb"].to_numpy()).astype(np.float32)
        shard_ids = df["id"].to_numpy().astype(np.int64)

        space_left = TARGET_ROWS - total_rows
        take = min(space_left, shard_rows)
        vectors[total_rows:total_rows + take] = shard_vectors[:take]
        doc_ids[total_rows:total_rows + take] = shard_ids[:take]
        total_rows += take
        print(f"[load] Total loaded: {total_rows}")

    if total_rows == 0:
        raise RuntimeError("No data loaded; aborting")

    if total_rows < TARGET_ROWS:
        print(f"[warn] Only loaded {total_rows} rows; continuing with available data")

    vectors = vectors[:total_rows]
    doc_ids = doc_ids[:total_rows]
    print(f"[load] Loaded {len(vectors)} vectors in {time.time() - start_global:.2f}s")

    print(f"[kmeans] Training FAISS K-Means (k={N_LISTS}) on {len(vectors)} vectors...")
    kmeans = faiss.Kmeans(DIM, N_LISTS, niter=20, verbose=True)
    kmeans.train(vectors)
    print("[kmeans] Assigning vectors...")
    _, list_ids = kmeans.index.search(vectors, 1)
    list_ids = list_ids.flatten()
    centroids = kmeans.centroids

    print("[write] Starting annpack write...")
    write_annpack(OUTPUT_FILE, centroids, list_ids, vectors, doc_ids)
    print(f"[done] Total time: {time.time() - start_global:.2f}s")


if __name__ == "__main__":
    main()
