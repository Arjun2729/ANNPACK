"""Builder utilities for ANNPack indexes."""

import json
import os
import random
import struct
from typing import Optional, Tuple

import faiss
import numpy as np
import polars as pl
from typing import Any

from .logutil import log_event, timed


def _try_import_torch() -> Optional[Any]:
    """Import torch if available; return None if missing."""
    try:
        import torch
    except Exception:
        return None
    return torch


def select_device() -> str:
    """Select an embedding device (cpu/cuda/mps)."""
    forced = os.environ.get("ANNPACK_DEVICE", "").strip().lower()
    if forced in ("cpu", "cuda", "mps"):
        return forced
    torch = _try_import_torch()
    if torch is None:
        return "cpu"
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _set_cpu_safety_env():
    """Set conservative CPU environment defaults to avoid crashes."""
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


def load_data(
    path: str, text_col: str, id_col: Optional[str], max_rows: int
) -> Tuple[pl.DataFrame, np.ndarray]:
    """Load a CSV/Parquet dataset into a DataFrame and id array."""
    if max_rows <= 0:
        raise ValueError("max_rows must be positive")
    ext = os.path.splitext(path)[1].lower()
    if ext in (".parquet", ".parq"):
        df = pl.read_parquet(path)
    elif ext == ".csv":
        df = pl.read_csv(path)
    else:
        raise ValueError(f"Unsupported input extension: {ext}")

    if df.height == 0:
        raise ValueError("Input file has no rows")

    if df.height > max_rows:
        df = df.head(max_rows)

    if text_col not in df.columns:
        raise ValueError(f"text column '{text_col}' not found; available: {df.columns}")

    if id_col:
        if id_col not in df.columns:
            raise ValueError(f"id column '{id_col}' not found; available: {df.columns}")
        ids = df[id_col].to_numpy().astype(np.int64, copy=False)
    else:
        ids = np.arange(df.height, dtype=np.int64)

    if ids.shape[0] != df.height:
        raise ValueError("ids length does not match dataframe height")

    print(f"[build] Loaded {df.height} rows from {path}")
    sample_text = str(df[text_col][0])[:80] if df.height else ""
    sample_id = ids[0] if ids.size else "n/a"
    print(f"[build] Sample: id={sample_id} text='{sample_text}'")
    return df, ids


def embed_texts(
    texts: list[str],
    model_name: str,
    batch_size: int,
    device: Optional[str] = None,
    seed: int = 1234,
) -> Tuple[np.ndarray, int]:
    """Embed a list of texts into float32 vectors."""
    device = device or select_device()
    if device == "cpu":
        _set_cpu_safety_env()
    offline = (
        os.environ.get("ANNPACK_OFFLINE") == "1" or os.environ.get("ANNPACK_OFFLINE_DUMMY") == "1"
    )
    if offline:
        rng = np.random.default_rng(seed)
        dim = 16
        vectors = rng.standard_normal((len(texts), dim), dtype=np.float32)
        faiss.normalize_L2(vectors)
        print(
            f"[embed] Offline embeddings enabled (ANNPACK_OFFLINE=1). Shape={vectors.shape}, dim={dim}"
        )
        return vectors, dim
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise ImportError(
            "SentenceTransformer not installed. Install with: pip install annpack[embed]"
        ) from e
    print(f"[embed] Loading model '{model_name}' on device '{device}' ...")
    model = SentenceTransformer(model_name, device=device)
    dim = model.get_sentence_embedding_dimension()
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    vectors = np.asarray(vectors, dtype=np.float32)
    faiss.normalize_L2(vectors)
    print(f"[embed] Done. Shape={vectors.shape}, dim={dim}")
    return vectors, dim


def _effective_lists(requested: int, n_vectors: int) -> int:
    """Clamp the list count to a safe range for the dataset size."""
    if n_vectors <= 0:
        raise ValueError("No vectors to cluster.")
    eff = max(1, min(requested, n_vectors))
    if eff != requested:
        print(f"[build] Adjusted lists from {requested} -> {eff} (limited by {n_vectors} vectors)")
    return eff


def _train_centroids(vectors: np.ndarray, k: int, seed: int) -> np.ndarray:
    """Return k centroids with a deterministic fallback for tiny datasets."""
    n_vecs, dim = vectors.shape
    if n_vecs == 0 or dim == 0:
        raise ValueError("Empty vectors for k-means")

    tiny = n_vecs <= k or n_vecs < max(8, 2 * k)
    if tiny:
        if n_vecs >= k:
            idx = np.linspace(0, n_vecs - 1, num=k).astype(int)
        else:
            pad = np.full(k - n_vecs, n_vecs - 1, dtype=int)
            idx = np.concatenate([np.arange(n_vecs, dtype=int), pad])
        centroids = vectors[idx].astype(np.float32, copy=True)
        # One deterministic Lloyd step for stability.
        dists = ((vectors[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        labels = np.argmin(dists, axis=1)
        for i in range(k):
            mask = labels == i
            if np.any(mask):
                centroids[i] = vectors[mask].mean(axis=0)
        return centroids.astype(np.float32, copy=False)

    threads_raw = os.environ.get("ANNPACK_FAISS_THREADS", "1")
    try:
        threads = int(threads_raw)
    except ValueError:
        threads = 1
    if threads > 0:
        faiss.omp_set_num_threads(threads)

    print(f"[build] Training FAISS K-Means (k={k}, dim={dim}) on {n_vecs} vectors...")
    cp = faiss.ClusteringParameters()
    cp.min_points_per_centroid = 1
    cp.verbose = False
    cp.seed = seed
    cp.niter = 20
    cp.nredo = 1
    kmeans = faiss.Clustering(dim, k, cp)
    index = faiss.IndexFlatL2(dim)
    kmeans.train(vectors, index)
    return faiss.vector_float_to_array(kmeans.centroids).reshape(k, dim).astype(np.float32)


def train_ivf(vectors: np.ndarray, dim: int, n_lists: int, seed: int = 1234):
    """Train IVF centroids and assign vectors to lists."""
    n_vecs = vectors.shape[0]
    n_lists = _effective_lists(n_lists, n_vecs)

    if n_lists == 1:
        print(f"[build] Small dataset ({n_vecs} vectors); using 1 list without k-means.")
        list_ids = np.zeros(n_vecs, dtype=np.int64)
        centroid = vectors.mean(axis=0, keepdims=True).astype(np.float32)
        return centroid, list_ids, n_lists

    tiny = n_vecs <= n_lists or n_vecs < max(8, 2 * n_lists)
    centroids = _train_centroids(vectors, n_lists, seed)
    if tiny:
        dists = ((vectors[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        list_ids = np.argmin(dists, axis=1).astype(np.int64)
    else:
        index = faiss.IndexFlatL2(dim)
        index.add(centroids)
        print("[build] Assigning vectors to clusters...")
        _, list_ids = index.search(vectors, 1)
        list_ids = list_ids.flatten()
    return centroids, list_ids, n_lists


def write_annpack(
    filename: str,
    dim: int,
    n_lists: int,
    vectors: np.ndarray,
    doc_ids: np.ndarray,
    centroids: np.ndarray,
    list_ids: np.ndarray,
) -> None:
    """Write ANNPack binary index to disk."""
    print(f"[write] Writing {filename} ...")
    with open(filename, "wb") as f:
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
                0,  # offset table ptr placeholder
            )
        )
        f.write(b"\x00" * (header_size - f.tell()))

        # Centroids
        f.write(centroids.tobytes())

        print("[write] Sorting for cluster-major layout...")
        order = np.argsort(list_ids)
        vecs_sorted = vectors[order]
        ids_sorted = doc_ids[order]
        counts = np.bincount(list_ids, minlength=n_lists)
        starts = np.concatenate(([0], np.cumsum(counts[:-1])))

        list_offsets = []
        list_lengths = []

        print("[write] Writing lists...")
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
        print(f"[write] Offset table at {table_pos} ...")
        for off, length in zip(list_offsets, list_lengths):
            f.write(struct.pack("<QQ", off, length))

        # Patch header pointer
        f.seek(36)
        f.write(struct.pack("<Q", table_pos))

    size_mb = os.path.getsize(filename) / (1024 * 1024)
    print(f"[write] Done: {filename} ({size_mb:.2f} MB)")


def write_metadata(meta_path: str, df: pl.DataFrame, ids: np.ndarray, text_col: str) -> None:
    """Write metadata JSONL matching the vector ids."""
    print(f"[meta] Writing metadata to {meta_path} ...")
    with open(meta_path, "w", encoding="utf-8") as mf:
        for idx, row in zip(ids.tolist(), df.to_dicts()):
            row["id"] = int(idx)
            if text_col in row:
                row[text_col] = str(row[text_col])
            mf.write(json.dumps(row, ensure_ascii=False))
            mf.write("\n")


def build_index_from_df(
    df: pl.DataFrame,
    text_col: str,
    ids: np.ndarray,
    output_prefix: str = "annpack_index",
    model_name: str = "all-MiniLM-L6-v2",
    n_lists: int = 1024,
    batch_size: int = 512,
    device: Optional[str] = None,
    seed: int = 1234,
) -> None:
    """Build an ANNPack index from a DataFrame and id array."""
    np.random.seed(seed)
    torch = _try_import_torch()
    if torch is not None:
        torch.manual_seed(seed)
    random.seed(seed)
    if text_col not in df.columns:
        raise ValueError(f"text column '{text_col}' not found; available: {df.columns}")
    if ids.shape[0] != df.height:
        raise ValueError("ids length does not match dataframe height")

    texts = [str(t) for t in df[text_col].to_list()]
    log_event("build_start", {"rows": len(texts), "lists": n_lists})
    with timed("embed", {"rows": len(texts)}):
        vectors, dim = embed_texts(texts, model_name, batch_size, device=device, seed=seed)
    with timed("train_ivf", {"dim": dim, "lists": n_lists}):
        centroids, list_ids, n_lists = train_ivf(vectors, dim, n_lists, seed=seed)

    ann_path = f"{output_prefix}.annpack"
    meta_path = f"{output_prefix}.meta.jsonl"
    with timed("write_annpack", {"path": ann_path}):
        write_annpack(ann_path, dim, n_lists, vectors, ids, centroids, list_ids)
    with timed("write_meta", {"path": meta_path}):
        write_metadata(meta_path, df, ids, text_col)

    print("[done] Built ANNPack index:")
    print(f"  - Vectors: {vectors.shape[0]}")
    print(f"  - Dim: {dim}")
    print(f"  - Lists: {n_lists}")
    print(f"  - File: {ann_path}")
    print(f"  - Metadata: {meta_path}")


def build_index(
    input_path: str,
    text_col: str,
    id_col: Optional[str] = None,
    output_prefix: str = "annpack_index",
    model_name: str = "all-MiniLM-L6-v2",
    n_lists: int = 1024,
    max_rows: int = 100000,
    batch_size: int = 512,
    device: Optional[str] = None,
    seed: int = 1234,
) -> None:
    """Build an ANNPack index from a CSV/Parquet file."""
    df, ids = load_data(input_path, text_col, id_col, max_rows)
    build_index_from_df(
        df=df,
        text_col=text_col,
        ids=ids,
        output_prefix=output_prefix,
        model_name=model_name,
        n_lists=n_lists,
        batch_size=batch_size,
        device=device,
        seed=seed,
    )


def load_hf_wikipedia(
    dataset_name: str = "wikimedia/wikipedia",
    config: str = "20231101.en",
    split: str = "train",
    max_rows: int = 1_000_000,
) -> Tuple[pl.DataFrame, np.ndarray]:
    """Load a HuggingFace Wikipedia dataset into a DataFrame."""
    if max_rows <= 0:
        raise ValueError("max_rows must be positive")
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError("Install embed dependencies with: pip install annpack[embed]") from e

    print(f"[hf] Loading dataset {dataset_name}/{config} split={split} ...")
    try:
        ds = load_dataset(dataset_name, config, split=split)
    except Exception as e:
        msg = (
            "Failed to load HF dataset. The legacy 'wikipedia' script is no longer supported. "
            "Try dataset_name='wikimedia/wikipedia' and a parquet config such as '20231101.en'. "
            f"Original error: {e}"
        )
        raise RuntimeError(msg) from e
    n = min(max_rows, len(ds))
    if n < len(ds):
        ds = ds.select(range(n))
    print(f"[hf] Loaded {len(ds)} rows from HF")

    pdf = ds.to_pandas()
    df = pl.DataFrame(pdf)
    if "id" not in df.columns or "text" not in df.columns:
        raise ValueError("HF wikipedia dataset must contain 'id' and 'text' columns")
    df = df.with_columns(pl.col("id").cast(pl.Int64))
    ids = df["id"].to_numpy().astype(np.int64, copy=False)
    return df, ids


def build_index_from_hf_wikipedia(
    output_prefix: str = "wikipedia_1M",
    dataset_name: str = "wikimedia/wikipedia",
    config: str = "20231101.en",
    split: str = "train",
    max_rows: int = 1_000_000,
    model_name: str = "all-MiniLM-L6-v2",
    n_lists: int = 4096,
    batch_size: int = 512,
    device: Optional[str] = None,
    seed: int = 1234,
) -> None:
    """Build an ANNPack index from a HF Wikipedia dataset."""
    df, ids = load_hf_wikipedia(
        dataset_name=dataset_name,
        config=config,
        split=split,
        max_rows=max_rows,
    )
    build_index_from_df(
        df=df,
        text_col="text",
        ids=ids,
        output_prefix=output_prefix,
        model_name=model_name,
        n_lists=n_lists,
        batch_size=batch_size,
        device=device,
        seed=seed,
    )
