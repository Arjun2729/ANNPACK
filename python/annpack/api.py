"""Public Python API for ANNPack."""

from __future__ import annotations

import hashlib
import json
import os
import struct
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Any, Dict, Iterable, List, Literal, Optional, TYPE_CHECKING, cast

import numpy as np

from .build import build_index
from .reader import ANNPackIndex
from .packset import open_packset

if TYPE_CHECKING:
    from .packset import PackSet


def _find_manifest(pack_dir: Path) -> Path:
    """Return the first manifest path in a pack directory."""
    candidates = list(pack_dir.glob("*.manifest.json")) + list(pack_dir.glob("manifest.json"))
    if not candidates:
        raise FileNotFoundError(f"No manifest found in {pack_dir}")
    return candidates[0]


def _read_header(path: Path) -> Dict[str, int]:
    """Read the ANNPack header and return a dict of fields."""
    with path.open("rb") as handle:
        header = handle.read(72)
    fields = struct.unpack("<QIIIIIIIQ", header[:44])
    magic, version, endian, header_size, dim, metric, n_lists, n_vectors, offset_table_pos = fields
    return {
        "magic": magic,
        "version": version,
        "endian": endian,
        "header_size": header_size,
        "dim": dim,
        "metric": metric,
        "n_lists": n_lists,
        "n_vectors": n_vectors,
        "offset_table_pos": offset_table_pos,
    }


def _write_manifest(prefix: Path, ann_path: Path, meta_path: Path) -> Path:
    """Write a minimal manifest.json for a single-pack output."""
    info = _read_header(ann_path)
    manifest = {
        "schema_version": 2,
        "version": 1,
        "created_by": "annpack.api",
        "dim": info["dim"],
        "n_lists": info["n_lists"],
        "n_vectors": info["n_vectors"],
        "shards": [
            {
                "name": prefix.name,
                "annpack": ann_path.name,
                "meta": meta_path.name,
                "n_vectors": info["n_vectors"],
            }
        ],
    }
    manifest_path = prefix.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return max(1, value)


DEFAULT_META_MAX_BYTES = _env_int(
    "ANNPACK_MAX_META_BYTES", _env_int("ANNPACK_META_MAX_BYTES", 1024 * 1024 * 1024)
)
DEFAULT_META_MAX_LINE = _env_int("ANNPACK_META_MAX_LINE", 1024 * 1024)


class _MetaLookup:
    """Abstract metadata lookup helper."""

    def get(self, key: int) -> Optional[Dict[str, Any]]:
        raise NotImplementedError

    def close(self) -> None:
        return None


class _MetaDict(_MetaLookup):
    def __init__(self, data: Dict[int, Dict[str, Any]]):
        self._data = data

    def get(self, key: int) -> Optional[Dict[str, Any]]:
        return self._data.get(key)


class _MetaStream(_MetaLookup):
    def __init__(self, path: Path, max_line: int):
        self._path = path
        self._max_line = max_line
        self._cache: Dict[int, Dict[str, Any]] = {}

    def get(self, key: int) -> Optional[Dict[str, Any]]:
        if key in self._cache:
            return self._cache[key]
        with self._path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if len(line) > self._max_line:
                    raise ValueError("Metadata line exceeds maximum length")
                if not line.strip():
                    continue
                row = json.loads(line)
                if "id" not in row:
                    continue
                doc_id = int(row["id"])
                if doc_id == key:
                    typed = cast(Dict[str, Any], row)
                    self._cache[key] = typed
                    return typed
        return None


class _MetaNull(_MetaLookup):
    def get(self, key: int) -> Optional[Dict[str, Any]]:
        return None


def _load_meta(
    meta_path: Path,
    max_bytes: int,
    max_line: int,
    stream: bool,
    allow_large: bool,
) -> _MetaLookup:
    """Load metadata JSONL with size guardrails."""
    size = meta_path.stat().st_size
    if not allow_large and size > max_bytes:
        raise ValueError(f"Metadata file too large: {size} bytes")
    if stream:
        return _MetaStream(meta_path, max_line=max_line)
    meta: Dict[int, Dict[str, Any]] = {}
    with meta_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if len(line) > max_line:
                raise ValueError("Metadata line exceeds maximum length")
            if not line.strip():
                continue
            row = json.loads(line)
            if "id" not in row:
                continue
            meta[int(row["id"])] = row
    return _MetaDict(meta)


def _hash_seed(text: str, seed: int) -> int:
    """Derive a stable seed for offline embedding."""
    h = hashlib.sha256(f"{seed}:{text}".encode("utf-8")).digest()
    return int.from_bytes(h[:8], "little", signed=False)


def _normalize(vec: np.ndarray) -> np.ndarray:
    """L2-normalize a vector."""
    norm = np.linalg.norm(vec)
    if norm == 0:
        raise ValueError("Zero vector")
    return cast(np.ndarray, vec / norm)


def _offline_embed(text: str, dim: int, seed: int) -> np.ndarray:
    """Generate a deterministic offline embedding for a single text."""
    rng = np.random.default_rng(_hash_seed(text, seed))
    vec = rng.standard_normal((dim,), dtype=np.float32)
    return _normalize(vec)


@dataclass
class _Shard:
    name: str
    index: ANNPackIndex
    meta: _MetaLookup


class Pack:
    """Loaded ANNPack with metadata and embedding support."""

    def __init__(self, pack_dir: Path, shards: List[_Shard], dim: int, seed: int = 0):
        self.pack_dir = pack_dir
        self._shards = shards
        self._dim = dim
        self._seed = seed
        self._model: Optional[Any] = None

    def _embed_query(self, text: str) -> np.ndarray:
        """Embed a query string (offline or model-backed)."""
        if os.environ.get("ANNPACK_OFFLINE") == "1":
            return _offline_embed(text, self._dim, self._seed)
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as e:
                raise ImportError(
                    "SentenceTransformer not installed. Install with: pip install annpack[embed]"
                ) from e
            self._model = SentenceTransformer("all-MiniLM-L6-v2")
        model = self._model
        if model is None:
            raise RuntimeError("Embedding model missing")
        vecs = model.encode([text], normalize_embeddings=True)
        return np.asarray(vecs[0], dtype=np.float32)

    def search(self, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search by text and return a list of result dicts."""
        vec = self._embed_query(query_text)
        return self.search_vec(vec, top_k=top_k)

    def search_vec(self, vector: Iterable[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """Search by vector and return a list of result dicts."""
        vec = np.asarray(list(vector), dtype=np.float32)
        if vec.ndim != 1 or vec.shape[0] != self._dim:
            raise ValueError(f"Vector must be 1-D of length {self._dim}")
        vec = _normalize(vec)

        results: List[Dict[str, Any]] = []
        for shard in self._shards:
            for doc_id, score in shard.index.search(vec, k=top_k):
                results.append(
                    {
                        "id": doc_id,
                        "score": float(score),
                        "shard": shard.name,
                        "meta": shard.meta.get(doc_id),
                    }
                )

        results.sort(key=lambda r: (-r["score"], r["shard"], r["id"]))
        return results[:top_k]

    def close(self) -> None:
        """Close all underlying indexes."""
        for shard in self._shards:
            shard.index.close()
            shard.meta.close()

    def __enter__(self) -> "Pack":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> Literal[False]:
        self.close()
        return False


def build_pack(
    input_csv: str,
    output_dir: str,
    text_col: str = "text",
    id_col: str = "id",
    lists: Optional[int] = None,
    seed: int = 0,
    offline: Optional[bool] = None,
    **kwargs: Any,
) -> Dict[str, str]:
    """Build a single-pack ANN index from a CSV/Parquet input."""
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    output_prefix = output / "pack"

    prev_offline = os.environ.get("ANNPACK_OFFLINE")
    if offline is not None:
        os.environ["ANNPACK_OFFLINE"] = "1" if offline else "0"

    try:
        build_index(
            input_path=input_csv,
            text_col=text_col,
            id_col=id_col,
            output_prefix=str(output_prefix),
            n_lists=lists or 1024,
            seed=seed,
            **kwargs,
        )
    finally:
        if offline is not None:
            if prev_offline is None:
                os.environ.pop("ANNPACK_OFFLINE", None)
            else:
                os.environ["ANNPACK_OFFLINE"] = prev_offline

    ann_path = output_prefix.with_suffix(".annpack")
    meta_path = output_prefix.with_suffix(".meta.jsonl")
    manifest = _write_manifest(output_prefix, ann_path, meta_path)
    return {
        "pack_dir": str(output),
        "manifest": str(manifest),
        "annpack": str(ann_path),
        "meta": str(meta_path),
    }


def open_pack(
    pack_dir: str,
    *,
    load_meta: bool = True,
    max_meta_bytes: int = DEFAULT_META_MAX_BYTES,
    max_meta_line: int = DEFAULT_META_MAX_LINE,
    stream_meta: bool = False,
) -> Pack | "PackSet":
    """Open a pack directory (schema v2) or packset (schema v3)."""
    allow_large = os.getenv("ANNPACK_ALLOW_LARGE_META") == "1"
    base = Path(pack_dir).expanduser().resolve()
    manifest_path = _find_manifest(base)
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    if data.get("schema_version") == 3:
        return open_packset(
            str(base),
            load_meta=load_meta,
            max_meta_bytes=max_meta_bytes,
            max_meta_line=max_meta_line,
        )
    shards = data.get("shards") or []
    if not shards:
        raise ValueError("Manifest contains no shards")

    shard_objs: List[_Shard] = []
    dim = None
    for shard in shards:
        ann_path = base / shard["annpack"]
        meta_path = base / shard["meta"]
        index = ANNPackIndex.open(str(ann_path))
        if index.header is None:
            raise ValueError(f"Index header missing for {ann_path}")
        if dim is None:
            dim = index.header.dim
        if load_meta:
            meta = _load_meta(
                meta_path,
                max_bytes=max_meta_bytes,
                max_line=max_meta_line,
                stream=stream_meta,
                allow_large=allow_large,
            )
        else:
            meta = _MetaNull()
        shard_objs.append(_Shard(name=shard.get("name", ann_path.name), index=index, meta=meta))

    return Pack(base, shard_objs, dim=dim or 0)
