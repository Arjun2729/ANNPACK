"""PackSet helpers for base + delta packs (schema v3)."""

from __future__ import annotations

import hashlib
import json
import os
import struct
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Set, Tuple, cast

import numpy as np

from .build import build_index
from .reader import ANNPackIndex


def _sha256_file(path: Path) -> str:
    """Return hex sha256 for a file."""
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_header(path: Path) -> Dict[str, int]:
    """Read ANNPack header fields into a dict."""
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


def _find_manifest(pack_dir: Path) -> Path:
    """Find the first manifest file in a pack directory."""
    candidates = list(pack_dir.glob("*.manifest.json")) + list(pack_dir.glob("manifest.json"))
    if not candidates:
        raise FileNotFoundError(f"No manifest found in {pack_dir}")
    return candidates[0]


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


def _load_meta(
    meta_path: Path, max_bytes: int, max_line: int, allow_large: bool
) -> Dict[int, Dict[str, Any]]:
    """Load metadata JSONL into a dict keyed by id with size guardrails."""
    size = meta_path.stat().st_size
    if not allow_large and size > max_bytes:
        raise ValueError(f"Metadata file too large: {size} bytes")
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
    return meta


def _hash_seed(text: str, seed: int) -> int:
    """Generate a stable seed for offline embeddings."""
    h = hashlib.sha256(f"{seed}:{text}".encode("utf-8")).digest()
    return int.from_bytes(h[:8], "little", signed=False)


def _normalize(vec: np.ndarray) -> np.ndarray:
    """L2-normalize a vector."""
    norm = np.linalg.norm(vec)
    if norm == 0:
        raise ValueError("Zero vector")
    return cast(np.ndarray, vec / norm)


def _offline_embed(text: str, dim: int, seed: int) -> np.ndarray:
    """Generate a deterministic offline embedding for one text."""
    rng = np.random.default_rng(_hash_seed(text, seed))
    vec = rng.standard_normal((dim,), dtype=np.float32)
    return _normalize(vec)


def _read_tombstones(path: Path) -> Set[int]:
    """Load tombstoned ids from a JSONL file."""
    ids: Set[int] = set()
    if not path.exists():
        return ids
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if "id" in row:
                ids.add(int(row["id"]))
    return ids


@dataclass
class _Shard:
    name: str
    index: ANNPackIndex
    meta: Optional[Dict[int, Dict[str, Any]]]


@dataclass
class DeltaInfo:
    seq: int
    path: Path
    annpack: Path
    meta: Path
    tombstones: Path
    base_sha256_annpack: str
    sha256_annpack: str
    sha256_meta: str
    sha256_tombstones: str


class PackSet:
    """Searchable view of a base pack plus ordered deltas."""

    def __init__(
        self,
        root_dir: Path,
        base_shards: List[_Shard],
        deltas: List[Tuple[DeltaInfo, List[_Shard]]],
        tombstoned_ids: Set[int],
        overridden_ids: Set[int],
        dim: int,
        seed: int = 0,
    ):
        self.root_dir = root_dir
        self._base_shards = base_shards
        self._deltas = deltas
        self._tombstoned_ids = tombstoned_ids
        self._overridden_ids = overridden_ids
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

    def _search_shards(
        self, shards: Sequence[_Shard], vec: np.ndarray, k: int
    ) -> List[Dict[str, Any]]:
        """Search a shard list and return sorted result rows."""
        results: List[Dict[str, Any]] = []
        for shard in shards:
            for doc_id, score in shard.index.search(vec, k=k):
                results.append(
                    {
                        "id": doc_id,
                        "score": float(score),
                        "shard": shard.name,
                        "meta": shard.meta.get(doc_id) if shard.meta else None,
                    }
                )
        results.sort(key=lambda r: (-r["score"], r["shard"], r["id"]))
        return results[:k]

    def search(self, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search by text and return result dicts."""
        vec = self._embed_query(query_text)
        return self.search_vec(vec, top_k=top_k)

    def search_vec(self, vector: Iterable[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """Search by vector and return result dicts."""
        vec = np.asarray(list(vector), dtype=np.float32)
        if vec.ndim != 1 or vec.shape[0] != self._dim:
            raise ValueError(f"Vector must be 1-D of length {self._dim}")
        vec = _normalize(vec)

        per_pack_k = max(top_k * 5, top_k)
        seen: Set[int] = set()
        results: List[Dict[str, Any]] = []

        for _, shards in sorted(self._deltas, key=lambda d: d[0].seq, reverse=True):
            hits = self._search_shards(shards, vec, per_pack_k)
            for row in hits:
                doc_id = int(row["id"])
                if doc_id in self._tombstoned_ids or doc_id in seen:
                    continue
                seen.add(doc_id)
                results.append(row)
                if len(results) >= top_k:
                    return results

        base_hits = self._search_shards(self._base_shards, vec, per_pack_k)
        for row in base_hits:
            doc_id = int(row["id"])
            if doc_id in self._tombstoned_ids or doc_id in self._overridden_ids or doc_id in seen:
                continue
            seen.add(doc_id)
            results.append(row)
            if len(results) >= top_k:
                break

        return results

    def close(self) -> None:
        """Close all underlying indexes."""
        for shard in self._base_shards:
            shard.index.close()
        for _, shards in self._deltas:
            for shard in shards:
                shard.index.close()

    def __enter__(self) -> "PackSet":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> Literal[False]:
        self.close()
        return False


def _open_pack_dir(
    pack_dir: Path,
    *,
    load_meta: bool,
    max_meta_bytes: int,
    max_meta_line: int,
) -> Tuple[List[_Shard], int]:
    """Open a base/delta pack directory and return shard objects + dim."""
    manifest_path: Optional[Path] = None
    shards: List[Dict[str, Any]] = []
    try:
        manifest_path = _find_manifest(pack_dir)
    except FileNotFoundError:
        manifest_path = None

    if manifest_path is not None:
        data: Dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
        shards = data.get("shards") or []
    if not shards:
        ann_path = pack_dir / "pack.annpack"
        meta_path = pack_dir / "pack.meta.jsonl"
        if not ann_path.exists() or not meta_path.exists():
            raise ValueError(f"Pack dir missing manifest and pack.* files: {pack_dir}")
        shards = [{"name": "pack", "annpack": "pack.annpack", "meta": "pack.meta.jsonl"}]

    shard_objs: List[_Shard] = []
    dim: Optional[int] = None
    for shard in shards:
        ann_path = pack_dir / str(shard["annpack"])
        meta_path = pack_dir / str(shard["meta"])
        index = ANNPackIndex.open(str(ann_path))
        if index.header is None:
            raise ValueError(f"Index header missing for {ann_path}")
        if dim is None:
            dim = index.header.dim
        if load_meta:
            allow_large = os.getenv("ANNPACK_ALLOW_LARGE_META") == "1"
            meta = _load_meta(
                meta_path,
                max_bytes=max_meta_bytes,
                max_line=max_meta_line,
                allow_large=allow_large,
            )
        else:
            meta = None
        shard_name = str(shard.get("name", ann_path.name))
        shard_objs.append(_Shard(name=shard_name, index=index, meta=meta))
    return shard_objs, dim or 0


def build_packset_base(
    input_csv: str,
    packset_dir: str,
    text_col: str = "text",
    id_col: str = "id",
    lists: int = 1024,
    seed: int = 0,
    offline: Optional[bool] = None,
    **kwargs: Any,
) -> Dict[str, str]:
    """Build a base pack and create a PackSet root manifest."""
    root = Path(packset_dir).expanduser().resolve()
    base_dir = root / "base"
    base_dir.mkdir(parents=True, exist_ok=True)

    prev_offline = os.environ.get("ANNPACK_OFFLINE")
    if offline is not None:
        os.environ["ANNPACK_OFFLINE"] = "1" if offline else "0"
    try:
        build_index(
            input_path=input_csv,
            text_col=text_col,
            id_col=id_col,
            output_prefix=str(base_dir / "pack"),
            n_lists=lists,
            seed=seed,
            **kwargs,
        )
    finally:
        if offline is not None:
            if prev_offline is None:
                os.environ.pop("ANNPACK_OFFLINE", None)
            else:
                os.environ["ANNPACK_OFFLINE"] = prev_offline

    base_ann = base_dir / "pack.annpack"
    base_meta = base_dir / "pack.meta.jsonl"
    base_manifest = base_dir / "pack.manifest.json"
    if not base_manifest.exists():
        info = _read_header(base_ann)
        manifest: Dict[str, Any] = {
            "schema_version": 2,
            "version": 1,
            "created_by": "annpack.packset",
            "dim": info["dim"],
            "n_lists": info["n_lists"],
            "n_vectors": info["n_vectors"],
            "shards": [
                {
                    "name": "pack",
                    "annpack": base_ann.name,
                    "meta": base_meta.name,
                    "n_vectors": info["n_vectors"],
                }
            ],
        }
        base_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    root_manifest: Dict[str, Any] = {
        "schema_version": 3,
        "base": {
            "path": "base",
            "annpack": "base/pack.annpack",
            "meta": "base/pack.meta.jsonl",
            "sha256_annpack": _sha256_file(base_ann),
            "sha256_meta": _sha256_file(base_meta),
        },
        "deltas": [],
    }
    (root / "pack.manifest.json").write_text(json.dumps(root_manifest, indent=2), encoding="utf-8")
    return {
        "packset_dir": str(root),
        "base_dir": str(base_dir),
        "manifest": str(root / "pack.manifest.json"),
    }


def build_delta(
    base_dir: str,
    add_csv: str,
    delete_ids: Optional[Iterable[int]] = None,
    out_delta_dir: str = "",
    text_col: str = "text",
    id_col: str = "id",
    lists: int = 1024,
    seed: int = 0,
    offline: Optional[bool] = None,
    **kwargs: Any,
) -> DeltaInfo:
    """Build a delta pack (adds/updates + tombstones)."""
    base = Path(base_dir).expanduser().resolve()
    if not base.exists():
        raise FileNotFoundError(f"Base dir not found: {base}")

    delta_dir = Path(out_delta_dir).expanduser().resolve()
    delta_dir.mkdir(parents=True, exist_ok=True)

    prev_offline = os.environ.get("ANNPACK_OFFLINE")
    if offline is not None:
        os.environ["ANNPACK_OFFLINE"] = "1" if offline else "0"

    try:
        build_index(
            input_path=add_csv,
            text_col=text_col,
            id_col=id_col,
            output_prefix=str(delta_dir / "pack"),
            n_lists=lists,
            seed=seed,
            **kwargs,
        )
    finally:
        if offline is not None:
            if prev_offline is None:
                os.environ.pop("ANNPACK_OFFLINE", None)
            else:
                os.environ["ANNPACK_OFFLINE"] = prev_offline

    tombstone_path = delta_dir / "tombstones.jsonl"
    ids = sorted(set(int(x) for x in delete_ids or []))
    with tombstone_path.open("w", encoding="utf-8") as handle:
        for doc_id in ids:
            handle.write(json.dumps({"id": doc_id}))
            handle.write("\n")

    base_ann = base / "pack.annpack"
    ann_path = delta_dir / "pack.annpack"
    meta_path = delta_dir / "pack.meta.jsonl"
    info = DeltaInfo(
        seq=0,
        path=delta_dir,
        annpack=ann_path,
        meta=meta_path,
        tombstones=tombstone_path,
        base_sha256_annpack=_sha256_file(base_ann),
        sha256_annpack=_sha256_file(ann_path),
        sha256_meta=_sha256_file(meta_path),
        sha256_tombstones=_sha256_file(tombstone_path),
    )
    delta_manifest: Dict[str, Any] = {
        "schema_version": 3,
        "base_sha256_annpack": info.base_sha256_annpack,
        "sha256_annpack": info.sha256_annpack,
        "sha256_meta": info.sha256_meta,
        "sha256_tombstones": info.sha256_tombstones,
    }
    (delta_dir / "delta.manifest.json").write_text(
        json.dumps(delta_manifest, indent=2), encoding="utf-8"
    )
    return info


def update_packset_manifest(packset_dir: str, delta_dir: str, seq: int) -> Path:
    """Append a delta entry to the PackSet manifest deterministically."""
    root = Path(packset_dir).expanduser().resolve()
    manifest_path = root / "pack.manifest.json"
    data: Dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
    if data.get("schema_version") != 3:
        raise ValueError("PackSet manifest must have schema_version=3")

    base_sha = data["base"]["sha256_annpack"]
    delta_path = Path(delta_dir).expanduser().resolve()
    ann_path = delta_path / "pack.annpack"
    meta_path = delta_path / "pack.meta.jsonl"
    tomb_path = delta_path / "tombstones.jsonl"
    if not tomb_path.exists():
        tomb_path.write_text("", encoding="utf-8")
    delta_entry = {
        "seq": seq,
        "path": str(delta_path.relative_to(root)),
        "annpack": str(ann_path.relative_to(root)),
        "meta": str(meta_path.relative_to(root)),
        "tombstones": str(tomb_path.relative_to(root)),
        "base_sha256_annpack": base_sha,
        "sha256_annpack": _sha256_file(ann_path),
        "sha256_meta": _sha256_file(meta_path),
        "sha256_tombstones": _sha256_file(tomb_path),
    }

    deltas: List[Dict[str, Any]] = data.get("deltas") or []
    deltas.append(delta_entry)
    deltas = sorted(deltas, key=lambda d: d["seq"])
    data["deltas"] = deltas
    manifest_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return manifest_path


def open_packset(
    packset_dir: str,
    *,
    load_meta: bool = True,
    max_meta_bytes: int = DEFAULT_META_MAX_BYTES,
    max_meta_line: int = DEFAULT_META_MAX_LINE,
) -> PackSet:
    """Open a PackSet root and return a searchable PackSet object."""
    root = Path(packset_dir).expanduser().resolve()
    manifest_path = root / "pack.manifest.json"
    data: Dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
    if data.get("schema_version") != 3:
        raise ValueError("Not a PackSet manifest")

    base_info: Dict[str, Any] = data["base"]
    base_dir = root / base_info["path"]
    base_ann = root / base_info["annpack"]
    base_meta = root / base_info["meta"]
    if _sha256_file(base_ann) != base_info["sha256_annpack"]:
        raise ValueError("Base annpack hash mismatch")
    if _sha256_file(base_meta) != base_info["sha256_meta"]:
        raise ValueError("Base meta hash mismatch")

    base_shards, dim = _open_pack_dir(
        base_dir,
        load_meta=load_meta,
        max_meta_bytes=max_meta_bytes,
        max_meta_line=max_meta_line,
    )

    tombstoned_ids: Set[int] = set()
    overridden_ids: Set[int] = set()
    deltas: List[Tuple[DeltaInfo, List[_Shard]]] = []

    for delta in data.get("deltas") or []:
        delta_info: Dict[str, Any] = delta
        delta_path = root / str(delta_info["path"])
        ann_path = root / str(delta_info["annpack"])
        meta_path = root / str(delta_info["meta"])
        tomb_path = root / str(delta_info["tombstones"])
        if delta_info["base_sha256_annpack"] != base_info["sha256_annpack"]:
            raise ValueError(f"Delta base hash mismatch for {delta_path}")
        if _sha256_file(ann_path) != delta_info["sha256_annpack"]:
            raise ValueError(f"Delta annpack hash mismatch for {delta_path}")
        if _sha256_file(meta_path) != delta_info["sha256_meta"]:
            raise ValueError(f"Delta meta hash mismatch for {delta_path}")
        if _sha256_file(tomb_path) != delta_info["sha256_tombstones"]:
            raise ValueError(f"Delta tombstones hash mismatch for {delta_path}")

        shard_objs, _ = _open_pack_dir(
            delta_path,
            load_meta=load_meta,
            max_meta_bytes=max_meta_bytes,
            max_meta_line=max_meta_line,
        )
        for shard in shard_objs:
            if shard.meta:
                overridden_ids.update(shard.meta.keys())
        tombstoned_ids.update(_read_tombstones(tomb_path))
        info = DeltaInfo(
            seq=int(delta_info["seq"]),
            path=delta_path,
            annpack=ann_path,
            meta=meta_path,
            tombstones=tomb_path,
            base_sha256_annpack=delta_info["base_sha256_annpack"],
            sha256_annpack=delta_info["sha256_annpack"],
            sha256_meta=delta_info["sha256_meta"],
            sha256_tombstones=delta_info["sha256_tombstones"],
        )
        deltas.append((info, shard_objs))

    return PackSet(
        root_dir=root,
        base_shards=base_shards,
        deltas=deltas,
        tombstoned_ids=tombstoned_ids,
        overridden_ids=overridden_ids,
        dim=dim,
    )


def create_packset(base_dir: str, delta_dirs: Sequence[str], out_dir: str) -> Path:
    """Create a PackSet directory from a base pack and optional deltas."""
    base_src = Path(base_dir).expanduser().resolve()
    if not base_src.exists():
        raise FileNotFoundError(f"Base dir not found: {base_src}")
    root = Path(out_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)

    base_dst = root / "base"
    if base_dst.exists():
        raise ValueError(f"PackSet base already exists: {base_dst}")
    base_dst.mkdir(parents=True, exist_ok=True)

    for path in base_src.iterdir():
        if path.is_file():
            (base_dst / path.name).write_bytes(path.read_bytes())

    base_ann = base_dst / "pack.annpack"
    base_meta = base_dst / "pack.meta.jsonl"
    if not base_ann.exists() or not base_meta.exists():
        raise ValueError("Base pack must include pack.annpack and pack.meta.jsonl")

    root_manifest: Dict[str, Any] = {
        "schema_version": 3,
        "base": {
            "path": "base",
            "annpack": "base/pack.annpack",
            "meta": "base/pack.meta.jsonl",
            "sha256_annpack": _sha256_file(base_ann),
            "sha256_meta": _sha256_file(base_meta),
        },
        "deltas": [],
    }
    manifest_path = root / "pack.manifest.json"
    manifest_path.write_text(json.dumps(root_manifest, indent=2), encoding="utf-8")

    deltas_dir = root / "deltas"
    deltas_dir.mkdir(parents=True, exist_ok=True)
    for idx, delta in enumerate(delta_dirs, start=1):
        src = Path(delta).expanduser().resolve()
        if not src.exists():
            raise FileNotFoundError(f"Delta dir not found: {src}")
        dst = deltas_dir / f"{idx:04d}.delta"
        dst.mkdir(parents=True, exist_ok=True)
        for path in src.iterdir():
            if path.is_file():
                (dst / path.name).write_bytes(path.read_bytes())
        update_packset_manifest(str(root), str(dst), seq=idx)

    return manifest_path


def promote_delta(packset_dir: str, delta_dir: str) -> Path:
    """Add a delta to an existing PackSet, assigning the next seq."""
    root = Path(packset_dir).expanduser().resolve()
    manifest_path = root / "pack.manifest.json"
    data: Dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
    deltas: List[Dict[str, Any]] = data.get("deltas") or []
    next_seq = max([int(d["seq"]) for d in deltas], default=0) + 1

    src = Path(delta_dir).expanduser().resolve()
    dst = root / "deltas" / f"{next_seq:04d}.delta"
    dst.mkdir(parents=True, exist_ok=True)
    for path in src.iterdir():
        if path.is_file():
            (dst / path.name).write_bytes(path.read_bytes())

    return update_packset_manifest(str(root), str(dst), seq=next_seq)


def revert_packset(packset_dir: str, to_seq: int) -> Path:
    """Revert a PackSet manifest to a given delta sequence (inclusive)."""
    root = Path(packset_dir).expanduser().resolve()
    manifest_path = root / "pack.manifest.json"
    data: Dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
    if data.get("schema_version") != 3:
        raise ValueError("PackSet manifest must have schema_version=3")
    data["deltas"] = [d for d in (data.get("deltas") or []) if int(d["seq"]) <= to_seq]
    manifest_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return manifest_path


def _write_csv(rows: List[Dict[str, Any]], path: Path, id_col: str, text_col: str) -> None:
    import csv

    keys: Set[str] = set()
    for row in rows:
        keys.update(row.keys())
    if id_col not in keys or text_col not in keys:
        raise ValueError(f"Missing required columns: {id_col}, {text_col}")
    other = sorted(k for k in keys if k not in (id_col, text_col))
    fieldnames = [id_col, text_col] + other

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = {k: ("" if row.get(k) is None else str(row.get(k))) for k in fieldnames}
            writer.writerow(out)


def rebase_packset(
    packset_dir: str,
    out_dir: str,
    text_col: str = "text",
    id_col: str = "id",
    lists: int = 1024,
    seed: int = 0,
    offline: Optional[bool] = None,
) -> Path:
    """Rebuild a new base pack by compacting all deltas into base."""
    root = Path(packset_dir).expanduser().resolve()
    manifest_path = root / "pack.manifest.json"
    data: Dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
    if data.get("schema_version") != 3:
        raise ValueError("PackSet manifest must have schema_version=3")

    base_dir = root / data["base"]["path"]
    base_meta_path = base_dir / "pack.meta.jsonl"
    allow_large = os.getenv("ANNPACK_ALLOW_LARGE_META") == "1"
    base_meta = _load_meta(
        base_meta_path,
        max_bytes=DEFAULT_META_MAX_BYTES,
        max_line=DEFAULT_META_MAX_LINE,
        allow_large=allow_large,
    )

    tombstones: Set[int] = set()
    for delta in sorted(data.get("deltas") or [], key=lambda d: int(d["seq"])):
        delta_dir = root / delta["path"]
        meta_path = delta_dir / "pack.meta.jsonl"
        delta_meta = _load_meta(
            meta_path,
            max_bytes=DEFAULT_META_MAX_BYTES,
            max_line=DEFAULT_META_MAX_LINE,
            allow_large=allow_large,
        )
        base_meta.update(delta_meta)
        tombstones.update(_read_tombstones(delta_dir / "tombstones.jsonl"))

    for doc_id in tombstones:
        base_meta.pop(doc_id, None)

    rows = [base_meta[k] for k in sorted(base_meta.keys())]
    tmp_csv = root / "_rebase.csv"
    _write_csv(rows, tmp_csv, id_col=id_col, text_col=text_col)

    out_root = Path(out_dir).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    build_packset_base(
        input_csv=str(tmp_csv),
        packset_dir=str(out_root),
        text_col=text_col,
        id_col=id_col,
        lists=lists,
        seed=seed,
        offline=offline,
    )
    tmp_csv.unlink(missing_ok=True)
    return out_root / "pack.manifest.json"


def run_canary(
    base_dir: str,
    candidate_dir: str,
    queries_path: str,
    top_k: int = 5,
    min_overlap: float = 0.7,
    avg_overlap: float = 0.8,
) -> Dict[str, Any]:
    """Compare search results between base and candidate packs."""
    from .api import open_pack

    base = open_pack(base_dir)
    cand = open_pack(candidate_dir)
    try:
        overlaps: List[float] = []
        with Path(queries_path).open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row: Dict[str, Any] = json.loads(line)
                if "text" in row:
                    base_hits = base.search(row["text"], top_k=top_k)
                    cand_hits = cand.search(row["text"], top_k=top_k)
                elif "vector" in row:
                    base_hits = base.search_vec(row["vector"], top_k=top_k)
                    cand_hits = cand.search_vec(row["vector"], top_k=top_k)
                else:
                    continue
                base_ids = {h["id"] for h in base_hits}
                cand_ids = {h["id"] for h in cand_hits}
                if not base_ids and not cand_ids:
                    overlap = 1.0
                else:
                    overlap = len(base_ids & cand_ids) / float(top_k)
                overlaps.append(overlap)
        if not overlaps:
            raise ValueError("No queries found in canary file")
        avg = sum(overlaps) / len(overlaps)
        min_val = min(overlaps)
        result: Dict[str, Any] = {
            "avg_overlap": avg,
            "min_overlap": min_val,
            "queries": len(overlaps),
        }
        if avg < avg_overlap or min_val < min_overlap:
            raise ValueError(json.dumps(result))
        return result
    finally:
        base.close()
        cand.close()
