from __future__ import annotations

import hashlib
import json
import os
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

from .build import build_index
from .reader import ANNPackIndex


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_header(path: Path) -> dict:
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
    candidates = list(pack_dir.glob("*.manifest.json")) + list(pack_dir.glob("manifest.json"))
    if not candidates:
        raise FileNotFoundError(f"No manifest found in {pack_dir}")
    return candidates[0]


def _load_meta(meta_path: Path) -> Dict[int, dict]:
    meta: Dict[int, dict] = {}
    with meta_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if "id" not in row:
                continue
            meta[int(row["id"])] = row
    return meta


def _hash_seed(text: str, seed: int) -> int:
    h = hashlib.sha256(f"{seed}:{text}".encode("utf-8")).digest()
    return int.from_bytes(h[:8], "little", signed=False)


def _normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm == 0:
        raise ValueError("Zero vector")
    return vec / norm


def _offline_embed(text: str, dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(_hash_seed(text, seed))
    vec = rng.standard_normal((dim,), dtype=np.float32)
    return _normalize(vec)


def _read_tombstones(path: Path) -> Set[int]:
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
    meta: Dict[int, dict]


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
        self._model: Optional[object] = None

    def _embed_query(self, text: str) -> np.ndarray:
        if os.environ.get("ANNPACK_OFFLINE") == "1":
            return _offline_embed(text, self._dim, self._seed)
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as e:
                raise ImportError("SentenceTransformer not installed. Install with: pip install annpack[embed]") from e
            self._model = SentenceTransformer("all-MiniLM-L6-v2")
        vec = self._model.encode([text], normalize_embeddings=True)[0].astype(np.float32)
        return vec

    def _search_shards(self, shards: Sequence[_Shard], vec: np.ndarray, k: int) -> List[dict]:
        results: List[dict] = []
        for shard in shards:
            for doc_id, score in shard.index.search(vec, k=k):
                results.append(
                    {
                        "id": doc_id,
                        "score": float(score),
                        "shard": shard.name,
                        "meta": shard.meta.get(doc_id),
                    }
                )
        results.sort(key=lambda r: (-r["score"], r["shard"], r["id"]))
        return results[:k]

    def search(self, query_text: str, top_k: int = 5) -> List[dict]:
        vec = self._embed_query(query_text)
        return self.search_vec(vec, top_k=top_k)

    def search_vec(self, vector: Iterable[float], top_k: int = 5) -> List[dict]:
        vec = np.asarray(list(vector), dtype=np.float32)
        if vec.ndim != 1 or vec.shape[0] != self._dim:
            raise ValueError(f"Vector must be 1-D of length {self._dim}")
        vec = _normalize(vec)

        per_pack_k = max(top_k * 5, top_k)
        seen: Set[int] = set()
        results: List[dict] = []

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
        for shard in self._base_shards:
            shard.index.close()
        for _, shards in self._deltas:
            for shard in shards:
                shard.index.close()

    def __enter__(self) -> "PackSet":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.close()
        return False


def _open_pack_dir(pack_dir: Path) -> Tuple[List[_Shard], int]:
    manifest_path = None
    shards = []
    try:
        manifest_path = _find_manifest(pack_dir)
    except FileNotFoundError:
        manifest_path = None

    if manifest_path is not None:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        shards = data.get("shards") or []
    if not shards:
        ann_path = pack_dir / "pack.annpack"
        meta_path = pack_dir / "pack.meta.jsonl"
        if not ann_path.exists() or not meta_path.exists():
            raise ValueError(f"Pack dir missing manifest and pack.* files: {pack_dir}")
        shards = [{"name": "pack", "annpack": "pack.annpack", "meta": "pack.meta.jsonl"}]

    shard_objs: List[_Shard] = []
    dim = None
    for shard in shards:
        ann_path = pack_dir / shard["annpack"]
        meta_path = pack_dir / shard["meta"]
        index = ANNPackIndex.open(str(ann_path))
        if dim is None:
            dim = index.header.dim
        meta = _load_meta(meta_path)
        shard_objs.append(_Shard(name=shard.get("name", ann_path.name), index=index, meta=meta))
    return shard_objs, dim or 0


def build_packset_base(
    input_csv: str,
    packset_dir: str,
    text_col: str = "text",
    id_col: str = "id",
    lists: int = 1024,
    seed: int = 0,
    offline: Optional[bool] = None,
    **kwargs,
) -> dict:
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
        manifest = {
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

    root_manifest = {
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
    **kwargs,
) -> DeltaInfo:
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
    delta_manifest = {
        "schema_version": 3,
        "base_sha256_annpack": info.base_sha256_annpack,
        "sha256_annpack": info.sha256_annpack,
        "sha256_meta": info.sha256_meta,
        "sha256_tombstones": info.sha256_tombstones,
    }
    (delta_dir / "delta.manifest.json").write_text(json.dumps(delta_manifest, indent=2), encoding="utf-8")
    return info


def update_packset_manifest(packset_dir: str, delta_dir: str, seq: int) -> Path:
    root = Path(packset_dir).expanduser().resolve()
    manifest_path = root / "pack.manifest.json"
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    if data.get("schema_version") != 3:
        raise ValueError("PackSet manifest must have schema_version=3")

    base_sha = data["base"]["sha256_annpack"]
    delta_path = Path(delta_dir).expanduser().resolve()
    ann_path = delta_path / "pack.annpack"
    meta_path = delta_path / "pack.meta.jsonl"
    tomb_path = delta_path / "tombstones.jsonl"
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

    deltas = data.get("deltas") or []
    deltas.append(delta_entry)
    deltas = sorted(deltas, key=lambda d: d["seq"])
    data["deltas"] = deltas
    manifest_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return manifest_path


def open_packset(packset_dir: str) -> PackSet:
    root = Path(packset_dir).expanduser().resolve()
    manifest_path = root / "pack.manifest.json"
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    if data.get("schema_version") != 3:
        raise ValueError("Not a PackSet manifest")

    base_info = data["base"]
    base_dir = root / base_info["path"]
    base_ann = root / base_info["annpack"]
    base_meta = root / base_info["meta"]
    if _sha256_file(base_ann) != base_info["sha256_annpack"]:
        raise ValueError("Base annpack hash mismatch")
    if _sha256_file(base_meta) != base_info["sha256_meta"]:
        raise ValueError("Base meta hash mismatch")

    base_shards, dim = _open_pack_dir(base_dir)

    tombstoned_ids: Set[int] = set()
    overridden_ids: Set[int] = set()
    deltas: List[Tuple[DeltaInfo, List[_Shard]]] = []

    for delta in data.get("deltas") or []:
        delta_path = root / delta["path"]
        ann_path = root / delta["annpack"]
        meta_path = root / delta["meta"]
        tomb_path = root / delta["tombstones"]
        if delta["base_sha256_annpack"] != base_info["sha256_annpack"]:
            raise ValueError(f"Delta base hash mismatch for {delta_path}")
        if _sha256_file(ann_path) != delta["sha256_annpack"]:
            raise ValueError(f"Delta annpack hash mismatch for {delta_path}")
        if _sha256_file(meta_path) != delta["sha256_meta"]:
            raise ValueError(f"Delta meta hash mismatch for {delta_path}")
        if _sha256_file(tomb_path) != delta["sha256_tombstones"]:
            raise ValueError(f"Delta tombstones hash mismatch for {delta_path}")

        shard_objs, _ = _open_pack_dir(delta_path)
        for shard in shard_objs:
            overridden_ids.update(shard.meta.keys())
        tombstoned_ids.update(_read_tombstones(tomb_path))
        info = DeltaInfo(
            seq=int(delta["seq"]),
            path=delta_path,
            annpack=ann_path,
            meta=meta_path,
            tombstones=tomb_path,
            base_sha256_annpack=delta["base_sha256_annpack"],
            sha256_annpack=delta["sha256_annpack"],
            sha256_meta=delta["sha256_meta"],
            sha256_tombstones=delta["sha256_tombstones"],
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
