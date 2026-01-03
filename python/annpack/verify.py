"""Verification and inspection helpers for ANNPack artifacts."""

from __future__ import annotations

import json
import os
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from importlib import metadata

from .packset import _sha256_file


@dataclass
class PackFileInfo:
    path: Path
    size: int
    header: Dict[str, int]


def _read_header(path: Path) -> Dict[str, int]:
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


def _validate_offsets(file_size: int, offsets: List[int], lengths: List[int]) -> None:
    for off, length in zip(offsets, lengths):
        if off < 0 or length < 0:
            raise ValueError("Negative offset or length")
        if length == 0:
            continue
        if off + length > file_size:
            raise ValueError("List offset/length exceeds file size")


def _verify_annpack(path: Path, deep: bool = False) -> PackFileInfo:
    size = path.stat().st_size
    header = _read_header(path)
    if header["magic"] != 0x504E4E41:
        raise ValueError(f"Bad magic for {path}")
    if header["header_size"] != 72:
        raise ValueError(f"Unsupported header size for {path}")
    if header["offset_table_pos"] <= 0:
        raise ValueError(f"Missing offset table for {path}")

    n_lists = int(header["n_lists"])
    table_size = n_lists * 16
    if header["offset_table_pos"] + table_size > size:
        raise ValueError("Offset table exceeds file size")

    offsets: List[int] = []
    lengths: List[int] = []
    with path.open("rb") as handle:
        handle.seek(header["offset_table_pos"])
        table = handle.read(table_size)
    for i in range(n_lists):
        off, length = struct.unpack_from("<QQ", table, i * 16)
        offsets.append(int(off))
        lengths.append(int(length))

    _validate_offsets(size, offsets, lengths)

    if deep:
        dim = int(header["dim"])
        with path.open("rb") as handle:
            for off, length in zip(offsets, lengths):
                if length == 0:
                    continue
                if length < 4:
                    raise ValueError("List length too small")
                handle.seek(off)
                count_bytes = handle.read(4)
                count = struct.unpack("<I", count_bytes)[0]
                needed = 4 + count * 8 + count * dim * 2
                if needed > length:
                    raise ValueError("List payload exceeds recorded length")

    return PackFileInfo(path=path, size=size, header=header)


def verify_pack(pack_dir: str, deep: bool = False) -> Dict[str, Any]:
    base = Path(pack_dir).expanduser().resolve()
    manifest_path = _find_manifest(base)
    data = json.loads(manifest_path.read_text(encoding="utf-8"))

    if data.get("schema_version") == 3:
        base_info = data["base"]
        base_ann = base / base_info["annpack"]
        base_meta = base / base_info["meta"]
        _verify_annpack(base_ann, deep=deep)
        if not base_meta.exists():
            raise ValueError("Base meta missing")
        if "sha256_annpack" in base_info:
            if _sha256_file(base_ann) != base_info["sha256_annpack"]:
                raise ValueError("Base annpack hash mismatch")
        if "sha256_meta" in base_info:
            if _sha256_file(base_meta) != base_info["sha256_meta"]:
                raise ValueError("Base meta hash mismatch")

        for delta in data.get("deltas") or []:
            ann_path = base / delta["annpack"]
            meta_path = base / delta["meta"]
            tomb_path = base / delta["tombstones"]
            _verify_annpack(ann_path, deep=deep)
            if not meta_path.exists() or not tomb_path.exists():
                raise ValueError("Delta meta/tombstones missing")
            if "sha256_annpack" in delta:
                if _sha256_file(ann_path) != delta["sha256_annpack"]:
                    raise ValueError("Delta annpack hash mismatch")
            if "sha256_meta" in delta:
                if _sha256_file(meta_path) != delta["sha256_meta"]:
                    raise ValueError("Delta meta hash mismatch")
            if "sha256_tombstones" in delta:
                if _sha256_file(tomb_path) != delta["sha256_tombstones"]:
                    raise ValueError("Delta tombstones hash mismatch")
        return {"ok": True, "manifest": str(manifest_path)}

    shards = data.get("shards") or []
    if not shards:
        raise ValueError("Manifest contains no shards")

    for shard in shards:
        ann_path = base / shard["annpack"]
        meta_path = base / shard["meta"]
        _verify_annpack(ann_path, deep=deep)
        if not meta_path.exists():
            raise ValueError(f"Missing meta: {meta_path}")

    return {"ok": True, "manifest": str(manifest_path)}


def inspect_pack(pack_dir: str) -> Dict[str, Any]:
    base = Path(pack_dir).expanduser().resolve()
    manifest_path = _find_manifest(base)
    data = json.loads(manifest_path.read_text(encoding="utf-8"))

    if data.get("schema_version") == 3:
        base_info = data["base"]
        base_ann = base / base_info["annpack"]
        base_meta = base / base_info["meta"]
        base_file = _verify_annpack(base_ann, deep=False)
        deltas: List[Dict[str, Any]] = []
        for delta in data.get("deltas") or []:
            ann_path = base / delta["annpack"]
            meta_path = base / delta["meta"]
            tomb_path = base / delta["tombstones"]
            deltas.append(
                {
                    "seq": delta["seq"],
                    "annpack": str(ann_path),
                    "meta": str(meta_path),
                    "tombstones": str(tomb_path),
                    "annpack_size": ann_path.stat().st_size if ann_path.exists() else 0,
                }
            )
        return {
            "schema_version": 3,
            "manifest": str(manifest_path),
            "base": {
                "annpack": str(base_ann),
                "meta": str(base_meta),
                "header": base_file.header,
                "size": base_file.size,
            },
            "deltas": deltas,
        }

    shards = data.get("shards") or []
    if not shards:
        raise ValueError("Manifest contains no shards")

    shard_info: List[Dict[str, Any]] = []
    for shard in shards:
        ann_path = base / shard["annpack"]
        meta_path = base / shard["meta"]
        info = _verify_annpack(ann_path, deep=False)
        shard_info.append(
            {
                "name": shard.get("name"),
                "annpack": str(ann_path),
                "meta": str(meta_path),
                "header": info.header,
                "size": info.size,
            }
        )
    return {
        "schema_version": data.get("schema_version", 2),
        "manifest": str(manifest_path),
        "shards": shard_info,
    }


def diagnose_env() -> Dict[str, Any]:
    def _version(pkg: str) -> Optional[str]:
        try:
            return metadata.version(pkg)
        except metadata.PackageNotFoundError:
            return None

    return {
        "python": sys.version,
        "platform": sys.platform,
        "annpack": _version("annpack"),
        "numpy": _version("numpy"),
        "faiss": _version("faiss-cpu"),
        "polars": _version("polars"),
        "torch": _version("torch"),
        "sentence_transformers": _version("sentence-transformers"),
        "ANNPACK_OFFLINE": os.environ.get("ANNPACK_OFFLINE"),
    }


def sign_manifest(pack_dir: str, key_path: str, out_path: Optional[str] = None) -> str:
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from cryptography.hazmat.primitives.serialization import load_pem_private_key

    base = Path(pack_dir).expanduser().resolve()
    manifest_path = _find_manifest(base)
    data = manifest_path.read_bytes()

    key_bytes = Path(key_path).read_bytes()
    key = load_pem_private_key(key_bytes, password=None)
    if not isinstance(key, Ed25519PrivateKey):
        raise ValueError("Expected Ed25519 private key")

    sig = key.sign(data)
    if out_path:
        sig_path = Path(out_path)
    else:
        sig_path = manifest_path.with_suffix(manifest_path.suffix + ".sig")
    sig_path.write_bytes(sig)
    return str(sig_path)


def verify_manifest_signature(
    pack_dir: str, pubkey_path: str, sig_path: Optional[str] = None
) -> bool:
    """Verify a manifest signature with an Ed25519 public key."""
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    from cryptography.hazmat.primitives.serialization import load_pem_public_key

    base = Path(pack_dir).expanduser().resolve()
    manifest_path = _find_manifest(base)
    data = manifest_path.read_bytes()

    if sig_path:
        sig_file = Path(sig_path)
    else:
        sig_file = manifest_path.with_suffix(manifest_path.suffix + ".sig")
    if not sig_file.exists():
        raise FileNotFoundError(f"Signature not found: {sig_file}")

    key_bytes = Path(pubkey_path).read_bytes()
    key = load_pem_public_key(key_bytes)
    if not isinstance(key, Ed25519PublicKey):
        raise ValueError("Expected Ed25519 public key")
    key.verify(sig_file.read_bytes(), data)
    return True
