import json
import os
from pathlib import Path

import pytest

from annpack.api import build_pack
from annpack.verify import (
    _validate_offsets,
    inspect_pack,
    sign_manifest,
    verify_pack,
    verify_manifest_signature,
)

try:
    from hypothesis import given, strategies as st  # type: ignore
except Exception:
    pytest.skip("hypothesis not installed", allow_module_level=True)


def _write_csv(path: Path) -> None:
    path.write_text("id,text\n0,hello\n1,paris is france\n", encoding="utf-8")


def test_verify_ok(tmp_path):
    os.environ["ANNPACK_OFFLINE"] = "1"
    csv_path = tmp_path / "tiny.csv"
    _write_csv(csv_path)
    out_dir = tmp_path / "out"
    build_pack(
        str(csv_path), str(out_dir), text_col="text", id_col="id", lists=4, seed=0, offline=True
    )
    result = verify_pack(str(out_dir), deep=True)
    assert result["ok"] is True


def test_verify_truncated(tmp_path):
    os.environ["ANNPACK_OFFLINE"] = "1"
    csv_path = tmp_path / "tiny.csv"
    _write_csv(csv_path)
    out_dir = tmp_path / "out"
    build_pack(
        str(csv_path), str(out_dir), text_col="text", id_col="id", lists=4, seed=0, offline=True
    )
    ann_path = out_dir / "pack.annpack"
    bad_path = out_dir / "pack_trunc.annpack"
    data = ann_path.read_bytes()
    bad_path.write_bytes(data[:64])

    manifest = out_dir / "pack.manifest.json"
    manifest_data = json.loads(manifest.read_text())
    manifest_data["shards"][0]["annpack"] = bad_path.name
    manifest.write_text(json.dumps(manifest_data), encoding="utf-8")

    with pytest.raises(ValueError):
        verify_pack(str(out_dir), deep=False)


def test_inspect(tmp_path):
    os.environ["ANNPACK_OFFLINE"] = "1"
    csv_path = tmp_path / "tiny.csv"
    _write_csv(csv_path)
    out_dir = tmp_path / "out"
    build_pack(
        str(csv_path), str(out_dir), text_col="text", id_col="id", lists=4, seed=0, offline=True
    )
    info = inspect_pack(str(out_dir))
    assert "manifest" in info


def test_sign_manifest(tmp_path):
    os.environ["ANNPACK_OFFLINE"] = "1"
    csv_path = tmp_path / "tiny.csv"
    _write_csv(csv_path)
    out_dir = tmp_path / "out"
    build_pack(
        str(csv_path), str(out_dir), text_col="text", id_col="id", lists=4, seed=0, offline=True
    )

    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from cryptography.hazmat.primitives.serialization import Encoding, PrivateFormat, NoEncryption

    key = Ed25519PrivateKey.generate()
    key_path = tmp_path / "key.pem"
    key_path.write_bytes(
        key.private_bytes(
            encoding=Encoding.PEM,
            format=PrivateFormat.PKCS8,
            encryption_algorithm=NoEncryption(),
        )
    )

    sig_path = sign_manifest(str(out_dir), str(key_path))
    assert Path(sig_path).exists()
    assert Path(sig_path).stat().st_size > 0

    from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

    pub_path = tmp_path / "key.pub"
    pub_path.write_bytes(
        key.public_key().public_bytes(
            encoding=Encoding.PEM, format=PublicFormat.SubjectPublicKeyInfo
        )
    )
    assert verify_manifest_signature(str(out_dir), str(pub_path), sig_path=str(sig_path))


@given(
    file_size=st.integers(min_value=1, max_value=1000),
    offset=st.integers(min_value=0, max_value=1000),
    length=st.integers(min_value=0, max_value=1000),
)
def test_validate_offsets(file_size: int, offset: int, length: int):
    offsets = [offset]
    lengths = [length]
    if length != 0 and offset + length > file_size:
        with pytest.raises(ValueError):
            _validate_offsets(file_size, offsets, lengths)
    else:
        _validate_offsets(file_size, offsets, lengths)
