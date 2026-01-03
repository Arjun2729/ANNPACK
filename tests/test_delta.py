import os
import shutil
from pathlib import Path

import pytest

from annpack.api import open_pack
from annpack.packset import build_packset_base, build_delta, update_packset_manifest


def _write_csv(path: Path, rows: str) -> None:
    path.write_text(rows, encoding="utf-8")


def test_packset_delta_merge(tmp_path):
    os.environ["ANNPACK_OFFLINE"] = "1"
    base_csv = tmp_path / "base.csv"
    _write_csv(base_csv, "id,text\n0,hello\n1,paris is france\n")

    packset_dir = tmp_path / "packset"
    build_packset_base(
        str(base_csv),
        str(packset_dir),
        text_col="text",
        id_col="id",
        lists=4,
        seed=123,
        offline=True,
    )

    delta_csv = tmp_path / "delta.csv"
    _write_csv(delta_csv, "id,text\n0,hello updated\n2,delta add\n")
    delta_dir = packset_dir / "deltas" / "0001.delta"

    build_delta(
        base_dir=str(packset_dir / "base"),
        add_csv=str(delta_csv),
        delete_ids=[1],
        out_delta_dir=str(delta_dir),
        text_col="text",
        id_col="id",
        lists=4,
        seed=123,
        offline=True,
    )
    update_packset_manifest(str(packset_dir), str(delta_dir), seq=1)

    pack = open_pack(str(packset_dir))
    hits_add = pack.search("delta add", top_k=3)
    hits_update = pack.search("hello updated", top_k=3)
    hits_deleted = pack.search("paris is france", top_k=3)
    pack.close()

    assert any(h["id"] == 2 for h in hits_add)
    updated = next((h for h in hits_update if h["id"] == 0), None)
    assert updated is not None
    assert updated.get("meta", {}).get("text") == "hello updated"
    assert all(h["id"] != 1 for h in hits_deleted)


def test_delta_determinism(tmp_path):
    os.environ["ANNPACK_OFFLINE"] = "1"
    base_csv = tmp_path / "base.csv"
    _write_csv(base_csv, "id,text\n0,hello\n1,paris is france\n")
    packset_dir = tmp_path / "packset"
    build_packset_base(
        str(base_csv),
        str(packset_dir),
        text_col="text",
        id_col="id",
        lists=4,
        seed=123,
        offline=True,
    )

    delta_csv = tmp_path / "delta.csv"
    _write_csv(delta_csv, "id,text\n0,hello updated\n2,delta add\n")
    delta_a = tmp_path / "delta_a"
    delta_b = tmp_path / "delta_b"

    build_delta(
        base_dir=str(packset_dir / "base"),
        add_csv=str(delta_csv),
        delete_ids=[1],
        out_delta_dir=str(delta_a),
        text_col="text",
        id_col="id",
        lists=4,
        seed=123,
        offline=True,
    )
    build_delta(
        base_dir=str(packset_dir / "base"),
        add_csv=str(delta_csv),
        delete_ids=[1],
        out_delta_dir=str(delta_b),
        text_col="text",
        id_col="id",
        lists=4,
        seed=123,
        offline=True,
    )

    assert (delta_a / "pack.annpack").read_bytes() == (delta_b / "pack.annpack").read_bytes()
    assert (delta_a / "pack.meta.jsonl").read_bytes() == (delta_b / "pack.meta.jsonl").read_bytes()
    assert (delta_a / "tombstones.jsonl").read_bytes() == (
        delta_b / "tombstones.jsonl"
    ).read_bytes()
    assert (delta_a / "delta.manifest.json").read_bytes() == (
        delta_b / "delta.manifest.json"
    ).read_bytes()


def test_packset_base_hash_mismatch(tmp_path):
    os.environ["ANNPACK_OFFLINE"] = "1"
    base_csv = tmp_path / "base.csv"
    _write_csv(base_csv, "id,text\n0,hello\n1,paris is france\n")
    packset_dir = tmp_path / "packset"
    build_packset_base(
        str(base_csv),
        str(packset_dir),
        text_col="text",
        id_col="id",
        lists=4,
        seed=123,
        offline=True,
    )

    delta_csv = tmp_path / "delta.csv"
    _write_csv(delta_csv, "id,text\n0,hello updated\n")
    delta_dir = packset_dir / "deltas" / "0001.delta"
    build_delta(
        base_dir=str(packset_dir / "base"),
        add_csv=str(delta_csv),
        delete_ids=[],
        out_delta_dir=str(delta_dir),
        text_col="text",
        id_col="id",
        lists=4,
        seed=123,
        offline=True,
    )
    update_packset_manifest(str(packset_dir), str(delta_dir), seq=1)

    alt_csv = tmp_path / "alt.csv"
    _write_csv(alt_csv, "id,text\n9,alt doc\n")
    alt_packset = tmp_path / "packset_alt"
    build_packset_base(
        str(alt_csv),
        str(alt_packset),
        text_col="text",
        id_col="id",
        lists=4,
        seed=123,
        offline=True,
    )

    shutil.copyfile(alt_packset / "base" / "pack.annpack", packset_dir / "base" / "pack.annpack")
    shutil.copyfile(
        alt_packset / "base" / "pack.meta.jsonl", packset_dir / "base" / "pack.meta.jsonl"
    )

    with pytest.raises(ValueError, match="Base annpack hash mismatch"):
        open_pack(str(packset_dir))


def test_packset_multi_delta_semantics(tmp_path):
    os.environ["ANNPACK_OFFLINE"] = "1"
    base_csv = tmp_path / "base.csv"
    _write_csv(base_csv, "id,text\n0,hello v0\n1,paris is france\n2,delete me\n")
    packset_dir = tmp_path / "packset"
    build_packset_base(
        str(base_csv),
        str(packset_dir),
        text_col="text",
        id_col="id",
        lists=4,
        seed=123,
        offline=True,
    )

    delta1_csv = tmp_path / "delta1.csv"
    _write_csv(delta1_csv, "id,text\n0,hello v1\n")
    delta1_dir = packset_dir / "deltas" / "0001.delta"
    build_delta(
        base_dir=str(packset_dir / "base"),
        add_csv=str(delta1_csv),
        delete_ids=[],
        out_delta_dir=str(delta1_dir),
        text_col="text",
        id_col="id",
        lists=4,
        seed=123,
        offline=True,
    )
    update_packset_manifest(str(packset_dir), str(delta1_dir), seq=1)

    delta2_csv = tmp_path / "delta2.csv"
    _write_csv(delta2_csv, "id,text\n0,hello v2\n")
    delta2_dir = packset_dir / "deltas" / "0002.delta"
    build_delta(
        base_dir=str(packset_dir / "base"),
        add_csv=str(delta2_csv),
        delete_ids=[2],
        out_delta_dir=str(delta2_dir),
        text_col="text",
        id_col="id",
        lists=4,
        seed=123,
        offline=True,
    )
    update_packset_manifest(str(packset_dir), str(delta2_dir), seq=2)

    pack = open_pack(str(packset_dir))
    hits = pack.search("hello", top_k=10)
    pack.close()

    ids = [h["id"] for h in hits]
    assert ids.count(0) == 1
    latest = next(h for h in hits if h["id"] == 0)
    assert latest.get("meta", {}).get("text") == "hello v2"
    assert 2 not in ids
