import os
from pathlib import Path

import pytest

from annpack.api import build_pack, open_pack


def _write_csv(path: Path) -> None:
    path.write_text("id,text\n0,hello\n1,paris is france\n", encoding="utf-8")


def test_build_deterministic(tmp_path):
    os.environ["ANNPACK_OFFLINE"] = "1"
    os.environ["ANNPACK_FAISS_THREADS"] = "1"
    csv_path = tmp_path / "tiny.csv"
    _write_csv(csv_path)

    out1 = tmp_path / "out1"
    out2 = tmp_path / "out2"
    build_pack(
        str(csv_path), str(out1), text_col="text", id_col="id", lists=4, seed=123, offline=True
    )
    build_pack(
        str(csv_path), str(out2), text_col="text", id_col="id", lists=4, seed=123, offline=True
    )

    assert (out1 / "pack.manifest.json").read_bytes() == (out2 / "pack.manifest.json").read_bytes()
    assert (out1 / "pack.meta.jsonl").read_bytes() == (out2 / "pack.meta.jsonl").read_bytes()
    assert (out1 / "pack.annpack").read_bytes() == (out2 / "pack.annpack").read_bytes()


def test_open_pack_search(tmp_path):
    os.environ["ANNPACK_OFFLINE"] = "1"
    os.environ["ANNPACK_FAISS_THREADS"] = "1"
    csv_path = tmp_path / "tiny.csv"
    _write_csv(csv_path)

    out = tmp_path / "out"
    build_pack(
        str(csv_path), str(out), text_col="text", id_col="id", lists=4, seed=123, offline=True
    )
    pack = open_pack(str(out))
    results = pack.search("hello", top_k=2)
    pack.close()

    assert isinstance(results, list)
    assert results
    for row in results:
        assert "id" in row
        assert "score" in row
        assert "shard" in row


def test_open_pack_meta_limits(tmp_path):
    os.environ["ANNPACK_OFFLINE"] = "1"
    os.environ["ANNPACK_FAISS_THREADS"] = "1"
    csv_path = tmp_path / "tiny.csv"
    _write_csv(csv_path)

    out = tmp_path / "out"
    build_pack(
        str(csv_path), str(out), text_col="text", id_col="id", lists=4, seed=123, offline=True
    )

    meta_path = out / "pack.meta.jsonl"
    meta_path.write_text('{"id":0,"text":"' + ("x" * 200) + '"}\n', encoding="utf-8")

    with pytest.raises(ValueError):
        open_pack(str(out), max_meta_line=50)

    meta_path.write_text('{"id":0,"text":"ok"}\n', encoding="utf-8")
    with pytest.raises(ValueError):
        open_pack(str(out), max_meta_bytes=10)


def test_open_pack_streaming_meta(tmp_path):
    os.environ["ANNPACK_OFFLINE"] = "1"
    os.environ["ANNPACK_FAISS_THREADS"] = "1"
    csv_path = tmp_path / "tiny.csv"
    _write_csv(csv_path)

    out = tmp_path / "out"
    build_pack(
        str(csv_path), str(out), text_col="text", id_col="id", lists=4, seed=123, offline=True
    )
    pack = open_pack(str(out), stream_meta=True)
    results = pack.search("hello", top_k=2)
    pack.close()
    assert results


def test_open_pack_load_meta_flag(tmp_path):
    os.environ["ANNPACK_OFFLINE"] = "1"
    os.environ["ANNPACK_FAISS_THREADS"] = "1"
    os.environ.pop("ANNPACK_ALLOW_LARGE_META", None)
    csv_path = tmp_path / "tiny.csv"
    _write_csv(csv_path)

    out = tmp_path / "out"
    build_pack(
        str(csv_path), str(out), text_col="text", id_col="id", lists=4, seed=123, offline=True
    )

    with pytest.raises(ValueError):
        open_pack(str(out), max_meta_bytes=1)

    pack = open_pack(str(out), load_meta=False, max_meta_bytes=1)
    results = pack.search("hello", top_k=1)
    pack.close()
    assert results[0]["meta"] is None


def test_tiny_build_deterministic_fallback(tmp_path):
    os.environ["ANNPACK_OFFLINE"] = "1"
    os.environ["ANNPACK_FAISS_THREADS"] = "1"
    csv_path = tmp_path / "tiny3.csv"
    csv_path.write_text(
        "id,text\n0,alpha\n1,beta\n2,gamma\n",
        encoding="utf-8",
    )
    out1 = tmp_path / "out1"
    out2 = tmp_path / "out2"
    build_pack(
        str(csv_path), str(out1), text_col="text", id_col="id", lists=8, seed=42, offline=True
    )
    build_pack(
        str(csv_path), str(out2), text_col="text", id_col="id", lists=8, seed=42, offline=True
    )
    assert (out1 / "pack.manifest.json").read_bytes() == (out2 / "pack.manifest.json").read_bytes()
    assert (out1 / "pack.meta.jsonl").read_bytes() == (out2 / "pack.meta.jsonl").read_bytes()
    assert (out1 / "pack.annpack").read_bytes() == (out2 / "pack.annpack").read_bytes()
