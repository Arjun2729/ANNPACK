import os
from pathlib import Path

from annpack.api import build_pack, open_pack


def _write_csv(path: Path) -> None:
    path.write_text("id,text\n0,hello\n1,paris is france\n", encoding="utf-8")


def test_build_deterministic(tmp_path):
    os.environ["ANNPACK_OFFLINE"] = "1"
    csv_path = tmp_path / "tiny.csv"
    _write_csv(csv_path)

    out1 = tmp_path / "out1"
    out2 = tmp_path / "out2"
    build_pack(str(csv_path), str(out1), text_col="text", id_col="id", lists=4, seed=123, offline=True)
    build_pack(str(csv_path), str(out2), text_col="text", id_col="id", lists=4, seed=123, offline=True)

    assert (out1 / "pack.manifest.json").read_bytes() == (out2 / "pack.manifest.json").read_bytes()
    assert (out1 / "pack.meta.jsonl").read_bytes() == (out2 / "pack.meta.jsonl").read_bytes()
    assert (out1 / "pack.annpack").read_bytes() == (out2 / "pack.annpack").read_bytes()


def test_open_pack_search(tmp_path):
    os.environ["ANNPACK_OFFLINE"] = "1"
    csv_path = tmp_path / "tiny.csv"
    _write_csv(csv_path)

    out = tmp_path / "out"
    build_pack(str(csv_path), str(out), text_col="text", id_col="id", lists=4, seed=123, offline=True)
    pack = open_pack(str(out))
    results = pack.search("hello", top_k=2)
    pack.close()

    assert isinstance(results, list)
    assert results
    for row in results:
        assert "id" in row
        assert "score" in row
        assert "shard" in row
