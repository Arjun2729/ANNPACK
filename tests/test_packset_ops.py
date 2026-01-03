import json
import os
from pathlib import Path

import pytest

from annpack.packset import create_packset, promote_delta, revert_packset, run_canary
from annpack.api import build_pack


def _write_csv(path: Path, rows: str) -> None:
    path.write_text(rows, encoding="utf-8")


def _build_pack(tmp_path: Path, name: str, rows: str) -> Path:
    csv_path = tmp_path / f"{name}.csv"
    _write_csv(csv_path, rows)
    out_dir = tmp_path / name
    build_pack(
        str(csv_path), str(out_dir), text_col="text", id_col="id", lists=4, seed=0, offline=True
    )
    return out_dir


def test_packset_create_promote_revert(tmp_path):
    os.environ["ANNPACK_OFFLINE"] = "1"
    base_dir = _build_pack(tmp_path, "base", "id,text\n0,hello\n1,paris\n")
    delta_dir = _build_pack(tmp_path, "delta", "id,text\n0,hello updated\n")

    packset_dir = tmp_path / "packset"
    manifest = create_packset(str(base_dir), [str(delta_dir)], str(packset_dir))
    data = json.loads(Path(manifest).read_text(encoding="utf-8"))
    assert data["schema_version"] == 3
    assert len(data["deltas"]) == 1

    delta2_dir = _build_pack(tmp_path, "delta2", "id,text\n2,new\n")
    manifest2 = promote_delta(str(packset_dir), str(delta2_dir))
    data2 = json.loads(Path(manifest2).read_text(encoding="utf-8"))
    assert len(data2["deltas"]) == 2

    manifest3 = revert_packset(str(packset_dir), to_seq=1)
    data3 = json.loads(Path(manifest3).read_text(encoding="utf-8"))
    assert len(data3["deltas"]) == 1


def test_canary(tmp_path):
    os.environ["ANNPACK_OFFLINE"] = "1"
    base_dir = _build_pack(tmp_path, "base", "id,text\n0,hello\n1,paris\n")
    cand_dir = _build_pack(tmp_path, "cand", "id,text\n0,hello\n1,paris\n")

    queries = tmp_path / "queries.jsonl"
    queries.write_text(json.dumps({"text": "hello"}) + "\n", encoding="utf-8")

    result = run_canary(
        str(base_dir), str(cand_dir), str(queries), top_k=2, min_overlap=0.5, avg_overlap=0.5
    )
    assert result["queries"] == 1

    with pytest.raises(ValueError):
        run_canary(
            str(base_dir), str(cand_dir), str(queries), top_k=2, min_overlap=1.01, avg_overlap=1.01
        )
