#!/usr/bin/env python3
"""Build a tiny pack and run a search using the Python API."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from annpack.api import build_pack, open_pack


def main() -> None:
    work = Path(tempfile.mkdtemp(prefix="annpack_example_"))
    csv_path = work / "tiny_docs.csv"
    csv_path.write_text("id,text\n0,hello\n1,paris is france\n", encoding="utf-8")

    os.environ["ANNPACK_OFFLINE"] = "1"
    pack_dir = work / "pack"
    build_pack(
        input_csv=str(csv_path),
        output_dir=str(pack_dir),
        text_col="text",
        id_col="id",
        lists=4,
        seed=0,
        offline=True,
    )

    pack = open_pack(str(pack_dir))
    try:
        hits = pack.search("hello", top_k=2)
        print(hits)
    finally:
        pack.close()


if __name__ == "__main__":
    main()
