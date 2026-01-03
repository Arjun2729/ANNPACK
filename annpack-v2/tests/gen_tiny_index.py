#!/usr/bin/env python3
import struct
import json
import sys
from pathlib import Path


def write_good(base: Path):
    dim = 4
    n_lists = 2
    metric = 1
    header_size = 72
    magic = 0x504E4E41
    version = 1
    endian = 1

    # Deterministic tiny dataset: two lists.
    # list 0: ids 100,101 aligned with e1
    # list 1: id 200 aligned with e2
    lists = {
        0: [
            (100, [1.0, 0.0, 0.0, 0.0]),
            (101, [0.9, 0.1, 0.0, 0.0]),
        ],
        1: [
            (200, [0.0, 1.0, 0.0, 0.0]),
        ],
    }
    n_vectors = sum(len(v) for v in lists.values())
    centroids = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
    ]

    fn = base.with_suffix(".annpack")
    with open(fn, "wb") as f:
        f.write(
            struct.pack(
                "<QIIIIIIIQ",
                magic,
                version,
                endian,
                header_size,
                dim,
                metric,
                n_lists,
                n_vectors,
                0,
            )
        )
        if f.tell() < header_size:
            f.write(b"\x00" * (header_size - f.tell()))

        # centroids
        for c in centroids:
            f.write(struct.pack("<" + "f" * dim, *c))

        list_offsets = []
        list_lengths = []
        for lid in range(n_lists):
            items = lists.get(lid, [])
            start = f.tell()
            f.write(struct.pack("<I", len(items)))
            if items:
                ids = [i for i, _ in items]
                vecs = [v for _, v in items]
                f.write(struct.pack("<" + "Q" * len(ids), *ids))
                for v in vecs:
                    f.write(struct.pack("<" + "e" * dim, *v))  # float16
            end = f.tell()
            list_offsets.append(start)
            list_lengths.append(end - start)

        table_pos = f.tell()
        for off, ln in zip(list_offsets, list_lengths):
            f.write(struct.pack("<QQ", off, ln))

        f.seek(36)
        f.write(struct.pack("<Q", table_pos))

    meta_fn = base.with_suffix(".meta.jsonl")
    with open(meta_fn, "w", encoding="utf-8") as w:
        for lid in range(n_lists):
            for idx, (vid, vec) in enumerate(lists[lid]):
                obj = {
                    "id": int(vid),
                    "title": f"vec{vid}",
                    "url": f"https://example.com/{vid}",
                    "snippet": f"list {lid} pos {idx} vec={vec}",
                }
                w.write(json.dumps(obj) + "\n")
    return fn, meta_fn


def write_bad_variants(base_good: Path):
    good = base_good.with_suffix(".annpack")
    data = good.read_bytes()
    bad_magic = base_good.with_name(base_good.name + "_badmagic.annpack")
    with open(bad_magic, "wb") as f:
        f.write(b"\x00\x00\x00\x00" + data[4:])

    truncated = base_good.with_name(base_good.name + "_trunc.annpack")
    with open(truncated, "wb") as f:
        f.write(data[:32])

    bad_table = base_good.with_name(base_good.name + "_badtable.annpack")
    with open(bad_table, "wb") as f:
        f.write(data)
        f.seek(36)
        f.write(struct.pack("<Q", 0))  # invalid offset_table_pos


def main():
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("tests/tmp")
    out_dir.mkdir(parents=True, exist_ok=True)
    base = out_dir / "tiny"
    fn, meta = write_good(base)
    write_bad_variants(base)
    print(fn)
    print(meta)


if __name__ == "__main__":
    main()
