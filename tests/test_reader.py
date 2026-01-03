from pathlib import Path

import pytest

from annpack.reader import ANNPackIndex


def test_reader_rejects_bad_magic(tmp_path: Path) -> None:
    path = tmp_path / "bad.annpack"
    path.write_bytes(b"BADC" * 18)
    with pytest.raises(ValueError):
        ANNPackIndex.open(str(path))


def test_reader_rejects_bad_header_size(tmp_path: Path) -> None:
    path = tmp_path / "bad_header.annpack"
    # magic ANNP, version=1, endian=1, header_size=1 (invalid), rest zeros
    header = (
        b"ANNP"
        + (1).to_bytes(4, "little")
        + (1).to_bytes(4, "little")
        + (1).to_bytes(4, "little")
        + b"\x00" * (72 - 16)
    )
    path.write_bytes(header)
    with pytest.raises(ValueError):
        ANNPackIndex.open(str(path))
