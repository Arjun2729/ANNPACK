from __future__ import annotations

import dataclasses
import mmap
import os
import struct
from types import TracebackType
from typing import List, Literal, Optional, Tuple

import numpy as np

from .logutil import timed


@dataclasses.dataclass
class ANNPackHeader:
    """Parsed header fields for an ANNPack file."""

    magic: int
    version: int
    endian: int
    header_size: int
    dim: int
    metric: int
    n_lists: int
    n_vectors: int
    offset_table_pos: int


class ANNPackIndex:
    """Memory-mapped reader for ANNPack IVF indexes."""

    def __init__(self, path: str, probe: int = 8):
        self.path = path
        self.probe = probe
        self._fd: Optional[int] = None
        self._mm: Optional[mmap.mmap] = None
        self.header: Optional[ANNPackHeader] = None
        self._centroids: Optional[np.ndarray] = None
        self._list_offsets: Optional[np.ndarray] = None
        self._list_lengths: Optional[np.ndarray] = None

    @classmethod
    def open(cls, path: str, probe: int = 8) -> "ANNPackIndex":
        """Open an ANNPack file and return a ready-to-search index."""
        obj = cls(path, probe=probe)
        obj._open()
        return obj

    def _open(self) -> None:
        fd = os.open(self.path, os.O_RDONLY)
        self._fd = fd
        size = os.fstat(fd).st_size
        mm = mmap.mmap(fd, size, access=mmap.ACCESS_READ)
        self._mm = mm

        header_bytes = mm[:72]
        fields = struct.unpack("<QIIIIIIIQ", header_bytes[:44])
        magic, version, endian, header_size, dim, metric, n_lists, n_vectors, offset_table_pos = (
            fields
        )
        if magic != 0x504E4E41:
            raise ValueError(f"Bad magic: {hex(magic)}")
        if version != 1 or endian != 1 or header_size != 72:
            raise ValueError("Unsupported ANNPack header parameters")
        if dim <= 0 or n_lists <= 0 or n_vectors < 0:
            raise ValueError("Invalid ANNPack header values")
        self.header = ANNPackHeader(
            magic=magic,
            version=version,
            endian=endian,
            header_size=header_size,
            dim=dim,
            metric=metric,
            n_lists=n_lists,
            n_vectors=n_vectors,
            offset_table_pos=offset_table_pos,
        )

        # Centroids
        cent_off = header_size
        self._centroids = np.frombuffer(
            mm, dtype="<f4", count=n_lists * dim, offset=cent_off
        ).reshape(n_lists, dim)

        # Offset table
        table = np.frombuffer(mm, dtype="<u8", count=n_lists * 2, offset=offset_table_pos).reshape(
            n_lists, 2
        )
        self._list_offsets = table[:, 0].astype(np.int64)
        self._list_lengths = table[:, 1].astype(np.int64)

    def search(self, query: np.ndarray, k: int = 10) -> List[Tuple[int, float]]:
        """Search the IVF index and return (id, score) pairs."""
        with timed("search", {"k": k}):
            return self._search_inner(query, k)

    def _search_inner(self, query: np.ndarray, k: int) -> List[Tuple[int, float]]:
        """Internal search implementation."""
        if self._mm is None:
            raise RuntimeError("Index not opened")
        if self.header is None:
            raise RuntimeError("Index header missing")
        q = np.asarray(query, dtype=np.float32)
        if q.ndim != 1 or q.shape[0] != self.header.dim:
            raise ValueError(f"Query must be 1-D of length {self.header.dim}")
        norm = np.linalg.norm(q)
        if norm == 0:
            raise ValueError("Zero query vector")
        q = q / norm

        centroids = self._centroids
        list_offsets = self._list_offsets
        list_lengths = self._list_lengths
        if centroids is None or list_offsets is None or list_lengths is None:
            raise RuntimeError("Index data not loaded")
        scores = centroids @ q
        probe = min(self.probe, self.header.n_lists)
        probe_idx = np.argpartition(scores, -probe)[-probe:]
        probe_idx = probe_idx[np.argsort(scores[probe_idx])[::-1]]

        top_scores = np.full(k, -1e9, dtype=np.float32)
        top_ids = np.zeros(k, dtype=np.int64)
        top_count = 0

        dim = self.header.dim
        for list_id in probe_idx:
            off = int(list_offsets[list_id])
            length = int(list_lengths[list_id])
            if length <= 0:
                continue
            blob = memoryview(self._mm)[off : off + length]
            if len(blob) < 4:
                continue
            count = struct.unpack_from("<I", blob, 0)[0]
            needed = 4 + count * 8 + count * dim * 2
            if count == 0 or needed > len(blob):
                continue

            ids = np.frombuffer(blob, dtype="<u8", count=count, offset=4)
            vec_offset = 4 + count * 8
            vecs = np.frombuffer(blob, dtype="<f2", count=count * dim, offset=vec_offset)
            vecs = vecs.astype(np.float32, copy=False).reshape(count, dim)

            local_scores = vecs @ q
            for cand_id, cand_score in zip(ids, local_scores):
                if top_count < k or cand_score > top_scores[top_count - 1]:
                    pos = top_count - 1 if top_count > 0 else 0
                    while pos >= 0 and cand_score > top_scores[pos]:
                        pos -= 1
                    pos += 1
                    if top_count < k:
                        top_count += 1
                    if top_count > k:
                        top_count = k
                    for m in range(top_count - 1, pos, -1):
                        top_scores[m] = top_scores[m - 1]
                        top_ids[m] = top_ids[m - 1]
                    top_scores[pos] = cand_score
                    top_ids[pos] = cand_id

        return [(int(i), float(s)) for i, s in zip(top_ids[:top_count], top_scores[:top_count])]

    def close(self) -> None:
        """Close the underlying mmap/file safely."""
        for attr in ("_centroids", "_list_offsets", "_list_lengths"):
            if hasattr(self, attr):
                setattr(self, attr, None)

        if self._mm is not None:
            self._mm.close()
            self._mm = None
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None

    def __enter__(self) -> "ANNPackIndex":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> Literal[False]:
        self.close()
        return False
