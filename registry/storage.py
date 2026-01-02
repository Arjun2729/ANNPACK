from __future__ import annotations

from pathlib import Path
from typing import Iterable


class LocalStorage:
    """Local filesystem storage backend."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def ensure_dir(self, path: Path) -> Path:
        path.mkdir(parents=True, exist_ok=True)
        return path

    def list_dirs(self, path: Path) -> Iterable[Path]:
        return [p for p in path.iterdir() if p.is_dir()]


class S3Storage:
    """Stub for future S3-compatible storage backend."""

    def __init__(self, bucket: str, prefix: str = "") -> None:
        self.bucket = bucket
        self.prefix = prefix

    def ensure_dir(self, path: Path) -> Path:
        raise NotImplementedError("S3 storage not implemented")

    def list_dirs(self, path: Path) -> Iterable[Path]:
        raise NotImplementedError("S3 storage not implemented")
