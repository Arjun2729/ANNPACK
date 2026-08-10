#!/usr/bin/env python3
"""Rebuild tracked demo packs twice and require byte-identical committed output."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = (
    Path("docs/docs-v1.annpack"),
    Path("docs/docs-v2.annpack"),
    Path("docs/packs/google-okf-ga4.annpack"),
    Path("docs/packs/google-okf-ga4.pub"),
)


def run(*args: str) -> None:
    subprocess.run(args, cwd=ROOT, check=True)


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with (ROOT / path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def snapshot() -> dict[str, str]:
    missing = [str(path) for path in ARTIFACTS if not (ROOT / path).is_file()]
    if missing:
        raise SystemExit(f"missing tracked demo artifacts: {', '.join(missing)}")
    return {str(path): digest(path) for path in ARTIFACTS}


def main() -> int:
    run("./scripts/build-demo-packs.sh")
    first = snapshot()
    run("./scripts/build-demo-packs.sh")
    second = snapshot()

    if first != second:
        print("demo rebuild is not byte-deterministic:")
        for path in sorted(first):
            if first[path] != second[path]:
                print(f"- {path}: {first[path]} != {second[path]}")
        return 1

    subprocess.run(
        ["git", "diff", "--exit-code", "--", *(str(path) for path in ARTIFACTS)],
        cwd=ROOT,
        check=True,
    )
    print("demo reproducibility passed:")
    for path, value in sorted(second.items()):
        print(f"- {path}: sha256:{value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
