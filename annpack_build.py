import os
import sys

# Ensure local package path is on sys.path when running from repo root
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
PKG_DIR = os.path.join(BASE_DIR, "python")
if PKG_DIR not in sys.path:
    sys.path.insert(0, PKG_DIR)


def _run() -> None:
    from annpack.cli import main

    main()


if __name__ == "__main__":
    _run()
