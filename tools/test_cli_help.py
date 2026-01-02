#!/usr/bin/env python
"""
Minimal help checks for annpack CLI.

Run with: python tools/test_cli_help.py
"""

import subprocess
import sys
import shutil


def check(cmd):
    print(f"[test] {' '.join(cmd)}")
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        out = result.stderr.decode("utf-8", errors="ignore")
        if out:
            sys.stderr.write(out)
        raise SystemExit(f"Command failed: {' '.join(cmd)}")


def main():
    check([sys.executable, "-m", "annpack.cli", "--help"])
    bin_path = shutil.which("annpack")
    if not bin_path:
        raise SystemExit("Console script 'annpack' not found (install package first).")
    check([bin_path, "--help"])
    print("OK help commands")


if __name__ == "__main__":
    main()
