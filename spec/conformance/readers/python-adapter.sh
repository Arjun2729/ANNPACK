#!/usr/bin/env bash
# Conformance adapter for the Python second reader.
set -uo pipefail
READER="$(cd "$(dirname "$0")" && pwd)/annpack_reader.py"
verb=$1; shift
case "$verb" in
  tokenize)       exec python3 "$READER" tokenize "$1" ;;
  search)         exec python3 "$READER" search "$1" "$2" ;;
  open)           python3 "$READER" open "$1" >/dev/null 2>&1; exit $? ;;
  verify-receipt) python3 "$READER" verify-receipt "$1" >/dev/null 2>&1; exit $? ;;
  *) echo "unknown verb: $verb" >&2; exit 2 ;;
esac
