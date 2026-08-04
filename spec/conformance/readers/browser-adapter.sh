#!/usr/bin/env bash
# Conformance adapter for the browser runtime.
set -uo pipefail
READER="$(cd "$(dirname "$0")" && pwd)/browser-reader.mjs"
verb=$1; shift
case "$verb" in
  tokenize)       exec node "$READER" tokenize "$1" ;;
  search)         exec node "$READER" search "$1" "$2" ;;
  open)           node "$READER" open "$1" >/dev/null 2>&1; exit $? ;;
  verify-receipt) node "$READER" verify-receipt "$1" >/dev/null 2>&1; exit $? ;;
  *) echo "unknown verb: $verb" >&2; exit 2 ;;
esac
