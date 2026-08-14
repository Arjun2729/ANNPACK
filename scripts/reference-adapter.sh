#!/usr/bin/env bash
# Conformance adapter for the reference implementation.
#
# Demonstrates the four-verb contract in spec/conformance/README.md. An
# independent implementation supplies its own script of this shape; the runner
# needs nothing else.
set -uo pipefail
ADYAR=${ADYAR:-${ANNPACK:-$(cd "$(dirname "$0")/.." && pwd)/target/release/adyar}}
verb=$1; shift
case "$verb" in
  # `--` guards inputs that begin with a dash; the tokenizer vectors include one.
  tokenize)       exec "$ADYAR" tokenize -- "$1" ;;
  search)         exec "$ADYAR" search "$1" "$2" --limit 10 --mode lexical --json ;;
  open)           "$ADYAR" verify "$1" >/dev/null 2>&1; exit $? ;;
  verify-receipt) "$ADYAR" verify-evidence "$1" >/dev/null 2>&1; exit $? ;;
  *) echo "unknown verb: $verb" >&2; exit 2 ;;
esac
