#!/usr/bin/env bash
set -euo pipefail

log() {
  echo "[stage_all] $*"
}

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export ANNPACK_OFFLINE=1
export ANNPACK_FAISS_THREADS=1
if [ -z "${PYTHON_BIN:-}" ] && [ -n "${pythonLocation:-}" ] && [ -x "${pythonLocation}/bin/python" ]; then
  PYTHON_BIN="${pythonLocation}/bin/python"
fi
PYTHON_BIN="${PYTHON_BIN:-$(command -v python || command -v python3)}"

if ! command -v annpack >/dev/null 2>&1; then
  echo "[stage_all] annpack not found on PATH; install with: pip install -e .[dev]"
  exit 1
fi

if ! "$PYTHON_BIN" -c "import build" >/dev/null 2>&1; then
  echo "[stage_all] missing build module; install with: $PYTHON_BIN -m pip install build"
  exit 1
fi

if ! "$PYTHON_BIN" -c "import twine" >/dev/null 2>&1; then
  echo "[stage_all] missing twine module; install with: $PYTHON_BIN -m pip install twine"
  exit 1
fi

if ! "$PYTHON_BIN" -c "import mkdocs" >/dev/null 2>&1; then
  echo "[stage_all] missing mkdocs module; install with: $PYTHON_BIN -m pip install mkdocs"
  exit 1
fi

if ! command -v npm >/dev/null 2>&1; then
  echo "[stage_all] npm not found; install Node.js"
  exit 1
fi

log "repo hygiene"
bash tools/repo_hygiene_check.sh
bash tools/repo_hygiene.sh

log "ruff check"
ruff check .

log "ruff format"
ruff format --check .

log "mypy"
mypy

log "pytest"
pytest -q

log "stage4 acceptance"
bash tools/stage4_acceptance.sh

log "determinism check"
bash tools/determinism_check.sh

if [[ "${ANNPACK_SKIP_NET_TESTS:-}" == "1" ]]; then
  echo "[stage_all] skipping ci_smoke (ANNPACK_SKIP_NET_TESTS=1)"
else
  log "ci smoke"
  bash tools/ci_smoke.sh
fi

log "web install/test/build"
pushd web >/dev/null
if [ ! -f package-lock.json ]; then
  echo "[stage_all] missing web/package-lock.json; run: cd web && npm install"
  exit 1
fi
npm_tmp="$(mktemp -d)"
old_home="${HOME:-}"
export HOME="$npm_tmp/home"
export NPM_CONFIG_CACHE="$npm_tmp/npm-cache"
export NPM_CONFIG_TMP="$npm_tmp/npm-tmp"
export NPM_CONFIG_LOGS_DIR="$npm_tmp/logs"
export NPM_CONFIG_USERCONFIG="$npm_tmp/npmrc"
export NPM_CONFIG_UPDATE_NOTIFIER=false
export NPM_CONFIG_FUND=false
export NPM_CONFIG_AUDIT=false
export NPM_CONFIG_LOGLEVEL=error
export TMPDIR="$npm_tmp/tmp"
mkdir -p "$HOME" "$NPM_CONFIG_CACHE" "$NPM_CONFIG_TMP" "$NPM_CONFIG_LOGS_DIR" "$TMPDIR"
cat >"$NPM_CONFIG_USERCONFIG" <<EOF
cache=$NPM_CONFIG_CACHE
logs-dir=$npm_tmp/logs
audit=false
fund=false
EOF
if ! npm ci --workspaces --include-workspace-root; then
  echo "[stage_all] npm ci failed; retrying with npm install --no-package-lock"
  rm -rf node_modules
  npm install --no-package-lock --workspaces --include-workspace-root
fi
rollup_native_ok=0
if [ -d node_modules/@rollup ]; then
  if find node_modules/@rollup -maxdepth 1 -type d -name "rollup-*" -print -quit | grep -q .; then
    rollup_native_ok=1
  fi
fi
if [ "$rollup_native_ok" -eq 0 ]; then
  echo "[stage_all] rollup native package missing; reinstalling with npm install --no-package-lock"
  rm -rf node_modules
  npm install --no-package-lock --workspaces --include-workspace-root
fi
npm test
npm run lint --workspaces --if-present
npm run format:check --workspaces --if-present
npm run typecheck --workspaces --if-present
npm run build
popd >/dev/null
if [ -n "${old_home:-}" ]; then
  export HOME="$old_home"
else
  unset HOME
fi
rm -rf "$npm_tmp"

log "mkdocs build"
mkdocs build

log "build + twine check"
rm -rf dist build
"$PYTHON_BIN" -m build
"$PYTHON_BIN" -m twine check dist/*

echo "PASS stage all"
