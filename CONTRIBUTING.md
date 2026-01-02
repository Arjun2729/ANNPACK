# Contributing

## Dev setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .[dev]
```

## Tests

```bash
pytest -q
```

## Smoke

```bash
tools/ci_smoke.sh
tools/determinism_check.sh
```

## Release

See `RELEASE.md` and `tools/release.sh`.
