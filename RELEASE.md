# Release Process (annpack)

This document describes how to cut and publish a release safely.

## Pre-flight checks

1) Run acceptance:
```bash
export PYTHON_BIN=/opt/homebrew/bin/python3.12
export ANNPACK_OFFLINE=1
bash tools/stage4_acceptance.sh
```

2) Run release guard:
```bash
bash tools/release_check.sh
```

3) Ensure CI passes on GitHub.

## Build artifacts

```bash
python -m pip install -U build twine
python -m build
python -m twine check dist/*
```

## Upload to TestPyPI

```bash
TWINE_USERNAME=__token__ \
TWINE_PASSWORD=YOUR_TEST_PYPI_TOKEN \
python -m twine upload --repository testpypi dist/*
```

## Upload to PyPI

```bash
TWINE_USERNAME=__token__ \
TWINE_PASSWORD=YOUR_PYPI_TOKEN \
python -m twine upload dist/*
```

Notes:
- Use per-project scoped tokens.
- Avoid exporting tokens into your shell profile; set them only for the upload command.
- Do not paste tokens into version control or logs.
