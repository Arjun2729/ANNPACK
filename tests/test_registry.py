import importlib
import logging
import sys
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
from fastapi import HTTPException


def _load_registry(tmp_path: Path, monkeypatch, dev_mode: str, secret: str | None):
    monkeypatch.setenv("REGISTRY_STORAGE", str(tmp_path))
    monkeypatch.setenv("REGISTRY_DEV_MODE", dev_mode)
    monkeypatch.setenv("REGISTRY_RATE_LIMIT_RPS", "0")
    monkeypatch.setenv("REGISTRY_RATE_LIMIT_BURST", "0")
    if secret is None:
        monkeypatch.delenv("REGISTRY_JWT_SECRET", raising=False)
    else:
        monkeypatch.setenv("REGISTRY_JWT_SECRET", secret)
    sys.modules.pop("registry.app", None)
    return importlib.import_module("registry.app")


def test_registry_requires_secret_in_prod(tmp_path, monkeypatch):
    monkeypatch.setenv("REGISTRY_STORAGE", str(tmp_path))
    monkeypatch.setenv("REGISTRY_DEV_MODE", "0")
    monkeypatch.delenv("REGISTRY_JWT_SECRET", raising=False)
    sys.modules.pop("registry.app", None)
    with pytest.raises(RuntimeError, match="REGISTRY_JWT_SECRET"):
        importlib.import_module("registry.app")


def test_registry_dev_mode_warns_without_secret(tmp_path, monkeypatch, caplog):
    caplog.set_level(logging.WARNING, logger="annpack.registry")
    module = _load_registry(tmp_path, monkeypatch, dev_mode="true", secret=None)
    assert module.DEV_MODE is True
    assert module.JWT_SECRET == "dev-secret"
    assert any("DEV_MODE enabled" in rec.message for rec in caplog.records)


def test_registry_rejects_path_traversal(tmp_path, monkeypatch):
    module = _load_registry(tmp_path, monkeypatch, dev_mode="true", secret="dev-secret")
    base = tmp_path / "base"
    base.mkdir()
    with pytest.raises(HTTPException):
        module._safe_path(base, "../secrets.txt")
