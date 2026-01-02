"""Logging helpers for ANNPack."""

from __future__ import annotations

import json
import logging
import os
import time
from contextlib import contextmanager
from typing import Dict, Iterator


def get_logger(name: str = "annpack") -> logging.Logger:
    """Return a configured logger."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    level_name = os.environ.get("ANNPACK_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logger.setLevel(level)

    handler = logging.StreamHandler()
    if os.environ.get("ANNPACK_LOG_JSON") == "1":
        handler.setFormatter(logging.Formatter("%(message)s"))
    else:
        handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def log_event(name: str, detail: Dict[str, object]) -> None:
    """Emit a structured log event if logging is enabled."""
    logger = get_logger()
    payload = {"event": name, **detail}
    if os.environ.get("ANNPACK_LOG_JSON") == "1":
        logger.info(json.dumps(payload))
    else:
        logger.info(f"{name} {detail}")


@contextmanager
def timed(event: str, detail: Dict[str, object] | None = None) -> Iterator[None]:
    """Context manager to log duration for an event."""
    start = time.perf_counter()
    yield
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    payload = detail.copy() if detail else {}
    payload["ms"] = round(elapsed_ms, 3)
    log_event(event, payload)
