"""Compatibility shim. ANNPack was renamed to Adyar.

This package exists so that `pip install annpack` and `import annpack` do not
silently keep serving a build that stopped being maintained at the rename. It
forwards everything to :mod:`adyar` and warns once.

The wire format did not change with the rename. Artifacts and receipts produced
before it remain valid and verifiable; only the names moved.
"""

from __future__ import annotations

import warnings

from adyar import *  # noqa: F401,F403  re-exported for the shim's whole purpose
from adyar import AdyarError, Client  # noqa: F401  explicit for static analysis

#: Retained so `except annpack.ANNPackError` in existing code still catches.
ANNPackError = AdyarError

warnings.warn(
    "the 'annpack' package was renamed to 'adyar'; this shim forwards to it "
    "and is the final release under the old name. Install 'adyar' and import "
    "it directly.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["Client", "AdyarError", "ANNPackError"]
