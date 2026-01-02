from importlib import metadata

from .build import build_index, build_index_from_df, build_index_from_hf_wikipedia
from .reader import ANNPackIndex, ANNPackHeader
from .api import build_pack, open_pack
from .packset import build_packset_base, build_delta, update_packset_manifest, open_packset

try:
    __version__ = metadata.version("annpack")
except metadata.PackageNotFoundError:
    __version__ = "0.1.1"

__all__ = [
    "build_index",
    "build_index_from_df",
    "build_index_from_hf_wikipedia",
    "ANNPackIndex",
    "ANNPackHeader",
    "build_pack",
    "open_pack",
    "build_packset_base",
    "build_delta",
    "update_packset_manifest",
    "open_packset",
    "__version__",
]
