"""
TensorDex: High-performance tensor database with Rust-accelerated operations.

A comprehensive tensor management system with features for:
- Fingerprint-based tensor identification and deduplication
- Local tensor storage with automatic compression
- Dynamic tensor loading from HuggingFace models
- Model-tensor mapping management

Example:
    >>> from tensordex import TensorDex
    >>> hub = TensorDex("./data/tensordex")              # create or reopen
    >>> hub = TensorDex.open("./data/tensordex")         # require it to exist
"""

from tensordex.compression import compress
from tensordex.core.engine import ModelNotReadyError, TensorDex
from tensordex.core.metadata import ClusterInfo, TensorMetadata

# Version information
__version__ = "0.1.0"
__author__ = "Tingfeng Lan"
__email__ = "tafflan2001@gmail.com"

# Public API
__all__ = [
    # Core
    "TensorDex",
    "ModelNotReadyError",
    "TensorMetadata",
    "ClusterInfo",
    # Compression
    "compress",
    # Version info
    "__version__",
    "__author__",
    "__email__",
]
