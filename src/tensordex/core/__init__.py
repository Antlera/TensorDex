"""
TensorDex Core Module.

- `TensorDex` engine (Rust MetadataStore + StorageBackend + FingerprintStore)
- Storage backends (local filesystem, S3)
- `TensorMetadata` / `ClusterInfo` dataclasses + BCS dimensionality defaults
- Shared timing utility
"""

from .engine import ModelNotReadyError, TensorDex
from .metadata import (
    BCS_DEFAULT_D,
    BCS_DEFAULT_W,
    ClusterInfo,
    TensorMetadata,
)
from .storage import LocalStorageBackend, S3StorageBackend, StorageBackend
from .utils import TensorDexTimer

__all__ = [
    # Engine
    "TensorDex",
    "ModelNotReadyError",
    # Storage
    "StorageBackend",
    "LocalStorageBackend",
    "S3StorageBackend",
    # Metadata
    "TensorMetadata",
    "ClusterInfo",
    "BCS_DEFAULT_D",
    "BCS_DEFAULT_W",
    # Utilities
    "TensorDexTimer",
]
