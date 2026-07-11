"""Tensor metadata dataclasses + BCS dimensionality defaults.

Fingerprint computation now happens inside the Rust ingest pipeline
(`tensordex._ops.ingest_from_safetensors_files`); the Rust
`compute_bcs_fingerprint_py` / `_u16_py` functions are still exported on
`tensordex._ops` for ad-hoc use but Python no longer wraps them here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

BCS_DEFAULT_D = 2
BCS_DEFAULT_W = 1024


@dataclass
class TensorMetadata:
    """Metadata for a tensor including shape, location info, and model context."""
    shape: Tuple[int, ...]
    unique_id: str
    param_name: str
    dtype: str
    storage_bytes: int = 0
    model_name: Optional[str] = None
    is_stored_locally: bool = False
    cluster_id: Optional[int] = None
    is_medoid: bool = False


@dataclass
class ClusterInfo:
    """Information about a tensor cluster."""
    cluster_id: int
    medoid_id: str
    tensor_ids: List[str]
    shape: Tuple[int, ...]
    total_distance: float
    size: int
