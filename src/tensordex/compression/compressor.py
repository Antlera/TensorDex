"""Thin dispatcher to the Rust compression executor.

Python's job here is only:
  1. Turn a `TensorDex` into the tensor-dir URI the Rust side expects
     (local path or s3://bucket/prefix?region=...).
  2. Serialize the caller's plan list and forward it to Rust.

All per-pair work — reading tensors, running the codec, accumulating
metrics, and (eventually) writing per-pair rows into ``results.db`` —
happens inside the Rust ``_ops.execute_batch_plans_pairwise_py`` entry.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional, Sequence

from tensordex.core.engine import TensorDex

try:
    from tensordex._ops import compress as _compress_rust
except ImportError as exc:
    _compress_rust = None  # type: ignore[assignment]
    _IMPORT_ERROR: Optional[Exception] = exc
else:
    _IMPORT_ERROR = None


def compress(
    hub: TensorDex,
    plans: Sequence[Dict[str, Any]],
    *,
    output_dir: Optional[str] = None,
    algorithm: Optional[str] = None,
    level: int = 3,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Execute a batch of compression plans against ``hub``.

    Args:
        hub: Source of base/target tensors.
        plans: Plan dicts — each must carry at least ``param_name``,
            ``target_tensor_id``, ``base_tensor_id``, ``target_shape``,
            ``target_dtype``. Matches the Rust ``CompressionPlan`` schema.
        output_dir: Where Rust writes compressed artifacts, or ``None`` to
            skip artifact output (dry-run / metrics-only).
        algorithm: Codec name understood by Rust (``"bitx"`` or ``"tensorx"``).
            ``None`` defers to the Rust default.
        level: zstd compression level (1–22).
        verbose: Forward to Rust for its own progress logging.

    Returns:
        The Rust summary dict: ``total_plans``, ``executed_plans``,
        ``failed_plans``, ``total_original_bytes``, ``total_compressed_bytes``,
        ``compression_ratio``, ``execution_time_ms``, ``pairs``, ``per_pair``,
        ``per_pair_json``.
    """
    if _compress_rust is None:
        raise RuntimeError(
            "Rust compression extension 'tensordex._ops' is required. "
            "Build it via `make dev-install`."
        ) from _IMPORT_ERROR

    if not plans:
        return {
            "total_plans": 0,
            "executed_plans": 0,
            "failed_plans": 0,
            "total_original_bytes": 0,
            "total_compressed_bytes": 0,
            "compression_ratio": 0.0,
            "execution_time_ms": 0,
        }

    return _compress_rust(
        json.dumps(list(plans)),
        _resolve_tensor_dir(hub),
        output_dir=output_dir,
        algorithm=algorithm,
        level=level,
        verbose=verbose,
    )


def _resolve_tensor_dir(hub: TensorDex) -> str:
    """Derive the URI the Rust executor should use to fetch tensors from ``hub``."""
    backend_kind = getattr(hub, "backend_kind", "local")
    options = getattr(hub, "backend_options", {}) or {}

    if backend_kind == "s3":
        bucket = options.get("bucket")
        if not bucket:
            raise ValueError("S3 backend requires 'bucket' in backend_options")
        prefix = str(options.get("prefix") or "").strip("/")
        uri = f"s3://{bucket}"
        if prefix:
            uri = f"{uri}/{prefix}"
        region = options.get("region")
        if region:
            uri = f"{uri}?region={region}"
        return uri

    if "root_dir" in options:
        return str(options["root_dir"])
    return str(hub.storage_dir)
