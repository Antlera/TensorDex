"""Self-describing blob codec for TensorDex.

Blobs on disk are always ``.safetensors`` files with a single payload
key ``"tensor"``. The ``__metadata__`` dict carried by safetensors is
the fork point between two physical states:

- **Raw**: metadata may be empty or carry only free-form header info;
  ``"tensor"`` holds the actual tensor bytes with the tensor's native
  dtype/shape.
- **Compressed (delta)**: metadata carries ``"codec"``,
  ``"base_tensor_id"``, and enough shape/dtype info to reconstruct the
  target tensor; ``"tensor"`` is a 1-D ``uint8`` tensor of the codec's
  output bytes.

This module owns the layout so the engine just calls
``load_blob(path)`` / ``save_compressed(path, ...)`` without caring
about safetensors quirks.

**Content-addressing invariant.** A tensor id is the XXH3-128 of the
tensor's *logical* (decoded) bytes, assigned at ingest. Compression
rewrites the blob in place, so a compressed blob's *physical* bytes are
codec output and no longer hash to its id — content-addressing holds for
the logical tensor, not the on-disk bytes. Read-time verification
(``get_tensor(verify=True)``) therefore re-hashes the *reconstructed*
tensor, not the raw blob.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
from safetensors import safe_open
from safetensors.torch import save_file

_PAYLOAD_KEY = "tensor"
_META_META_KEY = "tensor"  # safetensors stores per-tensor metadata under its key

CODEC_TENSORX = "tensorx"
SUPPORTED_CODECS = {CODEC_TENSORX}


@dataclass
class CompressedBlob:
    """In-memory view of a compressed blob's header + payload."""

    codec: str
    base_tensor_id: str
    item_size: int
    level: int
    target_shape: Tuple[int, ...]
    target_dtype: str
    compressed_bytes: bytes

    @property
    def is_compressed(self) -> bool:
        return True


@dataclass
class RawBlob:
    """In-memory view of an uncompressed blob."""

    tensor: torch.Tensor

    @property
    def is_compressed(self) -> bool:
        return False


def _parse_metadata(meta: Optional[Dict[str, str]]) -> Optional[Dict[str, Any]]:
    """Return the decoded engine metadata dict, or ``None`` if this is raw.

    Historically Rust ingest stores ``{_META_META_KEY: <json string>}``;
    the JSON *may* include our ``codec`` fields (for compressed blobs) or
    unrelated ingest info (for raw ones). This normalises either shape.
    """
    if not meta:
        return None
    raw = meta.get(_META_META_KEY)
    if raw is None:
        return None
    try:
        decoded = json.loads(raw)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    return decoded if isinstance(decoded, dict) else None


def probe_codec(path: Path) -> Optional[str]:
    """Cheap header-only peek — returns the codec name, or ``None`` for raw blobs."""
    with safe_open(str(path), framework="pt") as f:
        meta = _parse_metadata(f.metadata())
    if meta is None:
        return None
    codec = meta.get("codec")
    return codec if isinstance(codec, str) and codec in SUPPORTED_CODECS else None


def read_base_id(path: Path) -> Optional[str]:
    """Return the ``base_tensor_id`` referenced by a compressed blob, if any."""
    with safe_open(str(path), framework="pt") as f:
        meta = _parse_metadata(f.metadata())
    if meta is None:
        return None
    base = meta.get("base_tensor_id")
    return base if isinstance(base, str) and base else None


def load_blob(path: Path):
    """Load a blob and classify it as ``RawBlob`` or ``CompressedBlob``.

    Returns a union so the caller ``isinstance`` dispatches into the
    decode path only when necessary.
    """
    from safetensors.torch import load_file as st_load

    tensors = st_load(str(path))
    # Re-open to grab metadata (safetensors.torch.load_file discards it).
    with safe_open(str(path), framework="pt") as f:
        meta = _parse_metadata(f.metadata())

    payload = tensors.get(_PAYLOAD_KEY)
    if payload is None:
        # Legacy / odd layouts — fall back to the first entry.
        payload = next(iter(tensors.values()))

    if meta and isinstance(meta.get("codec"), str) and meta["codec"] in SUPPORTED_CODECS:
        return CompressedBlob(
            codec=str(meta["codec"]),
            base_tensor_id=str(meta["base_tensor_id"]),
            item_size=int(meta.get("item_size", 2)),
            level=int(meta.get("level", 3)),
            target_shape=tuple(int(x) for x in meta["target_shape"]),
            target_dtype=str(meta["target_dtype"]),
            compressed_bytes=bytes(payload.numpy().tobytes()),
        )

    return RawBlob(tensor=payload)


def save_compressed(
    path: Path,
    *,
    codec: str,
    base_tensor_id: str,
    item_size: int,
    level: int,
    target_shape: Tuple[int, ...],
    target_dtype: str,
    compressed_bytes: bytes,
) -> int:
    """Atomically write a compressed blob. Returns the final on-disk size in bytes."""
    if codec not in SUPPORTED_CODECS:
        raise ValueError(f"Unsupported codec {codec!r}")

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        _PAYLOAD_KEY: torch.frombuffer(
            bytearray(compressed_bytes), dtype=torch.uint8
        ).clone()
    }
    header = {
        "codec": codec,
        "base_tensor_id": base_tensor_id,
        "item_size": int(item_size),
        "level": int(level),
        "target_shape": [int(x) for x in target_shape],
        "target_dtype": str(target_dtype),
    }
    metadata = {_META_META_KEY: json.dumps(header, separators=(",", ":"))}

    tmp_fd, tmp_path = tempfile.mkstemp(
        dir=str(path.parent), prefix=f"{path.name}.", suffix=".tmp"
    )
    os.close(tmp_fd)
    tmp_path = Path(tmp_path)
    try:
        save_file(payload, str(tmp_path), metadata=metadata)
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)

    return path.stat().st_size


__all__ = [
    "CODEC_TENSORX",
    "SUPPORTED_CODECS",
    "CompressedBlob",
    "RawBlob",
    "probe_codec",
    "read_base_id",
    "load_blob",
    "save_compressed",
]
