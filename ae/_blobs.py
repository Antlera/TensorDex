"""Shared helpers for AE verification — content-addressed blob I/O, the
TensorDex content hash, and the TensorX delta ratio.

Everything here dogfoods the *installed* `tensordex` package (its Rust
`_ops.content_hash` and `_ops.compress_tensorx_rust`), so a passing check
proves the reviewer's freshly built extension reproduces the published cache
— not that some bundled script agrees with itself.
"""
from __future__ import annotations

import os
from typing import Iterable, Optional, Tuple

import torch
from safetensors import safe_open

from tensordex import _ops

# The TensorX delta codec that filled results.db used zstd level 1 and a
# 2-byte item size (bf16). These are fixed constants of the published trace.
TENSORX_LEVEL = 1
BF16_ITEM_SIZE = 2


def real_tensor_key(keys: Iterable[str]) -> Optional[str]:
    """A blob holds three keys — `_fingerprint`, the tensor, and
    `<name>_fingerprint`. Return the real tensor key."""
    for k in keys:
        if k == "_fingerprint" or k.endswith("_fingerprint"):
            continue
        return k
    return None


def blob_path(root: str, tid: str) -> str:
    """Content-addressed path: `<root>/<xx>/<yy>/<tid>.safetensors`."""
    return os.path.join(root, tid[:2], tid[2:4], f"{tid}.safetensors")


def load_tensor_bytes(path: str) -> Tuple[bytes, torch.dtype, Tuple[int, ...]]:
    """Raw little-endian bytes of the real tensor stored in `path`."""
    with safe_open(path, framework="pt") as f:
        key = real_tensor_key(list(f.keys()))
        if key is None:
            raise ValueError(f"no tensor key in {path}")
        t = f.get_tensor(key)
    raw = t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return raw, t.dtype, tuple(t.shape)


def content_id(raw: bytes) -> str:
    """TensorDex content id (XXH3-128 hex) — via the package's Rust kernel."""
    return _ops.content_hash(raw)


def tensorx_ratio(target_raw: bytes, base_raw: bytes,
                  item_size: int = BF16_ITEM_SIZE,
                  level: int = TENSORX_LEVEL) -> Tuple[int, float]:
    """Compress `target` as a TensorX delta against `base`; return
    (compressed_bytes, ratio) where ratio = compressed / original."""
    comp = _ops.compress_tensorx_rust(target_raw, base_raw, item_size, level)
    return len(comp), len(comp) / len(target_raw)


# FM++ is optional — present only when the extension was built with
# `--features fmpp` (links the vendored FM-Delta lib). See ae/README.md.
HAS_FMPP = hasattr(_ops, "compress_fmpp_rust")


def fmpp_ratio(target_raw: bytes, base_raw: bytes,
               item_size: int = BF16_ITEM_SIZE) -> Tuple[int, float]:
    """Compress `target` as an FM++ delta against `base`; return
    (compressed_bytes, ratio). Requires the `fmpp` feature build."""
    comp = _ops.compress_fmpp_rust(target_raw, base_raw, item_size)
    return len(comp), len(comp) / len(target_raw)


def available_ids(blob_root: str) -> set:
    """Set of tensor ids present under a content-addressed blob root."""
    ids = set()
    if not os.path.isdir(blob_root):
        return ids
    for d1 in os.listdir(blob_root):
        p1 = os.path.join(blob_root, d1)
        if not os.path.isdir(p1):
            continue
        for d2 in os.listdir(p1):
            p2 = os.path.join(p1, d2)
            if not os.path.isdir(p2):
                continue
            for fn in os.listdir(p2):
                if fn.endswith(".safetensors"):
                    ids.add(fn[: -len(".safetensors")])
    return ids
