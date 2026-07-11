"""Read-time integrity: a tensor's bytes must hash to its content id.

Covers the #5 fix — get_tensor(verify=True) recomputes the XXH3-128 of
the reconstructed tensor and rejects a blob whose bytes don't match.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from tensordex.core.engine import IntegrityError, TensorDex


def _ingest(hub: TensorDex, tmp: Path, tensor: torch.Tensor) -> str:
    shard = tmp / "s.safetensors"
    save_file({"w": tensor}, str(shard))
    hub.init_model("org/m")
    mapping = hub.ingest([str(shard)], "org/m")
    hub.commit_model("org/m")
    return mapping["w"]


def test_verify_passes_for_intact_blob(tmp_path: Path) -> None:
    hub = TensorDex(str(tmp_path / "hub"), hydrate_all=False)
    t = torch.arange(64, dtype=torch.float32).reshape(8, 8)
    tid = _ingest(hub, tmp_path, t)
    got = hub.get_tensor(tensor_id=tid, verify=True)
    assert torch.equal(got, t)


def test_verify_catches_corruption(tmp_path: Path) -> None:
    hub = TensorDex(str(tmp_path / "hub"), hydrate_all=False)
    t = torch.arange(64, dtype=torch.float32).reshape(8, 8)
    tid = _ingest(hub, tmp_path, t)

    # Flip a byte in the tensor payload (last byte of the safetensors file).
    blob = hub.storage_backend.blob_path_for_id(tid)
    data = bytearray(blob.read_bytes())
    data[-1] ^= 0xFF
    blob.write_bytes(bytes(data))

    with pytest.raises(IntegrityError):
        hub.get_tensor(tensor_id=tid, verify=True)

    # Without verify, the corrupt bytes load silently — no integrity check.
    hub.get_tensor(tensor_id=tid, verify=False)
