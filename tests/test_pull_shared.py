"""Pull a model whose tied tensors dedup to one id (e.g. rotary buffers)."""

from __future__ import annotations

from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from tensordex import TensorDex


def _make_hub(tmp_path: Path) -> tuple[TensorDex, Path]:
    # a.inv_freq and b.inv_freq are byte-identical → one content id → on pull
    # they decode to the SAME cached object (aliased storage).
    w = torch.arange(8, dtype=torch.float32)
    sd = {
        "a.inv_freq": w,
        "b.inv_freq": w.clone(),
        "big.weight": torch.randn(16, 16),
    }
    src = tmp_path / "m.safetensors"
    save_file(sd, str(src))
    hub = TensorDex(str(tmp_path / "hub"))
    hub.init_model("m")
    hub.ingest([str(src)], "m")
    hub.commit_model("m")
    return hub, w


def test_pull_handles_aliased_tensors(tmp_path: Path) -> None:
    hub, w = _make_hub(tmp_path)
    out = tmp_path / "out"
    hub.pull("m", str(out), verify=True)  # must not raise on shared storage

    got = load_file(str(out / "model.safetensors"))
    assert torch.equal(got["a.inv_freq"], w)
    assert torch.equal(got["b.inv_freq"], w)
    assert tuple(got["big.weight"].shape) == (16, 16)


def test_pull_sharded_handles_aliased_tensors(tmp_path: Path) -> None:
    hub, w = _make_hub(tmp_path)
    out = tmp_path / "out_sharded"
    hub.pull("m", str(out), max_shard_size=64)  # force shards; must not raise
    merged: dict[str, torch.Tensor] = {}
    for p in out.glob("*.safetensors"):
        merged.update(load_file(str(p)))
    assert torch.equal(merged["a.inv_freq"], w)
    assert torch.equal(merged["b.inv_freq"], w)
