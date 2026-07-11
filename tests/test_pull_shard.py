"""Sharded pull (#7): bound peak memory by writing HF-style shards + index."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from tensordex.core.engine import TensorDex


def _ingest_model(hub: TensorDex, tmp: Path, tensors: dict) -> None:
    shard = tmp / "m.safetensors"
    save_file(tensors, str(shard))
    hub.init_model("org/m")
    hub.ingest([str(shard)], "org/m")
    hub.commit_model("org/m")


def test_pull_shards_and_reconstructs(tmp_path: Path) -> None:
    hub = TensorDex(str(tmp_path / "hub"), hydrate_all=False)
    # 6 tensors x 256 bytes each; a 600-byte cap → ~2 per shard.
    tensors = {f"layer.{i}.w": torch.full((64,), float(i)) for i in range(6)}
    _ingest_model(hub, tmp_path, tensors)

    out = tmp_path / "out"
    res = hub.pull("org/m", str(out), max_shard_size=600)
    assert res["shards"] > 1

    index = json.loads((out / "model.safetensors.index.json").read_text())
    assert set(index["weight_map"]) == set(tensors)
    assert index["metadata"]["total_size"] == sum(t.numel() * 4 for t in tensors.values())

    merged: dict = {}
    for shard_name in set(index["weight_map"].values()):
        merged.update(load_file(str(out / shard_name)))
    assert set(merged) == set(tensors)
    for name, tensor in tensors.items():
        assert torch.equal(merged[name], tensor)


def test_pull_single_file_default(tmp_path: Path) -> None:
    hub = TensorDex(str(tmp_path / "hub"), hydrate_all=False)
    tensors = {"a": torch.ones(4), "b": torch.zeros(8)}
    _ingest_model(hub, tmp_path, tensors)

    out = tmp_path / "out"
    res = hub.pull("org/m", str(out))  # no max_shard_size → single file
    assert res["shards"] == 1
    assert Path(res["output_path"]).name == "model.safetensors"
    got = load_file(res["output_path"])
    assert set(got) == set(tensors)
