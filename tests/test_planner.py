"""Attach planner + auto-compress over the SQL-backed metadata (#8).

plan_attach no longer reads a Python shadow cache; it pulls shape/dtype
straight from SQLite (the model's tids in batch, or the whole table for
include_existing_bases). These tests exercise that path end to end,
including a cross-model attach and exact reconstruction afterwards.
"""

from __future__ import annotations

from pathlib import Path

import torch
from safetensors.torch import save_file

from tensordex.core.engine import TensorDex


def _ingest(hub: TensorDex, model: str, tensors: dict, tmp: Path) -> dict:
    shard = tmp / f"{model.replace('/', '_')}.safetensors"
    save_file(tensors, str(shard))
    hub.init_model(model)
    mapping = hub.ingest([str(shard)], model)
    hub.commit_model(model)
    return mapping


def test_auto_compress_cross_model_roundtrip(tmp_path: Path) -> None:
    hub = TensorDex(str(tmp_path / "hub"), hydrate_all=False)
    torch.manual_seed(0)
    base = torch.randn(128, 128, dtype=torch.float32)
    target = base + 1e-3  # near-identical → strong attach candidate, same shape

    _ingest(hub, "org/base", {"w": base}, tmp_path)
    target_id = _ingest(hub, "org/target", {"w": target}, tmp_path)["w"]

    # include_existing_bases iterates the whole tensors table from SQL.
    res = hub.auto_compress(
        "org/target", include_existing_bases=True, cr_threshold=0.95
    )
    assert res["executed"]
    ok = [r for r in res["results"] if r.get("status") == "ok"]
    assert ok, "expected a cross-model attach against org/base"
    assert any(r["base_tensor_id"] != target_id for r in ok)

    # The now delta-encoded target still reconstructs byte-for-byte.
    assert torch.equal(hub.get_tensor(tensor_id=target_id), target)


def test_compress_bundle_is_star_not_chain(tmp_path: Path) -> None:
    """A checkpoint bundle compresses as a star: first ckpt raw, rest delta'd to it."""
    import sqlite3

    hub = TensorDex(str(tmp_path / "hub"), hydrate_all=False)
    torch.manual_seed(2)
    base = {"layer.0.w": torch.randn(64, 64), "layer.1.w": torch.randn(64, 64)}
    originals = {}
    for i in range(4):  # ck0 = reference, ck1..3 = progressively perturbed
        ckpt = {k: v + i * 1e-3 for k, v in base.items()}
        originals[i] = ckpt
        _ingest(hub, f"run/ck{i}", ckpt, tmp_path)

    res = hub.compress_bundle([f"run/ck{i}" for i in range(4)], cr_threshold=0.95)
    assert res["executed"]
    assert res["n_bases"] == 2  # one base per param (ck0's two tensors)

    db = sqlite3.connect(str(tmp_path / "hub" / "metadata.db"))

    def n_deltas(model: str) -> int:
        return db.execute(
            "SELECT COUNT(*) FROM tensor_deltas WHERE tensor_id IN "
            "(SELECT tensor_id FROM model_mappings WHERE model_name=?)",
            (model,),
        ).fetchone()[0]

    assert n_deltas("run/ck0") == 0  # reference stays raw (star anchor)
    assert n_deltas("run/ck3") == 2  # later checkpoint fully delta'd

    # Every checkpoint still reconstructs byte-for-byte.
    for i in range(4):
        for param, tid in hub.get_model_tensors(f"run/ck{i}").items():
            assert torch.equal(hub.get_tensor(tensor_id=tid), originals[i][param])


def test_plan_attach_within_model_is_param_ordered(tmp_path: Path) -> None:
    hub = TensorDex(str(tmp_path / "hub"), hydrate_all=False)
    torch.manual_seed(1)
    a = torch.randn(64, 64)
    tensors = {"layer.0.w": a, "layer.1.w": a + 1e-3, "layer.2.w": a + 2e-3}
    _ingest(hub, "org/m", tensors, tmp_path)

    plan = hub.plan_attach("org/m", cr_threshold=0.95)
    # All planned targets belong to the model; never a self-pair.
    model_tids = set(hub.get_model_tensors("org/m").values())
    for pair in plan["pairs"]:
        assert pair["target_tensor_id"] in model_tids
        assert pair["target_tensor_id"] != pair["base_tensor_id"]
