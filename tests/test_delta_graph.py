"""The delta-base graph lives in SQL (tensor_deltas), not blob headers.

Covers the #1 refactor: compress records a delta edge, gc protects bases
via SQL, the manifest closes over the base chain from SQL, and a
compressed tensor still reconstructs byte-for-byte.
"""

from __future__ import annotations

from pathlib import Path

import torch
from safetensors.torch import save_file

from tensordex.core.engine import TensorDex


def _ingest_one(hub: TensorDex, model: str, name: str, tensor: torch.Tensor, tmp: Path) -> str:
    shard = tmp / f"{model.replace('/', '_')}.safetensors"
    save_file({name: tensor}, str(shard))
    hub.init_model(model)
    mapping = hub.ingest([str(shard)], model)
    hub.commit_model(model)
    return mapping[name]


def _make_hub_with_delta(tmp_path: Path):
    """Two models sharing a shape; target compressed against base."""
    hub = TensorDex(str(tmp_path / "hub"), hydrate_all=False)
    base = torch.arange(256, dtype=torch.float32).reshape(16, 16)
    target = base + 0.01  # close to base -> good delta candidate, same shape/dtype
    base_id = _ingest_one(hub, "org/base", "w", base, tmp_path)
    target_id = _ingest_one(hub, "org/target", "w", target, tmp_path)

    res = hub.compress_pair(target_id, base_id, codec="tensorx")
    assert res["status"] == "ok"
    return hub, base_id, target_id, target


def test_compress_records_delta_edge_in_sql(tmp_path: Path) -> None:
    hub, base_id, target_id, _ = _make_hub_with_delta(tmp_path)
    # The base is protected purely via the SQL delta graph — no header scan.
    assert base_id in hub.metadata.protected_base_ids()


def test_manifest_closes_over_base_chain_from_sql(tmp_path: Path) -> None:
    hub, base_id, target_id, _ = _make_hub_with_delta(tmp_path)
    rows = hub.metadata.manifest_blobs([target_id])
    by_id = {r[0]: r for r in rows}
    # Closure pulled in the base even though we only asked for the target.
    assert set(by_id) == {target_id, base_id}
    # target row: (tid, size, shape, dtype, codec, base)
    assert by_id[target_id][4] == "tensorx"
    assert by_id[target_id][5] == base_id
    # base row is raw: no codec / base
    assert by_id[base_id][4] is None
    assert by_id[base_id][5] is None


def test_compressed_tensor_reconstructs_exactly(tmp_path: Path) -> None:
    hub, _base_id, target_id, target = _make_hub_with_delta(tmp_path)
    got = hub.get_tensor(tensor_id=target_id)
    assert torch.equal(got, target)


def test_pull_decodes_shared_base_once(tmp_path: Path, monkeypatch) -> None:
    """A base shared by N delta targets is decoded once per pull, not N times."""
    from safetensors.torch import load_file

    import tensordex.core.engine as eng

    hub = TensorDex(str(tmp_path / "hub"), hydrate_all=False)
    base = torch.arange(256, dtype=torch.float32).reshape(16, 16)
    base_id = _ingest_one(hub, "org/base", "w", base, tmp_path)

    # One model, three params, each a small delta off the shared base.
    shard = tmp_path / "tm.safetensors"
    params = {p: base + i * 0.01 for i, p in enumerate(("a", "b", "c"), 1)}
    save_file(params, str(shard))
    hub.init_model("org/tm")
    mapping = hub.ingest([str(shard)], "org/tm")
    hub.commit_model("org/tm")
    for p in ("a", "b", "c"):
        assert hub.compress_pair(mapping[p], base_id, codec="tensorx")["status"] == "ok"

    counts: dict[str, int] = {}
    real_load_blob = eng.load_blob

    def counting_load_blob(path):
        counts[str(path)] = counts.get(str(path), 0) + 1
        return real_load_blob(path)

    monkeypatch.setattr(eng, "load_blob", counting_load_blob)

    out = hub.pull("org/tm", str(tmp_path / "out"))

    base_path = str(hub.storage_backend.blob_path_for_id(base_id))
    assert counts.get(base_path, 0) == 1, "shared base should decode exactly once"

    got = load_file(out["output_path"])
    for p in ("a", "b", "c"):
        assert torch.equal(got[p], params[p])


def test_gc_protects_base_referenced_only_by_delta(tmp_path: Path) -> None:
    hub, base_id, target_id, target = _make_hub_with_delta(tmp_path)
    backend = hub.storage_backend

    # Drop the model that maps the base → base is now an orphan in
    # model_mappings, kept alive solely by the delta edge.
    hub.rm("org/base")

    result = hub.gc()
    assert result["tensors_deleted"] == 0          # nothing collectable
    assert result["bases_protected"] >= 1          # base protected via SQL
    assert backend.blob_path_for_id(base_id).exists()
    assert backend.blob_path_for_id(target_id).exists()

    # And the compressed target still reconstructs (its base survived gc).
    assert torch.equal(hub.get_tensor(tensor_id=target_id), target)
