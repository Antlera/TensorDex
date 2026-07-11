"""Ingest from legacy PyTorch ``.bin`` checkpoints.

TensorDex stores safetensors internally, but a source checkpoint may ship
pickle ``.bin`` shards (e.g. LLM360/Amber). The HF bridge converts those to
safetensors before the normal ingest pipeline runs.
"""

from __future__ import annotations

from pathlib import Path

import torch
from safetensors.torch import load_file

from tensordex import TensorDex
from tensordex.integrations.hf_io import _convert_torch_bin_to_safetensors


def test_convert_bin_preserves_dtype_and_tied_weights(tmp_path: Path) -> None:
    embed = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    # sharded .bin; embed and lm_head are tied (share storage)
    torch.save(
        {"embed.weight": embed, "lm_head.weight": embed},
        tmp_path / "pytorch_model-00001-of-00002.bin",
    )
    torch.save(
        {"attn.weight": torch.randn(4, 4).to(torch.bfloat16)},
        tmp_path / "pytorch_model-00002-of-00002.bin",
    )
    torch.save({"step": 1}, tmp_path / "training_args.bin")  # must be ignored

    out = _convert_torch_bin_to_safetensors(str(tmp_path))
    assert len(out) == 2

    merged: dict[str, torch.Tensor] = {}
    for path in out:
        merged.update(load_file(path))

    assert set(merged) == {"embed.weight", "lm_head.weight", "attn.weight"}
    assert torch.equal(merged["embed.weight"], embed)
    assert torch.equal(merged["lm_head.weight"], embed)  # tied → materialized
    assert merged["attn.weight"].dtype == torch.bfloat16  # bf16 preserved


def test_bin_ingest_round_trip(tmp_path: Path) -> None:
    torch.save(
        {"w": torch.randn(8, 8).to(torch.bfloat16)},
        tmp_path / "pytorch_model.bin",
    )
    sfs = _convert_torch_bin_to_safetensors(str(tmp_path))

    hub = TensorDex(str(tmp_path / "hub"))
    hub.init_model("m")
    hub.ingest(sfs, model_name="m")
    hub.commit_model("m")

    w = hub.get_tensor(model_name="m", param_name="w")
    assert tuple(w.shape) == (8, 8)
    assert w.dtype == torch.bfloat16
