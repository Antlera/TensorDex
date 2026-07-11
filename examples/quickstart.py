#!/usr/bin/env python3
"""End-to-end TensorDex round-trip: ingest → get → pull.

Run it after installing the package (``pip install -e .`` or
``maturin develop``):

    python examples/quickstart.py

It creates a throwaway hub in a temp dir, ingests a local safetensors
shard, reads one tensor back, reconstructs the whole model to a new
safetensors file, and asserts the bytes round-trip exactly.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from tensordex import TensorDex

MODEL = "demo/tiny"


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="tensordex-quickstart.") as tmp:
        tmp_path = Path(tmp)

        # 1. Produce a local safetensors shard to ingest.
        original = {
            "model.embed.weight": torch.arange(12, dtype=torch.float32).reshape(3, 4),
            "model.norm.weight": torch.ones(4, dtype=torch.float16),
        }
        shard = tmp_path / "shard.safetensors"
        save_file(original, str(shard))

        # 2. Ingest. init → ingest → commit is the model lifecycle.
        hub = TensorDex(str(tmp_path / "hub"), hydrate_all=False)
        hub.init_model(MODEL)
        mapping = hub.ingest([str(shard)], model_name=MODEL)
        hub.commit_model(MODEL)
        print(f"ingested {len(mapping)} tensor(s): {sorted(mapping)}")

        # 3. Fetch a single tensor by <model>:<param>.
        embed = hub.get_tensor(model_name=MODEL, param_name="model.embed.weight")
        assert torch.equal(embed, original["model.embed.weight"])
        print(f"get_tensor model.embed.weight -> shape={tuple(embed.shape)} dtype={embed.dtype}")

        # 4. Reconstruct the whole model back to a safetensors bundle.
        out_dir = tmp_path / "out"
        result = hub.pull(MODEL, str(out_dir))
        restored = load_file(result["output_path"])

        # 5. Verify the round-trip is exact.
        assert restored.keys() == original.keys()
        for name, tensor in original.items():
            assert torch.equal(restored[name], tensor), f"mismatch in {name}"
        print(f"pull -> {result['output_path']} ({result['num_tensors']} tensors)")
        print("round-trip OK ✓")


if __name__ == "__main__":
    main()
