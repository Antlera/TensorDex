#!/usr/bin/env python3
"""End-to-end TensorDex demo.

Simulates a base model + a fine-tune that shares most weights and tweaks a
few, then walks the full lifecycle and prints what happens at each step:

    ingest (tensor-level dedup) -> compress (raw -> delta) -> inspect the
    SQL delta graph -> integrity-verified pull -> sharded pull -> gc.

Run after `pip install -e .` (or `maturin develop`):

    python examples/demo.py
"""

from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from tensordex.core.engine import TensorDex


def rule(title: str) -> None:
    print(f"\n\033[1m{'─' * 4} {title} {'─' * (66 - len(title))}\033[0m")


def mib(n: int) -> str:
    return f"{n / 1024 / 1024:.2f} MiB"


def main() -> None:
    torch.manual_seed(0)
    with tempfile.TemporaryDirectory(prefix="tensordex-demo.") as tmp:
        tmp_path = Path(tmp)
        hub = TensorDex(str(tmp_path / "hub"))  # lazy by default now

        # ---- build a base model + a fine-tune of it ----------------------
        # Frozen tensors are byte-identical (→ dedup); the fine-tune changes
        # ~5% of one weight matrix (→ a sparse, highly compressible delta).
        embed = torch.randn(4096, 512)
        frozen = torch.randn(512, 512)
        attn_base = torch.randn(512, 512)

        attn_ft = attn_base.clone()
        idx = torch.randperm(attn_ft.numel())[: attn_ft.numel() // 20]
        attn_ft.view(-1)[idx] += 0.1  # 5% of weights nudged

        base_model = {"embed.weight": embed, "attn.weight": attn_base, "norm.weight": frozen}
        ft_model = {"embed.weight": embed, "attn.weight": attn_ft, "norm.weight": frozen}

        rule("1. ingest two models (tensor-level dedup)")
        for name, sd in (("base", base_model), ("finetune", ft_model)):
            shard = tmp_path / f"{name}.safetensors"
            save_file(sd, str(shard))
            hub.init_model(name)
            hub.ingest([str(shard)], name)
            hub.commit_model(name)
            print(f"  ingested {name:9s} ({len(sd)} params)")

        stats = hub.get_statistics()
        print(
            f"  → {stats['total_models']} models, 6 logical params, "
            f"but only {stats['total_tensors']} unique tensors stored "
            f"(embed + norm are shared → stored once)"
        )

        rule("2. compress the fine-tune against the base (raw → delta)")
        base_id = hub.get_model_tensors("base")["attn.weight"]
        ft_id = hub.get_model_tensors("finetune")["attn.weight"]
        res = hub.compress_pair(ft_id, base_id, codec="tensorx")
        print(
            f"  attn.weight: {mib(res['original_bytes'])} → {mib(res['compressed_bytes'])} "
            f"\033[1m({res['ratio']:.1f}x)\033[0m via {res['codec']} delta"
        )

        rule("3. the delta-base graph lives in SQL (tensor_deltas)")
        db = sqlite3.connect(hub.storage_dir / "metadata.db")
        for tid, base, codec in db.execute(
            "SELECT tensor_id, base_tensor_id, codec FROM tensor_deltas"
        ):
            print(f"  {tid[:12]}…  --delta({codec})-->  {base[:12]}…")
        print(f"  protected bases (gc-safe): {[b[:12] + '…' for b in hub.metadata.protected_base_ids()]}")

        rule("4. pull the fine-tune back, with integrity verification")
        out = hub.pull("finetune", str(tmp_path / "out"), verify=True)
        restored = load_file(out["output_path"])
        exact = all(torch.equal(restored[k], ft_model[k]) for k in ft_model)
        print(f"  pull --verify → {out['num_tensors']} tensors, byte-exact round-trip: {exact}")
        print("  (attn.weight was delta-decoded against the base, then re-hashed to its id)")

        rule("5. sharded pull (bounds peak memory; HF-style index)")
        sharded = hub.pull("finetune", str(tmp_path / "sharded"), max_shard_size=8 * 1024 * 1024)
        print(f"  --max-shard-size 8MiB → {sharded['shards']} shards + index.json")

        rule("6. gc keeps the base alive because a delta still needs it")
        hub.rm("base")  # base model's mappings gone, but ft's delta needs base attn
        gc = hub.gc()
        print(
            f"  removed 'base' model; gc deleted {gc['tensors_deleted']} tensors, "
            f"protected {gc['bases_protected']} delta base(s)"
        )
        still_ok = torch.equal(hub.get_tensor(tensor_id=ft_id), attn_ft)
        print(f"  finetune still reconstructs after gc: {still_ok}")

        print("\n\033[1mdemo complete ✓\033[0m")


if __name__ == "__main__":
    main()
