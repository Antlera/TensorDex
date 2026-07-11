#!/usr/bin/env python3
"""Download a set of checkpoints, compress them small, and load one back.

Walks the whole TensorDex lifecycle on real models from the Hub:

    download → ingest (tensor-level dedup) → compress (FlexSplit / --bundle)
    → load a model back out byte-exact → generate with it, perfectly restored.

The default is a real training **series** — eight adjacent Pythia-160m
checkpoints, compressed with **FlexSplit**, which opens a raw base wherever it
pays off so each checkpoint deltas against a nearby one. Consecutive
checkpoints differ only slightly, so the deltas are tiny and compress hard
(2.25x smaller, ~56% saved, lossless). FlexSplit holds ~56% even at 100
checkpoints, where a single-base star drops to ~54%. What matters is that
checkpoints are *close* in training: a series whose checkpoints are far
apart (or unrelated models) barely compresses.

    pip install .  &&  pip install transformers       # transformers for step 5
    python examples/compress_checkpoints.py            # ~4.8 GB download

Point it at other checkpoints — two forms are accepted:

    # one repo, many revisions (a training run's checkpoints)
    python examples/compress_checkpoints.py EleutherAI/pythia-160m \\
        step138000 step139000 step140000 step141000 step142000 step143000

    # a collection of full model ids (fine-tunes / variants of one base)
    python examples/compress_checkpoints.py \\
        unsloth/Llama-3.2-3B unsloth/Llama-3.2-3B-Instruct

Source format does not matter — repos that ship PyTorch ``.bin`` shards are
converted to safetensors on ingest automatically.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from safetensors.torch import load_file

from tensordex import TensorDex

# Keep the demo output clean — silence HuggingFace's per-file download bars.
try:
    from huggingface_hub.utils import disable_progress_bars

    disable_progress_bars()
except Exception:  # pragma: no cover - older huggingface_hub
    pass

# Default: a real training series — adjacent checkpoints barely differ, so
# their deltas are tiny and compress hard (the workload delta compression is for).
DEFAULT_MODEL = "EleutherAI/pythia-160m"
DEFAULT_REVISIONS = [f"step{s}000" for s in range(136, 144)]  # 8 adjacent ckpts


def du(path: Path) -> int:
    """Total bytes of every file under ``path``."""
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def human(n: float) -> str:
    n = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if n < 1024 or unit == "TiB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.2f} {unit}"
        n /= 1024
    return f"{n:.2f} TiB"


def rule(title: str) -> None:
    print(f"\n\033[1m{'─' * 3} {title} {'─' * (62 - len(title))}\033[0m")


def plan_downloads(argv: list[str]) -> list[tuple[str, str | None, str]]:
    """Return a list of (hf_id, revision, stored_name) to fetch.

    - no args              -> the default Pythia-160m checkpoint series
    - all args contain "/" -> a collection of full model ids (revision=None)
    - else                 -> argv[0] is a repo, the rest are its revisions
    """
    if not argv:
        short = DEFAULT_MODEL.split("/")[-1]
        return [(DEFAULT_MODEL, r, f"{short}@{r}") for r in DEFAULT_REVISIONS]
    if all("/" in a for a in argv):
        return [(m, None, m.split("/")[-1]) for m in argv]
    model_id, revs = argv[0], argv[1:]
    short = model_id.split("/")[-1]
    return [(model_id, r, f"{short}@{r}") for r in revs]


def main() -> None:
    jobs = plan_downloads(sys.argv[1:])

    with tempfile.TemporaryDirectory(prefix="tensordex-ckpts.") as tmp:
        hub_path = Path(tmp) / "hub"
        hub = TensorDex(str(hub_path))

        # 1. Download each checkpoint and ingest it (tensor-level dedup).
        rule(f"1. download + ingest {len(jobs)} checkpoints")
        names = []
        for hf_id, rev, name in jobs:
            tag = f"{hf_id}@{rev}" if rev else hf_id
            print(f"  ↓ {tag}  →  {name}")
            hub.download(hf_id, stored_model_name=name, revision=rev)
            names.append(name)

        # 2. Ingested. Identical tensors dedup to one blob; adjacent checkpoints
        #    share none, so here dedup is ~0 and the savings come from step 3.
        baseline = sum(hub.info(n)["total_bytes"] for n in names)
        db_bytes = sum(p.stat().st_size for p in hub_path.glob("metadata.db*"))
        blob_bytes = du(hub_path) - db_bytes
        rule("2. ingested — content-addressed (dedup-ready)")
        print(f"  all checkpoints in full : {human(baseline)}")
        print(f"  stored as blobs         : {human(blob_bytes)}  "
              f"({blob_bytes / baseline * 100:.1f}% — identical tensors dedup; adjacent ckpts share none)")
        print(f"  + sqlite metadata       : {human(db_bytes)}")

        # 3. Compress the whole group at once with FlexSplit — it opens a raw base
        #    wherever it pays off, so each checkpoint deltas against a *nearby*
        #    one (not all against the earliest). The win grows with run length.
        rule("3. compress the group — FlexSplit (compress --bundle)")
        res = hub.compress_bundle(names, strategy="flexsplit")
        after = du(hub_path)
        print(f"  plan          : {res['n_bases']} raw base(s), {res['n_pairs']} delta(s)")
        print(f"  on disk now   : {human(after)}")
        print(f"  vs full       : {after / baseline * 100:.1f}%  →  "
              f"\033[1;36m{baseline / after:.2f}× smaller, "
              f"{(1 - after / baseline) * 100:.1f}% saved\033[0m")

        # 4. Load a model straight back out of the store.
        rule("4. load a model from the hub (byte-exact)")
        name = names[-1]
        params = list(hub.get_model_tensors(name))
        sample = "model.embed_tokens.weight" if "model.embed_tokens.weight" in params else params[0]
        w = hub.get_tensor(model_name=name, param_name=sample)   # in-memory, no files
        print(f"  hub.get_tensor('{name}', '{sample}') → {tuple(w.shape)} {w.dtype}")

        out = Path(tmp) / "restored"
        hub.pull(name, str(out), verify=True)                    # full .safetensors, re-hashed
        sd = load_file(str(out / "model.safetensors"))
        print(f"  hub.pull(...) → {out}/model.safetensors  ({len(sd)} tensors)")
        print("  ↳ a standard safetensors dir — load it with transformers / vLLM directly.")

        # 5. Prove the restore is perfect: load the reconstructed weights into a
        #    transformers model and actually generate. (transformers optional.)
        rule("5. run the restored model (transformers)")
        cfg_id, cfg_rev = jobs[-1][0], jobs[-1][1]
        try:
            import torch
            from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

            config = AutoConfig.from_pretrained(cfg_id, revision=cfg_rev)
            model = AutoModelForCausalLM.from_config(config)
            missing, unexpected = model.load_state_dict(sd, strict=False)
            model.eval()
            tok = AutoTokenizer.from_pretrained(cfg_id)
            prompt = "The meaning of life is"
            ids = tok(prompt, return_tensors="pt")
            with torch.no_grad():
                gen = model.generate(**ids, max_new_tokens=20, do_sample=False)
            print(f"  loaded {cfg_id} from the reconstructed weights "
                  f"({len(missing)} missing / {len(unexpected)} unexpected keys)")
            print(f"  prompt     : {prompt!r}")
            print(f"  generated  : {tok.decode(gen[0], skip_special_tokens=True)!r}")
            print("  ↳ the restored weights load and generate — a perfect, usable model.")
        except ImportError:
            print("  (skipped — `pip install transformers` to run the restored model)")
        except Exception as exc:  # noqa: BLE001 - bonus step, never fail the demo
            print(f"  (skipped — generation failed: {type(exc).__name__}: {exc})")

    print("\nDone — download → compress → load → generate, perfectly restored.")


if __name__ == "__main__":
    main()
