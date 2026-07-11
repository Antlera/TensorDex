"""HuggingFace → TensorDex bridge.

Python's only jobs on the ingest path are:

1. ``snapshot_download`` the repo's ``.safetensors`` (+ metadata json) files.
2. Enumerate the downloaded shard paths.
3. Call ``hub.ingest(paths, model_name, param_filter=...)`` — Rust takes
   over from there (mmap, hash, dedup, BCS, write, SQL).

Reading a model *back* out is still pure Python — build the ``nn.Module``
skeleton via ``transformers``, pull each tensor via ``hub.get_tensor``,
and call ``load_state_dict``.
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional, Set

import torch
from huggingface_hub import snapshot_download

from tensordex.core.engine import TensorDex

if TYPE_CHECKING:  # pragma: no cover - optional dep, typing only
    from transformers import PreTrainedModel


logger = logging.getLogger(__name__)


def _convert_torch_bin_to_safetensors(snapshot_dir: str) -> List[str]:
    """Convert PyTorch ``.bin`` weight shards to safetensors, in place.

    TensorDex stores safetensors internally, but a *source* checkpoint may
    ship legacy pickle ``.bin`` shards (e.g. LLM360/Amber). Load each weight
    shard on CPU and re-save it as ``.safetensors`` beside it, one shard at a
    time to bound peak memory. Tied/shared tensors are cloned so each is
    materialized independently (identical bytes simply dedup at ingest).
    """
    from safetensors.torch import save_file

    root = Path(snapshot_dir)
    bins = sorted(p for p in root.rglob("*.bin") if p.name.startswith("pytorch_model"))
    if not bins:  # fall back to any .bin that isn't optimizer / training state
        skip = {"optimizer.bin", "scheduler.bin", "training_args.bin", "rng_state.bin"}
        bins = sorted(p for p in root.rglob("*.bin") if p.name not in skip)

    out: List[str] = []
    for shard in bins:
        state = torch.load(str(shard), map_location="cpu", weights_only=True)
        tensors = {
            k: v.detach().to("cpu").contiguous().clone()
            for k, v in state.items()
            if isinstance(v, torch.Tensor)
        }
        if not tensors:
            continue
        target = shard.with_suffix(".safetensors")
        save_file(tensors, str(target))
        out.append(str(target))
        del state, tensors  # release before loading the next shard
    return sorted(out)


def _snapshot(
    hf_model_id: str, tmp_cache_dir: str, revision: Optional[str] = None
) -> List[str]:
    """Download the repo's weights and return sorted safetensors paths.

    Prefers ``.safetensors``; if a repo ships only legacy PyTorch ``.bin``
    shards, fetch and convert them — TensorDex stores safetensors internally,
    so the *source* format does not matter. ``revision`` selects a branch/tag/
    commit — e.g. a training-step checkpoint like ``step1000``.
    """
    snapshot_dir = snapshot_download(
        repo_id=hf_model_id,
        revision=revision,
        allow_patterns=["*.safetensors", "*.json"],
        cache_dir=tmp_cache_dir,
        local_dir=tmp_cache_dir,
    )
    files = sorted(str(p) for p in Path(snapshot_dir).rglob("*.safetensors"))
    if files:
        return files

    # No safetensors — fall back to PyTorch .bin shards and convert them.
    logger.info("[%s] No safetensors found; fetching .bin shards to convert", hf_model_id)
    snapshot_dir = snapshot_download(
        repo_id=hf_model_id,
        revision=revision,
        allow_patterns=["*.bin", "*.json"],
        cache_dir=tmp_cache_dir,
        local_dir=tmp_cache_dir,
    )
    files = _convert_torch_bin_to_safetensors(snapshot_dir)
    if not files:
        raise ValueError(
            f"No .safetensors or PyTorch .bin weights found in HF repo '{hf_model_id}'."
        )
    return files


def _run_ingest(
    hub: TensorDex,
    hf_model_id: str,
    logical_name: str,
    param_filter: Optional[Iterable[str]],
    revision: Optional[str] = None,
) -> Dict[str, str]:
    """Download + delegate to Rust, wrapping the model lifecycle transitions."""
    hub.init_model(logical_name)
    logger.info("[%s] Starting ingestion from %s", logical_name, hf_model_id)
    try:
        with tempfile.TemporaryDirectory(
            dir=os.environ.get("TDB_HF_TMP_DIR") or None
        ) as tmp:
            files = _snapshot(hf_model_id, tmp, revision=revision)
            logger.info(
                "[%s] Ingesting %d shard(s) via hub.ingest", logical_name, len(files)
            )
            result = hub.ingest(
                files, model_name=logical_name, param_filter=param_filter
            )
        hub.commit_model(logical_name)
        logger.info(
            "[%s] Commit complete (%d tensors ingested)", logical_name, len(result)
        )
        return result
    except Exception:
        hub.fail_model(logical_name)
        logger.warning("[%s] Ingestion failed; marking as failed", logical_name)
        raise


def ingest_model(
    hub: TensorDex,
    hf_model_id: str,
    *,
    stored_model_name: Optional[str] = None,
    revision: Optional[str] = None,
) -> Dict[str, str]:
    """Download and ingest every tensor from an HF safetensors checkpoint."""
    return _run_ingest(
        hub,
        hf_model_id,
        logical_name=stored_model_name or hf_model_id,
        param_filter=None,
        revision=revision,
    )


def ingest_model_partial(
    hub: TensorDex,
    hf_model_id: str,
    target_params: Set[str],
    *,
    stored_model_name: Optional[str] = None,
    revision: Optional[str] = None,
) -> Dict[str, str]:
    """Download and ingest only the selected parameters from an HF model."""
    if not target_params:
        return {}
    logical_name = stored_model_name or hf_model_id
    result = _run_ingest(
        hub, hf_model_id, logical_name, param_filter=target_params, revision=revision
    )
    missing = set(target_params) - set(result.keys())
    if missing:
        logger.warning(
            "Parameters not found in repo %s: %s",
            hf_model_id,
            ", ".join(sorted(missing)),
        )
    return result


def load_hf_model_from_db(
    hub: TensorDex,
    stored_model_name: str,
    *,
    hf_model_id_for_config: Optional[str] = None,
    model_class: Optional[type[PreTrainedModel]] = None,
    device: str = "cpu",
) -> PreTrainedModel:
    """Reconstruct a HuggingFace model using tensors stored inside TensorDex."""
    try:
        from transformers import AutoConfig, AutoModelForCausalLM
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "transformers is required to load HF models from TensorDex"
        ) from exc

    config_id = hf_model_id_for_config or stored_model_name
    config = AutoConfig.from_pretrained(config_id)
    model_cls = model_class or AutoModelForCausalLM
    model = model_cls.from_config(config)

    mappings = hub.get_model_tensors(stored_model_name)
    if not mappings:
        raise ValueError(f"No tensors found for model '{stored_model_name}'")

    state_dict: Dict[str, torch.Tensor] = {
        param_name: hub.get_tensor(
            tensor_id=tensor_id,
            model_name=stored_model_name,
            param_name=param_name,
        )
        for param_name, tensor_id in mappings.items()
    }

    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    return model


__all__ = [
    "ingest_model",
    "ingest_model_partial",
    "load_hf_model_from_db",
]
