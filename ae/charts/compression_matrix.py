"""
Compression Ratio Matrix Heatmap — All-to-All tensor pairing within a model family.

Uses BCS fingerprints + Hybrid prediction model to show reduction ratios
across models for a specific layer's tensors.

Charts:
    - compression_matrix_heatmap: Reduction ratio heatmap (ref model vs N models)
    - compression_matrix_bar: Mean reduction ratio bar chart
"""

import json
import re
import sys
import time
import sqlite3
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Optional

# Ensure project root is importable
from ._root import PROJECT_ROOT as _PROJECT_ROOT
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(_PROJECT_ROOT))

from tests.test_flexsplit import (
    HYBRID_COEFFS,
    BCS_W,
    predict_cr_hybrid,
    bcs_norm_hamming,
    load_bcs_fingerprints_from_db,
    _hydrate_for_models,
)
from tensordb.core.engine import load_tensordb
from algorithms.micro_algorithms import tensor_nbytes

# ─────────────────────────────────────────────────────────────────────
# Config defaults
# ─────────────────────────────────────────────────────────────────────

DEFAULT_CONFIG = {
    "db_path": str(_PROJECT_ROOT / "data/tensordb_s3"),
    "models_json": str(_PROJECT_ROOT / "data/models/models.json"),
    "base_model": "Qwen/Qwen2.5-7B",
    "ref_model": "real-jiakai/Qwen2.5-7B-Instruct-Jiakai",
    "max_models": 200,
    "n_select": 50,
    "dedup_threshold": 0.98,
    "layer": "layers.0.",
    "exclude_layernorm": True,
    "exclude_bias": True,
}

# ─────────────────────────────────────────────────────────────────────
# Helpers (same as test_compression_matrix.py)
# ─────────────────────────────────────────────────────────────────────

def _load_model_family(models_json: str, base_model: str) -> List[str]:
    with open(models_json) as f:
        data = json.load(f)
    return [base_model] + data["base_models"].get(base_model, [])


def _get_model_tensors(db, model_name: str) -> Dict[str, str]:
    result = {}
    for (mn, pn), tid in db._model_index.items():
        if mn == model_name:
            result[pn] = tid
    return result


def _compute_cr_matrix_for_tensor(tensor_name, model_names, model_tensor_maps,
                                   bcs_fp_db, db):
    n = len(model_names)
    tids, fps, valid_mask = [], [], []
    n_bits = None
    for i, tmap in enumerate(model_tensor_maps):
        tid = tmap.get(tensor_name)
        if tid and tid in bcs_fp_db:
            tids.append(tid)
            fps.append(bcs_fp_db[tid])
            valid_mask.append(i)
            if n_bits is None:
                meta = db.metadata_db.get(tid)
                if meta:
                    n_bits = int(np.prod(meta.shape)) * 16
        else:
            tids.append(None)
            fps.append(None)
    if len(valid_mask) < 2 or n_bits is None:
        return None
    cr_matrix = np.full((n, n), np.nan, dtype=np.float64)
    for ii, i in enumerate(valid_mask):
        cr_matrix[i, i] = 0.0
        for jj in range(ii + 1, len(valid_mask)):
            j = valid_mask[jj]
            dist = bcs_norm_hamming(fps[i], fps[j], n_bits)
            cr = predict_cr_hybrid(dist)
            cr_matrix[i, j] = cr
            cr_matrix[j, i] = cr
    return cr_matrix


def _filter_layer_tensors(tensor_names, layer_prefix, exclude_ln, exclude_bias):
    result = [tn for tn in tensor_names if layer_prefix in tn]
    if exclude_ln:
        result = [tn for tn in result if "layernorm" not in tn]
    if exclude_bias:
        result = [tn for tn in result if not tn.endswith(".bias")]
    return sorted(result)


def _simplify_tensor_name(tn: str, layer_alias: str = None) -> str:
    m = re.search(r'layers\.(\d+)\.', tn)
    if m:
        layer = layer_alias if layer_alias else f"layer{m.group(1)}"
    else:
        layer = ""
    short = tn
    for prefix in ["model.layers.", "layers."]:
        if short.startswith(prefix):
            short = re.sub(r'^(model\.)?layers\.\d+\.', '', short)
            break
    if short.startswith("model."):
        short = short[len("model."):]
    for old, new in [
        ("self_attn.", "attn."), ("input_layernorm", "in_ln"),
        ("post_attention_layernorm", "post_ln"),
        ("q_proj", "q"), ("k_proj", "k"), ("v_proj", "v"), ("o_proj", "o"),
        ("gate_proj", "gate"), ("up_proj", "up"), ("down_proj", "down"),
        (".weight", ""), (".bias", ".b"), ("mlp.", "mlp."),
    ]:
        short = short.replace(old, new)
    return f"{layer}.{short}" if layer else short


# ─────────────────────────────────────────────────────────────────────
# Data loading (cached)
# ─────────────────────────────────────────────────────────────────────

_data_cache = None


def _load_data(cfg=None):
    global _data_cache
    if _data_cache is not None:
        return _data_cache

    cfg = cfg or DEFAULT_CONFIG

    # Load models
    all_family = _load_model_family(cfg["models_json"], cfg["base_model"])
    model_names = all_family[:cfg["max_models"]]
    if cfg["ref_model"] and cfg["ref_model"] not in model_names:
        model_names.append(cfg["ref_model"])

    # Load DB
    db = load_tensordb(cfg["db_path"], backend="local", hydrate_all=False)
    _hydrate_for_models(db, model_names)

    model_tensor_maps = [_get_model_tensors(db, mn) for mn in model_names]

    # Load BCS fingerprints
    all_tids = list({tid for tmap in model_tensor_maps for tid in tmap.values()})
    metadata_db_path = str(Path(cfg["db_path"]) / "metadata.db")
    bcs_fp_db = load_bcs_fingerprints_from_db(metadata_db_path, tensor_ids=all_tids)

    # Dedup
    n_models = len(model_names)
    if cfg["dedup_threshold"] < 1.0:
        base_tmap = model_tensor_maps[0]
        keep = [0]
        for j in range(1, n_models):
            tmap_j = model_tensor_maps[j]
            if not tmap_j:
                continue
            cr_sum, cr_cnt = 0.0, 0
            for tn, tid_base in base_tmap.items():
                tid_j = tmap_j.get(tn)
                if tid_base in bcs_fp_db and tid_j and tid_j in bcs_fp_db:
                    meta = db.metadata_db.get(tid_base)
                    if meta:
                        n_bits = int(np.prod(meta.shape)) * 16
                        dist = bcs_norm_hamming(bcs_fp_db[tid_base], bcs_fp_db[tid_j], n_bits)
                        cr_sum += predict_cr_hybrid(dist)
                        cr_cnt += 1
            if cr_cnt > 0 and (1.0 - cr_sum / cr_cnt) <= cfg["dedup_threshold"]:
                keep.append(j)
        model_names = [model_names[i] for i in keep]
        model_tensor_maps = [model_tensor_maps[i] for i in keep]
        n_models = len(model_names)

    # Common tensors
    tn_counts = defaultdict(int)
    for tmap in model_tensor_maps:
        for tn in tmap:
            tn_counts[tn] += 1
    common_tensors = sorted([tn for tn, c in tn_counts.items() if c >= 2])

    # CR matrices
    tensor_cr_matrices = {}
    for tn in common_tensors:
        cr_mat = _compute_cr_matrix_for_tensor(tn, model_names, model_tensor_maps, bcs_fp_db, db)
        if cr_mat is not None:
            tensor_cr_matrices[tn] = cr_mat

    _data_cache = {
        "model_names": model_names,
        "model_tensor_maps": model_tensor_maps,
        "tensor_cr_matrices": tensor_cr_matrices,
        "db": db,
        "bcs_fp_db": bcs_fp_db,
    }
    return _data_cache


# ─────────────────────────────────────────────────────────────────────
# Chart functions
# ─────────────────────────────────────────────────────────────────────

def chart_compression_matrix_heatmap(rc):
    """Reduction ratio heatmap: ref model's layer tensors vs selected diverse models."""
    cfg = DEFAULT_CONFIG.copy()
    data = _load_data(cfg)

    model_names = data["model_names"]
    model_tensor_maps = data["model_tensor_maps"]
    tensor_cr_matrices = data["tensor_cr_matrices"]
    n_models = len(model_names)

    # Resolve ref model
    ref_idx = model_names.index(cfg["ref_model"]) if cfg["ref_model"] in model_names else 0

    # Mean CR per model vs ref
    all_cr = np.full(n_models, np.nan)
    cr_counts = np.zeros(n_models, dtype=np.int64)
    for cr_mat in tensor_cr_matrices.values():
        row = cr_mat[ref_idx, :]
        for j in range(n_models):
            if not np.isnan(row[j]) and j != ref_idx:
                if np.isnan(all_cr[j]):
                    all_cr[j] = 0.0
                all_cr[j] += row[j]
                cr_counts[j] += 1
    for j in range(n_models):
        if cr_counts[j] > 0:
            all_cr[j] /= cr_counts[j]

    # Select top-N most diverse
    candidates = [(j, all_cr[j]) for j in range(n_models)
                  if not np.isnan(all_cr[j]) and j != ref_idx]
    candidates.sort(key=lambda x: -x[1])
    selected = [j for j, _ in candidates[:cfg["n_select"]]]
    n_sel = len(selected)

    # Layer tensors
    ref_tmap = model_tensor_maps[ref_idx]
    layer_tensors = _filter_layer_tensors(
        list(ref_tmap.keys()), cfg["layer"],
        cfg["exclude_layernorm"], cfg["exclude_bias"])

    # Add global tensors
    for tn in sorted(ref_tmap.keys()):
        if "embed_tokens" in tn or (tn.endswith("norm.weight") and "layers." not in tn):
            if tn not in layer_tensors and tn in tensor_cr_matrices:
                layer_tensors.append(tn)

    # Build heatmap
    heatmap = np.full((len(layer_tensors), n_sel), np.nan)
    for row_i, tn in enumerate(layer_tensors):
        if tn in tensor_cr_matrices:
            cr_mat = tensor_cr_matrices[tn]
            for col_i, j in enumerate(selected):
                heatmap[row_i, col_i] = 1.0 - cr_mat[ref_idx, j]

    # Sort columns by last row (norm) reduction ratio, descending
    last_row = heatmap[-1, :]
    sort_order = np.argsort(np.where(np.isnan(last_row), -np.inf, last_row))[::-1]
    heatmap = heatmap[:, sort_order]
    selected = [selected[i] for i in sort_order]

    lm = re.search(r'layers\.(\d+)', cfg["layer"])
    layer_alias = f"layer{lm.group(1)}" if lm else None
    display_names = [_simplify_tensor_name(tn, layer_alias=layer_alias) for tn in layer_tensors]

    # Plot
    FS = rc.get("tick_label_size", 42)
    fig, ax = plt.subplots(figsize=(rc.get("figsize_w", 25), rc.get("figsize_h", 10)))

    im = ax.imshow(heatmap, aspect="auto", cmap="Reds",
                   vmin=0.4, vmax=1.0, interpolation="nearest")

    ax.set_yticks(range(len(display_names)))
    ax.set_yticklabels(display_names, fontsize=FS, family="monospace")
    xtick_step = max(1, n_sel // 10)
    xticks = list(range(0, n_sel, xtick_step))
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks, fontsize=FS)

    ax.set_ylabel("Tensor", fontsize=FS)
    ax.set_xlabel("Model Index", fontsize=FS)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Reduction Ratio", fontsize=FS)
    cbar.ax.tick_params(labelsize=FS)

    plt.tight_layout()
    return fig


def _compute_diversity(tensor_cr_matrices, n_models):
    """For each model, count how many distinct source models provide the best tensor match.

    Uses ALL tensor CR matrices (not filtered by layer).
    Returns dict: model_idx -> n_unique_sources.
    """
    diversity = {}
    for mi in range(n_models):
        sources = set()
        for cr_mat in tensor_cr_matrices.values():
            row = cr_mat[mi, :].copy()
            row[mi] = np.inf
            row[np.isnan(row)] = np.inf
            if not np.all(np.isinf(row)):
                sources.add(int(np.argmin(row)))
        diversity[mi] = len(sources)
    return diversity


def chart_compression_matrix_bar(rc):
    """Mean reduction ratio bar chart: ref model vs selected diverse models."""
    cfg = DEFAULT_CONFIG.copy()
    data = _load_data(cfg)

    model_names = data["model_names"]
    tensor_cr_matrices = data["tensor_cr_matrices"]
    n_models = len(model_names)

    ref_idx = model_names.index(cfg["ref_model"]) if cfg["ref_model"] in model_names else 0

    # Mean CR per model
    all_cr = np.full(n_models, np.nan)
    cr_counts = np.zeros(n_models, dtype=np.int64)
    for cr_mat in tensor_cr_matrices.values():
        row = cr_mat[ref_idx, :]
        for j in range(n_models):
            if not np.isnan(row[j]) and j != ref_idx:
                if np.isnan(all_cr[j]):
                    all_cr[j] = 0.0
                all_cr[j] += row[j]
                cr_counts[j] += 1
    for j in range(n_models):
        if cr_counts[j] > 0:
            all_cr[j] /= cr_counts[j]

    # Select top-N
    candidates = [(j, all_cr[j]) for j in range(n_models)
                  if not np.isnan(all_cr[j]) and j != ref_idx]
    candidates.sort(key=lambda x: -x[1])
    selected = sorted([j for j, _ in candidates[:cfg["n_select"]]])
    n_sel = len(selected)

    sel_rr = np.array([1.0 - all_cr[j] for j in selected])

    FS = rc.get("tick_label_size", 42)
    fig, ax = plt.subplots(figsize=(rc.get("figsize_w", 25), rc.get("figsize_h", 10)))

    ax.bar(range(n_sel), sel_rr, color="steelblue", edgecolor="none", width=0.9)
    xtick_step = max(1, n_sel // 10)
    xticks = list(range(0, n_sel, xtick_step))
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks, fontsize=FS)
    ax.set_ylabel("Mean Reduction Ratio", fontsize=FS)
    ax.set_xlabel("Model Index", fontsize=FS)
    ax.tick_params(axis='y', labelsize=FS)
    ax.axhline(y=0.25, color="red", linestyle="--", alpha=0.5, label="standalone zstd (~0.25)")
    ax.legend(fontsize=FS * 0.6)
    plt.tight_layout()
    return fig


def chart_source_diversity(rc):
    """Per-model best-source diversity: # distinct source models across ALL tensors."""
    cfg = DEFAULT_CONFIG.copy()
    data = _load_data(cfg)

    model_names = data["model_names"]
    tensor_cr_matrices = data["tensor_cr_matrices"]
    n_models = len(model_names)

    diversity = _compute_diversity(tensor_cr_matrices, n_models)
    div_values = [diversity.get(i, 0) for i in range(n_models)]

    FS = rc.get("tick_label_size", 42)
    fig, ax = plt.subplots(figsize=(rc.get("figsize_w", 25), rc.get("figsize_h", 10)))

    ax.bar(range(n_models), div_values, color="coral", edgecolor="none", width=0.9)
    xtick_step = max(1, n_models // 10)
    xticks = list(range(0, n_models, xtick_step))
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks, fontsize=FS)
    ax.set_ylabel("# Unique Source Models", fontsize=FS)
    ax.set_xlabel("Model Index", fontsize=FS)
    ax.tick_params(axis='y', labelsize=FS)
    plt.tight_layout()
    return fig


# ── Chart Registry ────────────────────────────────────────────────────

CHARTS = {
    "compression_matrix_heatmap": {
        "name": "Compression Matrix Heatmap",
        "category": "Model Family Analysis",
        "desc": "Reduction ratio heatmap: ref model layer tensors vs diverse models in the same family",
        "fn": chart_compression_matrix_heatmap,
    },
    "compression_matrix_bar": {
        "name": "Compression Matrix Bar",
        "category": "Model Family Analysis",
        "desc": "Mean reduction ratio bar chart: ref model vs diverse models",
        "fn": chart_compression_matrix_bar,
    },
    "source_diversity": {
        "name": "Source Diversity",
        "category": "Model Family Analysis",
        "desc": "Per-model best-source diversity: # distinct source models across all tensors",
        "fn": chart_source_diversity,
    },
}
