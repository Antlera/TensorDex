#!/usr/bin/env python3
"""
Benchmark facility-location solvers using real cached compression ratios.

Sources:
- Cached pairwise results: SQLite table `compression_results`
- Tensor metadata & fingerprints: TensorDex (local or S3)
- Solvers: heuristic split (ratio-based), primal-dual, ILP (Gurobi if present)

The script builds per-parameter problems (typically layer 0) and injects cached
ratios as edge costs (treating them as symmetric). It reports objective cost,
compression ratio, bases opened, and runtime.
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# Optional numba acceleration (falls back to Python if unavailable)
try:
    from numba import njit, prange

    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

    def njit(*args, **kwargs):
        def decorator(func):
            return func

        return decorator

    def prange(*args):
        return range(*args)

# Ensure repo root is on sys.path for direct script execution
# Run standalone (`python ae/bench/bench_facility.py`) — put this dir on the
# path so the sibling `algorithms`/`algos` packages import.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from algorithms.ilp_facility_location import ilp_facility_location_gurobi
from algorithms.micro_algorithms import FacilityParams, MicroCompressionModel, tensor_nbytes
from algorithms.primal_dual_facility_location import primal_dual_facility_location
from algos.primal_dual_ratio import primal_dual_facility_location_ratio
from algos.ilp_ratio import ilp_facility_location_ratio_gurobi


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class CacheRow:
    target: str
    base: str
    ratio: float
    param: str


@dataclass
class MetaRow:
    tensor_id: str
    model_name: Optional[str]
    param_name: Optional[str]
    shape: Tuple[int, ...]
    dtype: str


@dataclass
class Problem:
    param: str
    ids: List[str]
    sizes: np.ndarray
    numel: np.ndarray
    sqrt_max: np.ndarray
    D_raw: np.ndarray  # scaled so that D_raw / sqrt_max = normalized distance
    ratio_overrides: int
    ratio_matrix: np.ndarray  # cached ratios (inf if missing)
    cached_pairs: Dict[Tuple[str, str], float]


# ---------------------------------------------------------------------------
# Numba-accelerated helpers (mirroring tests/test_zipllm_clustering.py)
# ---------------------------------------------------------------------------

@njit(parallel=True, fastmath=True)
def find_best_split_jit(
    worse_indices,
    sizes,
    costs_to_base,
    ratio_matrix,
    base_meta_bytes,
):
    n_cands = len(worse_indices)
    n_points = sizes.shape[0]
    gains = np.empty(n_cands, dtype=np.float64)

    for k in prange(n_cands):
        cand_idx = worse_indices[k]
        # Opening new base: pay full size + metadata instead of compressed cost
        gain = -((sizes[cand_idx] + base_meta_bytes) - costs_to_base[cand_idx])

        for i in range(n_points):
            if i == cand_idx:
                continue

            ratio = ratio_matrix[i, cand_idx]
            if not np.isfinite(ratio):
                continue
            cost_new = ratio * sizes[i]
            cost_old = costs_to_base[i]

            if cost_new < cost_old:
                gain += (cost_old - cost_new)

        gains[k] = gain

    argmax_k = -1
    max_val = -np.inf
    for k in range(n_cands):
        if gains[k] > max_val:
            max_val = gains[k]
            argmax_k = k

    if max_val > 0.0:
        return max_val, worse_indices[argmax_k]
    return 0.0, -1


def load_cached_rows(
    db_path: Path,
    param_filter: Optional[str],
    target_param_exact: Optional[str] = None,
) -> Dict[str, List[CacheRow]]:
    """Load cached ratios grouped by param_name (or shape fallback)."""
    conn = sqlite3.connect(db_path)
    query = """
        SELECT target_id, base_id, param_name, shape, bytes_in, bytes_out, ratio
        FROM compression_results
        WHERE bytes_out IS NOT NULL
    """
    params: List = []
    if target_param_exact:
        # Match suffix/prefixed variants (mirror generate_layer0_plans substring logic)
        query += " AND param_name LIKE ?"
        params.append(f"%{target_param_exact}")
    elif param_filter:
        query += " AND param_name LIKE ?"
        params.append(f"%{param_filter}%")
    cur = conn.execute(query, params)
    groups: Dict[str, List[CacheRow]] = defaultdict(list)
    for target_id, base_id, param_name, shape, bytes_in, bytes_out, ratio in cur:
        param = param_name or shape or "unknown"
        # Normalize param key so prefixed variants collapse to target_param_exact
        if target_param_exact and isinstance(param, str) and param.endswith(target_param_exact):
            param = target_param_exact
        if param_filter and param_filter not in param:
            continue
        try:
            b_in = float(bytes_in) if bytes_in is not None else None
            b_out = float(bytes_out)
            r = float(ratio) if ratio is not None else (b_out / b_in if b_in else None)
        except Exception:
            continue
        if r is None or not math.isfinite(r):
            continue
        groups[param].append(CacheRow(str(target_id), str(base_id), r, param))
    conn.close()
    return groups


def build_problem(
    metadata_map: Dict[str, MetaRow],
    rows: List[CacheRow],
    *,
    max_items: Optional[int],
    min_items: int,
) -> Tuple[Optional[Problem], Optional[str]]:
    """Construct per-parameter problem using cached (symmetric) ratios only."""
    ids: List[str] = []
    cache: Dict[Tuple[str, str], float] = {}
    for row in rows:
        cache[(row.target, row.base)] = row.ratio
        ids.extend([row.target, row.base])

    ids = list(dict.fromkeys(ids))  # stable unique
    if len(ids) < min_items:
        return None, "too_few_items"
    if max_items:
        ids = ids[:max_items]
        cache = {(t, b): r for (t, b), r in cache.items() if t in ids and b in ids}
    metas = [metadata_map.get(tid) for tid in ids]
    if any(m is None for m in metas):
        return None, "missing_metadata"
    sizes = np.array([tensor_nbytes(m.shape, str(m.dtype)) for m in metas], dtype=np.float64)
    numel = np.array([int(np.prod(m.shape)) for m in metas], dtype=np.float64)
    # Distance-related artifacts are unused in ratio-only solvers; keep placeholders.
    D_raw = np.zeros((len(ids), len(ids)), dtype=np.float64)
    sqrt_max = np.ones_like(D_raw, dtype=np.float64)
    ratio_matrix = np.full_like(D_raw, np.inf, dtype=np.float64)
    np.fill_diagonal(ratio_matrix, 0.0)
    overrides = 0
    id_to_idx = {tid: idx for idx, tid in enumerate(ids)}
    for (t, b), r in cache.items():
        if t not in id_to_idx or b not in id_to_idx:
            continue
        i = id_to_idx[t]
        j = id_to_idx[b]
        if not math.isfinite(r):
            continue
        ratio_matrix[i, j] = min(ratio_matrix[i, j], r)
        if i != j:
            ratio_matrix[j, i] = min(ratio_matrix[j, i], r)
        overrides += 1
    np.fill_diagonal(D_raw, 0.0)
    return Problem(
        param=rows[0].param if rows else "unknown",
        ids=ids,
        sizes=sizes,
        numel=numel,
        sqrt_max=sqrt_max,
        D_raw=D_raw,
        ratio_overrides=overrides,
        ratio_matrix=ratio_matrix,
        cached_pairs=cache,
    ), None


def total_cost_from_ratio(ratio_matrix: np.ndarray, sizes: np.ndarray, bases: Sequence[int], base_meta_bytes: int) -> float:
    """Compute objective cost given bases and ratio matrix."""
    bases = list(bases)
    if not bases:
        return float("inf")
    base_cost = sum(sizes[b] + base_meta_bytes for b in bases)
    conn_cost = 0.0
    for i in range(len(sizes)):
        best = float("inf")
        for b in bases:
            r = ratio_matrix[i, b]
            if not math.isfinite(r):
                continue
            best = min(best, r * sizes[i])
        if not math.isfinite(best):
            return float("inf")
        conn_cost += best
    return base_cost + conn_cost


def heuristic_split(
    problem: Problem,
    base_meta_bytes: int,
    max_splits: int,
    ratio_matrix_override: Optional[np.ndarray] = None,
) -> Dict:
    """Single-step split heuristic mirroring tests/test_zipllm_clustering.py."""
    # Single-step split heuristic matching tests/test_zipllm_clustering.py,
    # accelerated with numba. We first pick the best single-base star, then
    # optionally add one more base selected from the "worse-than-average"
    # targets connected to that base.
    n = len(problem.ids)
    if n < 2:
        return {"bases": [0], "cost": float("inf"), "num_bases": 1, "splits": 0}
    ratio_matrix = ratio_matrix_override if ratio_matrix_override is not None else problem.ratio_matrix

    if max_splits <= 0:
        # Still choose best single-base star for compatibility
        best_base = 0
        best_cost = float("inf")
        for b in range(n):
            c = total_cost_from_ratio(ratio_matrix, problem.sizes, [b], base_meta_bytes)
            if c < best_cost:
                best_cost = c
                best_base = b
        return {"bases": [best_base], "cost": best_cost, "num_bases": 1, "splits": 0}

    sizes = problem.sizes.astype(np.float64)

    # Choose best base using ratio_matrix for consistency with downstream metrics
    best_base = 0
    best_cost = float("inf")
    for b in range(n):
        c = total_cost_from_ratio(ratio_matrix, sizes, [b], base_meta_bytes)
        if c < best_cost:
            best_cost = c
            best_base = b
    base_idx = best_base

    ratios_to_base = ratio_matrix[:, base_idx].astype(np.float64)
    ratios_to_base[base_idx] = 0.0
    costs_to_base = ratios_to_base * sizes
    costs_to_base[base_idx] = sizes[base_idx] + base_meta_bytes

    target_indices = [i for i in range(n) if i != base_idx]
    missing_targets = [i for i in target_indices if not np.isfinite(ratios_to_base[i])]
    if missing_targets:
        # Heuristic star cannot connect everyone because cached ratios are missing.
        return {
            "bases": [base_idx],
            "cost": float("inf"),
            "num_bases": 1,
            "splits": 0,
            "heuristic_msg": f"Missing cached ratios to {len(missing_targets)} tensors; coverage too low",
        }
    if not target_indices:
        return {
            "bases": [base_idx],
            "cost": float(costs_to_base.sum()),
            "num_bases": 1,
            "splits": 0,
        }

    avg_ratio = float(np.mean(ratios_to_base[target_indices])) if target_indices else 0.0
    worse_indices = [i for i in target_indices if ratios_to_base[i] > avg_ratio]

    if not worse_indices:
        return {
            "bases": [base_idx],
            "cost": float(costs_to_base.sum()),
            "num_bases": 1,
            "splits": 0,
            "heuristic_msg": "No points worse than average ratio",
        }

    worse_indices_arr = np.array(worse_indices, dtype=np.int64)
    best_gain, best_new_base_idx = find_best_split_jit(
        worse_indices_arr,
        sizes.astype(np.float64),
        costs_to_base.astype(np.float64),
        ratio_matrix.astype(np.float64),
        float(base_meta_bytes),
    )

    bases = [base_idx]
    cost = float(costs_to_base.sum())
    splits = 0

    if best_gain > 0.0 and best_new_base_idx >= 0:
        bases.append(int(best_new_base_idx))
        cost = cost - float(best_gain)
        splits = 1

    return {
        "bases": bases,
        "cost": cost,
        "num_bases": len(bases),
        "splits": splits,
    }


def run_primal_dual(
    problem: Problem, params: FacilityParams, model: MicroCompressionModel, ratio_matrix: Optional[np.ndarray] = None
) -> Dict:
    t0 = time.time()
    res = primal_dual_facility_location_ratio(
        tensor_ids=problem.ids,
        ratio_matrix=ratio_matrix if ratio_matrix is not None else problem.ratio_matrix,
        sizes=problem.sizes,
        base_meta_bytes=params.base_meta_bytes,
        base_storage_multiplier=params.base_storage_multiplier,
        verbose=params.verbose,
    )
    return {
        "bases": res.get("bases", []),
        "cost": res.get("costs", {}).get("objective_total_bytes", float("inf")),
        "num_bases": len(res.get("bases", [])),
        "runtime": time.time() - t0,
    }


def run_ilp(
    problem: Problem,
    params: FacilityParams,
    model: MicroCompressionModel,
    ratio_matrix: Optional[np.ndarray] = None,
) -> Optional[Dict]:
    try:
        import gurobipy  # noqa: F401
    except Exception:
        return None
    t0 = time.time()
    res = ilp_facility_location_ratio_gurobi(
        tensor_ids=problem.ids,
        ratio_matrix=ratio_matrix if ratio_matrix is not None else problem.ratio_matrix,
        sizes=problem.sizes,
        base_meta_bytes=params.base_meta_bytes,
        base_storage_multiplier=params.base_storage_multiplier,
        verbose=params.verbose,
    )
    return {
        "bases": res.get("bases", []),
        "cost": res.get("costs", {}).get("objective_total_bytes", float("inf")),
        "num_bases": len(res.get("bases", [])),
        "runtime": time.time() - t0,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Benchmark facility solvers using cached ratios.")
    p.add_argument("--db-path", type=str, default="data/tensordb_s3", help="TensorDex storage directory")
    p.add_argument(
        "--metadata-db",
        type=str,
        default=None,
        help="Path to metadata.db (if not provided, defaults to <db-path>/metadata.db)",
    )
    p.add_argument("--results-db", type=str, default="results.db", help="SQLite with compression_results")
    p.add_argument(
        "--param-filter",
        type=str,
        default=None,
        help="Substring filter for param_name/shape (optional; layer0 pattern enforced)",
    )
    p.add_argument("--models-json", type=str, default="data/models/models.json", help="Path to models.json (for model filtering)")
    p.add_argument("--family", type=str, default="Qwen/Qwen2.5-7B", help="Model family to filter (matches cache gen script)")
    p.add_argument("--limit-models", type=int, default=500, help="Limit number of models (same logic as cache gen)")
    p.add_argument("--target-param", type=str, default=None, help="Restrict to a single param name (substring match)")
    p.add_argument(
        "--target-params",
        type=str,
        default=None,
        help="Comma-separated list of param names (substring match). Applied after --param-filter.",
    )
    p.add_argument("--max-items", type=int, default=None, help="Max tensors per param (truncates; None=all)")
    p.add_argument("--min-items", type=int, default=2, help="Skip params with fewer tensors")
    p.add_argument("--max-splits", type=int, default=2, help="Max additional bases in split heuristic")
    p.add_argument("--base-meta-bytes", type=int, default=0, help="Metadata overhead per opened base")
    p.add_argument("--methods", type=str, default="split,primal_dual,ilp", help="Comma list of methods to run")
    p.add_argument("--verbose", action="store_true", help="Verbose solver logs")
    p.add_argument("--output-csv", type=str, default=None, help="Optional CSV to write results")
    p.add_argument("--list-only", action="store_true", help="List matching params and exit (no solving)")
    p.add_argument("--family-filter", type=str, default="Qwen/Qwen2.5", help="Substring to match model_name")
    p.add_argument("--s3-bucket", type=str, default=None, help="Optional S3 bucket for TensorDex")
    p.add_argument("--s3-region", type=str, default="us-east-1", help="S3 region")
    p.add_argument("--s3-prefix", type=str, default="", help="S3 prefix")
    return p.parse_args()


def _normalize_target_params(target_param: Optional[str], target_params: Optional[str]) -> Optional[List[str]]:
    params: List[str] = []
    if target_param:
        params.append(target_param.strip())
    if target_params:
        params.extend([p.strip() for p in target_params.split(",") if p.strip()])
    return params or None


def _filter_cache_groups(
    cache_groups: Dict[str, List[CacheRow]], target_params: Optional[List[str]]
) -> Dict[str, List[CacheRow]]:
    if not target_params:
        return cache_groups
    # Substring match to mirror test_layer0_pairwise_compression.py usage.
    return {param: rows for param, rows in cache_groups.items() if any(tp in param for tp in target_params)}


def _is_layer0_param(param: str) -> bool:
    """Match layer 0 pattern as in test_layer0_pairwise_compression.py."""
    return "model.layers.0." in param or param.startswith("layers.0.")


def _filter_rows_like_test(
    cache_groups: Dict[str, List[CacheRow]],
    metadata_map: Dict[str, MetaRow],
    allowed_models: Optional[List[str]],
    target_param: Optional[str],
    param_filter: Optional[str],
    allowed_tensor_ids: Optional[set],
) -> Dict[str, List[CacheRow]]:
    """Match filtering behavior of tests/test_layer0_pairwise_compression.py."""
    model_set = set(allowed_models) if allowed_models else None
    filtered: Dict[str, List[CacheRow]] = {}

    for param, rows in cache_groups.items():
        if param_filter and param_filter not in param:
            continue
        if not _is_layer0_param(param):
            continue
        # Allow prefixed variants when target_param is provided (e.g.,
        # language_model.model.layers.0.* should match model.layers.0.*).
        canonical_param = param
        if target_param:
            if target_param not in param:
                continue
            if param.endswith(target_param):
                canonical_param = target_param

        kept_by_shape: Dict[Tuple, List[CacheRow]] = {}
        for r in rows:
            meta_t = metadata_map.get(r.target)
            meta_b = metadata_map.get(r.base)
            if not meta_t or not meta_b:
                continue
            if allowed_tensor_ids:
                if r.target not in allowed_tensor_ids or r.base not in allowed_tensor_ids:
                    continue
            if model_set:
                mt = meta_t.model_name
                mb = meta_b.model_name
                if mt not in model_set or mb not in model_set:
                    continue
            # Shape consistency check mirrors plan construction in test script
            if meta_t.shape != meta_b.shape:
                continue
            shape_key = tuple(meta_t.shape) if meta_t.shape else ()
            kept_by_shape.setdefault(shape_key, []).append(r)

        for shape_key, rows_for_shape in kept_by_shape.items():
            if not rows_for_shape:
                continue
            # Split groups per (param, shape) to mirror plan-time shape gating.
            key = f"{canonical_param}|shape={shape_key}" if shape_key else canonical_param
            filtered.setdefault(key, []).extend(rows_for_shape)

    return filtered


def _collect_allowed_tensor_ids(
    metadata_map: Dict[str, MetaRow],
    model_set: Optional[set],
    target_param: Optional[str],
    param_filter: Optional[str],
) -> set:
    """
    Derive the set of tensor_ids that would have been eligible in generate_layer0_plans:
    - model_name in model_set (if provided)
    - param matches layer0 pattern and optional target_param/param_filter
    """
    ids = set()
    for tid, meta in metadata_map.items():
        param = meta.param_name or ""
        if not _is_layer0_param(param):
            continue
        if target_param and target_param not in param:
            continue
        if param_filter and param_filter not in param:
            continue
        if model_set and meta.model_name not in model_set:
            continue
        ids.add(tid)
    return ids


def _load_allowed_models(models_json: str, family: str, limit_models: Optional[int]) -> Optional[List[str]]:
    """Replicate test_layer0_pairwise_compression model selection."""
    try:
        with open(models_json, "r") as f:
            data = json.load(f)
        if family not in data.get("base_models", {}):
            print(f"Family {family} not found in base_models; skipping model filter.")
            return None
        model_list = data["base_models"][family]
        if family not in model_list:
            model_list = [family] + model_list
        # Dedup while preserving order
        dedup = []
        seen = set()
        for m in model_list:
            if m in seen:
                continue
            seen.add(m)
            dedup.append(m)
        if limit_models:
            dedup = dedup[:limit_models]
        return dedup
    except Exception as e:
        print(f"Failed to load models.json ({models_json}): {e}. Proceeding without model filter.")
        return None


def _parse_shape(shape_val) -> Tuple[int, ...]:
    if shape_val is None:
        return tuple()
    if isinstance(shape_val, (list, tuple)):
        return tuple(int(x) for x in shape_val)
    try:
        import json
        parsed = json.loads(shape_val)
        if isinstance(parsed, list):
            return tuple(int(x) for x in parsed)
    except Exception:
        pass
    try:
        import ast
        parsed = ast.literal_eval(str(shape_val))
        if isinstance(parsed, (list, tuple)):
            return tuple(int(x) for x in parsed)
    except Exception:
        pass
    return tuple()


def load_metadata_sqlite(meta_db_path: Path, tensor_ids: Sequence[str]) -> Dict[str, MetaRow]:
    """Load minimal metadata from metadata.db without full TensorDex."""
    meta_map: Dict[str, MetaRow] = {}
    if not meta_db_path.exists():
        print(f"metadata.db not found at {meta_db_path}")
        return meta_map
    conn = sqlite3.connect(meta_db_path)
    conn.row_factory = sqlite3.Row

    # Expect the standard TensorDex layout: tensors + model_mappings.
    tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    missing = {"tensors", "model_mappings"} - tables
    if missing:
        print(f"metadata.db at {meta_db_path} is missing tables: {sorted(missing)}")
        conn.close()
        return meta_map

    # SQLite parameter limit ~999; chunk queries.
    ids_list = list(dict.fromkeys(tensor_ids))
    chunk = 900
    for i in range(0, len(ids_list), chunk):
        batch = ids_list[i : i + chunk]
        placeholders = ",".join("?" for _ in batch)
        query = f"""
            SELECT
                t.id AS tensor_id,
                t.shape AS shape,
                t.dtype AS dtype,
                mm.model_name AS model_name,
                mm.param_name AS param_name
            FROM tensors AS t
            LEFT JOIN model_mappings AS mm
                ON mm.tensor_id = t.id
            WHERE t.id IN ({placeholders})
        """
        for row in conn.execute(query, batch):
            tid = str(row["tensor_id"])
            if tid in meta_map:
                continue
            meta_map[tid] = MetaRow(
                tensor_id=tid,
                model_name=row["model_name"] if row["model_name"] is not None else None,
                param_name=row["param_name"] if row["param_name"] is not None else None,
                shape=_parse_shape(row["shape"]),
                dtype=str(row["dtype"]),
            )
    conn.close()
    return meta_map


def list_params_for_family(
    cache_groups: Dict[str, List[CacheRow]],
    metadata_map: Dict[str, MetaRow],
    family_filter: str,
    target_params: Optional[List[str]] = None,
):
    print(f"Listing params for family substring '{family_filter}'...")
    shown = 0
    filtered_groups = _filter_cache_groups(cache_groups, target_params)
    for param, rows in filtered_groups.items():
        ids = []
        for r in rows:
            ids.extend([r.target, r.base])
        ids = list(dict.fromkeys(ids))
        models = []
        for tid in ids:
            meta = metadata_map.get(tid)
            mname = meta.model_name if meta else None
            if mname and mname not in models:
                models.append(mname)
        if family_filter and not any(family_filter in m for m in models if m):
            continue
        shown += 1
        print(f"- {param}: tensors={len(ids)} models={len(models)}")


def main():
    args = parse_args()
    methods = {m.strip().lower() for m in args.methods.split(",") if m.strip()}
    t_start = time.time()
    print("Stage: init args parsed")

    t_models = time.time()
    allowed_models = _load_allowed_models(args.models_json, args.family, args.limit_models)
    print(f"Stage done: models filter loaded in {time.time() - t_models:.2f}s")

    model = MicroCompressionModel()
    params = FacilityParams(verbose=args.verbose, candidate_reduction=False, base_meta_bytes=args.base_meta_bytes)

    t_cache = time.time()
    cache_groups = load_cached_rows(Path(args.results_db), args.param_filter, args.target_param)
    print(f"Stage done: loaded cache rows for {len(cache_groups)} params in {time.time() - t_cache:.2f}s")

    # Load minimal metadata via sqlite (shape/dtype/model_name) for involved tensor_ids
    t_meta = time.time()
    all_ids: List[str] = []
    for rows in cache_groups.values():
        for r in rows:
            all_ids.extend([r.target, r.base])
    metadata_db_path = Path(args.metadata_db) if args.metadata_db else Path(args.db_path) / "metadata.db"
    metadata_map = load_metadata_sqlite(metadata_db_path, all_ids)
    print(f"Stage done: loaded metadata for {len(metadata_map)} tensors in {time.time() - t_meta:.2f}s")

    t_filter = time.time()
    allowed_tensor_ids = _collect_allowed_tensor_ids(
        metadata_map,
        set(allowed_models) if allowed_models else None,
        args.target_param,
        args.param_filter,
    )
    # Align filtering with test_layer0_pairwise_compression.py: layer0 pattern,
    # optional target_param substring, model set restriction, and shape match.
    cache_groups = _filter_rows_like_test(
        cache_groups,
        metadata_map,
        allowed_models,
        args.target_param,
        args.param_filter,
        allowed_tensor_ids,
    )
    print(f"Stage done: filtered to {len(cache_groups)} params in {time.time() - t_filter:.2f}s")
    if args.verbose:
        print(f"Filtered cache groups to {len(cache_groups)} params using test-like rules")
    if not cache_groups:
        print("No cached rows found with the given filter.")
        return

    if args.list_only:
        list_params_for_family(cache_groups, metadata_map, args.family_filter, None)
        return

    results = []
    total_params = len(cache_groups)
    t_solve_all = time.time()
    for idx, (param, rows) in enumerate(cache_groups.items()):
        t_build = time.time()
        problem, reason = build_problem(
            metadata_map,
            rows,
            max_items=args.max_items,
            min_items=args.min_items,
        )
        build_dt = time.time() - t_build
        if problem is None:
            if args.verbose:
                print(f"Skipping {param}: {reason}")
            continue

        print(f"\n[{idx + 1}/{total_params}] Param: {param} (build {build_dt:.2f}s)")
        total_in = float(problem.sizes.sum())
        ratio_mat = problem.ratio_matrix
        cover = problem.ratio_overrides / max(1, len(problem.ids) * (len(problem.ids) - 1))

        # Heuristic split
        split_res = None
        if "split" in methods:
            t0 = time.time()
            split_res = heuristic_split(
                problem, args.base_meta_bytes, args.max_splits, ratio_matrix_override=ratio_mat
            )
            split_res["runtime"] = time.time() - t0
            split_res["ratio"] = (
                split_res["cost"] / total_in if total_in > 0 and math.isfinite(split_res["cost"]) else None
            )

        # Primal-dual
        pd_res = None
        if "primal_dual" in methods:
            pd_res = run_primal_dual(problem, params, model, ratio_matrix=ratio_mat)
            pd_res["ratio"] = pd_res["cost"] / total_in if total_in > 0 else None

        # ILP
        ilp_res = None
        if "ilp" in methods:
            ilp_res = run_ilp(problem, params, model, ratio_matrix=ratio_mat)
            if ilp_res:
                ilp_res["ratio"] = ilp_res["cost"] / total_in if total_in > 0 else None

        res_row = {
            "param": param,
            "n": len(problem.ids),
            "cached_pairs": len(problem.cached_pairs),
            "override_cov": cover,
            "split_ratio": split_res["ratio"] if split_res else None,
            "split_bases": split_res["num_bases"] if split_res else None,
            "split_time": split_res["runtime"] if split_res else None,
            "pd_ratio": pd_res["ratio"] if pd_res else None,
            "pd_bases": pd_res["num_bases"] if pd_res else None,
            "pd_time": pd_res["runtime"] if pd_res else None,
            "ilp_ratio": ilp_res["ratio"] if ilp_res else None,
            "ilp_bases": ilp_res["num_bases"] if ilp_res else None,
            "ilp_time": ilp_res["runtime"] if ilp_res else None,
        }
        results.append(res_row)

        print(f"\nParam: {param}")
        print(f"  tensors={res_row['n']} cached_pairs={res_row['cached_pairs']} override_cov={cover:.2%}")
        if split_res:
            split_ratio = res_row["split_ratio"]
            if split_ratio is None:
                msg = split_res.get("heuristic_msg", "unreachable (missing ratios)")
                print(f"  split   ratio=nan bases={res_row['split_bases']} time={res_row['split_time']:.3f}s note={msg}")
            else:
                print(f"  split   ratio={split_ratio:.4f} bases={res_row['split_bases']} time={res_row['split_time']:.3f}s")
        if pd_res:
            pd_ratio = res_row["pd_ratio"]
            if pd_ratio is None:
                print(f"  pd      ratio=nan bases={res_row['pd_bases']} time={res_row['pd_time']:.3f}s")
            else:
                print(f"  pd      ratio={pd_ratio:.4f} bases={res_row['pd_bases']} time={res_row['pd_time']:.3f}s")
        if ilp_res:
            ilp_ratio = res_row["ilp_ratio"]
            if ilp_ratio is None:
                print(f"  ilp     ratio=nan bases={res_row['ilp_bases']} time={res_row['ilp_time']:.3f}s")
            else:
                print(f"  ilp     ratio={ilp_ratio:.4f} bases={res_row['ilp_bases']} time={res_row['ilp_time']:.3f}s")
        elif "ilp" in methods:
            print("  ilp     skipped (gurobipy not available)")

    if args.output_csv and results:
        import csv

        out_path = Path(args.output_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cols = list(results[0].keys())
        with out_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=cols)
            writer.writeheader()
            writer.writerows(results)
        print(f"\nWrote summary to {out_path}")


if __name__ == "__main__":
    main()
