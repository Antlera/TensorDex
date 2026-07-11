#!/usr/bin/env python3
"""
Primal-dual facility location solver that works directly on a ratio matrix.

Inputs:
- tensor_ids: list of ids (for bookkeeping; can be simple indices/strings)
- sizes: 1D array of tensor sizes (bytes)
- ratio_matrix: NxN matrix where ratio[i, j] * sizes[i] = cost to connect i -> base j

This mirrors the event-based primal-dual logic in algorithms/primal_dual_facility_location.py
but removes distance normalization and computes feasibility purely from ratio_matrix finiteness.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np


def primal_dual_facility_location_ratio(
    tensor_ids: Sequence[str],
    ratio_matrix: np.ndarray,
    sizes: Sequence[float],
    *,
    base_meta_bytes: float = 0.0,
    base_storage_multiplier: float = 1.0,
    candidate_indices: Optional[Sequence[int]] = None,
    verbose: bool = False,
) -> Dict:
    """
    Solve facility location using a provided ratio matrix.

    Returns a dict with bases, assignment, costs, and timing metadata.
    """
    total_start = time.perf_counter()
    tensor_ids = list(tensor_ids)
    sizes = np.asarray(sizes, dtype=np.float64)
    ratio_full = np.asarray(ratio_matrix, dtype=np.float64)
    n = len(sizes)

    if ratio_full.shape != (n, n):
        raise ValueError(f"ratio_matrix shape {ratio_full.shape} does not match sizes length {n}")
    if len(tensor_ids) != n:
        raise ValueError(f"tensor_ids length {len(tensor_ids)} must match sizes length {n}")

    # Connection costs derived directly from ratio matrix
    ratio_full = ratio_full.copy()
    np.fill_diagonal(ratio_full, 0.0)
    connection_cost = ratio_full * sizes[:, None]
    feasible = np.isfinite(connection_cost)
    connection_cost[~feasible] = np.inf
    np.fill_diagonal(connection_cost, 0.0)

    base_open_cost = base_storage_multiplier * (sizes + base_meta_bytes)

    allowed_bases = (
        sorted({int(idx) for idx in candidate_indices if 0 <= int(idx) < n})
        if candidate_indices is not None
        else list(range(n))
    )
    assignment = np.full(n, -1, dtype=int)
    open_bases: Set[int] = set()
    unassigned: Set[int] = set(range(n))

    allowed_array = np.asarray(allowed_bases, dtype=int)
    if allowed_array.size > 0:
        feasible_candidates = np.isfinite(connection_cost[:, allowed_array])
        has_candidate = feasible_candidates.any(axis=1)
    else:
        has_candidate = np.zeros(n, dtype=bool)

    forced_indices = np.where(~has_candidate)[0].tolist()
    for idx in forced_indices:
        assignment[idx] = idx
        open_bases.add(int(idx))
        unassigned.discard(int(idx))

    if verbose and forced_indices:
        print(f"[pd-ratio] forced {len(forced_indices)} self-bases (no finite ratio to candidates)")

    # Precompute candidate edges sorted by cost
    facility_edges: Dict[int, List[Tuple[float, int]]] = {}
    for base_idx in allowed_bases:
        if base_idx in open_bases:
            continue
        column = connection_cost[:, base_idx]
        finite_mask = np.isfinite(column)
        if not np.any(finite_mask):
            continue
        clients = np.nonzero(finite_mask)[0]
        edges = [(float(column[i]), int(i)) for i in clients]
        edges.sort(key=lambda pair: (pair[0], pair[1]))
        facility_edges[base_idx] = edges

    def compute_event(base_idx: int) -> Optional[Tuple[float, List[int]]]:
        edges = facility_edges.get(base_idx)
        if not edges or not unassigned:
            return None
        contributing: List[int] = []
        sum_costs = 0.0
        for pos, (cost, client_idx) in enumerate(edges):
            if client_idx not in unassigned:
                continue
            contributing.append(client_idx)
            sum_costs += cost
            count = len(contributing)
            if count == 0:
                continue
            required = (float(base_open_cost[base_idx]) + sum_costs) / count
            current_cost = cost
            t_candidate = float(max(current_cost, required))
            next_cost = float("inf")
            next_pos = pos + 1
            while next_pos < len(edges):
                nxt_cost, nxt_client = edges[next_pos]
                if nxt_client in unassigned:
                    next_cost = nxt_cost
                    break
                next_pos += 1
            if t_candidate <= next_cost:
                return t_candidate, list(contributing)
        return None

    remaining_bases = [b for b in allowed_bases if b not in open_bases]
    while unassigned and remaining_bases:
        best_time = float("inf")
        best_base: Optional[int] = None
        best_clients: Optional[List[int]] = None
        for base_idx in remaining_bases:
            event = compute_event(base_idx)
            if event is None:
                continue
            t_candidate, clients = event
            if not clients:
                continue
            if t_candidate < best_time:
                best_time = t_candidate
                best_base = base_idx
                best_clients = clients
        if best_base is None:
            break
        open_bases.add(best_base)
        if best_clients:
            for client_idx in best_clients:
                if client_idx in unassigned:
                    assignment[client_idx] = best_base
                    unassigned.discard(client_idx)
        remaining_bases = [b for b in remaining_bases if b != best_base]

    if unassigned and open_bases:
        open_sorted = sorted(open_bases)
        open_arr = np.asarray(open_sorted, dtype=int)
        unassigned_list = sorted(unassigned)
        if open_arr.size > 0 and unassigned_list:
            subset = connection_cost[np.ix_(unassigned_list, open_arr)]
            best_idx = np.argmin(subset, axis=1)
            best_costs = subset[np.arange(len(unassigned_list)), best_idx]
            for offset, client_idx in enumerate(unassigned_list):
                cost = best_costs[offset]
                if np.isfinite(cost):
                    assignment[client_idx] = int(open_arr[best_idx[offset]])
                    unassigned.discard(client_idx)

    if unassigned:
        for client_idx in list(unassigned):
            assignment[client_idx] = client_idx
            open_bases.add(client_idx)
            unassigned.discard(client_idx)

    bases = sorted(open_bases)
    assignment_dict: Dict[int, int] = {int(i): int(j) for i, j in enumerate(assignment)}

    sum_Rd_si = 0.0
    for i, j in assignment_dict.items():
        if i == j:
            continue
        cost = float(connection_cost[i, j])
        if not np.isfinite(cost):
            continue
        sum_Rd_si += cost

    base_storage_total = float(base_open_cost[bases].sum()) if bases else 0.0
    objective_total = base_storage_total + float(sum_Rd_si)
    runtime_sec = time.perf_counter() - total_start

    if verbose:
        print(
            "[pd-ratio] bases={} objective={:.3f} runtime={:.3f}s forced_self_bases={}".format(
                len(bases), objective_total, runtime_sec, len(forced_indices)
            )
        )

    return {
        "bases": bases,
        "assignment": assignment_dict,
        "costs": {
            "objective_total_bytes": float(objective_total),
            "base_storage_total_bytes": float(base_storage_total),
            "sum_Rd_si_bytes": float(sum_Rd_si),
        },
        "meta": {
            "num_targets": n,
            "num_bases": len(bases),
            "runtime_sec": runtime_sec,
            "candidate_indices": allowed_bases,
            "forced_self_bases": forced_indices,
        },
    }


__all__ = ["primal_dual_facility_location_ratio"]

