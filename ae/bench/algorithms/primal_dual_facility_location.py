"""
Primal-Dual facility location solver for TensorDex clustering and compression.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np

from .micro_algorithms import (
    FacilityParams,
    MicroCompressionModel,
    compute_ratio_matrix_from_D,
    normalize_distance_matrix_by_size,
    select_candidate_indices,
)


def primal_dual_facility_location(
    tensor_ids: Sequence[str],
    D: np.ndarray,
    sizes: np.ndarray,
    params: FacilityParams,
    model: MicroCompressionModel,
    *,
    tensor_shapes: Optional[Mapping[str, Sequence[int]]] = None,
    tensor_num_elements: Optional[Mapping[str, int]] = None,
    per_item_num_elements: Optional[Sequence[Optional[int]]] = None,
    ratio_override: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Primal-dual facility location solver aligned with ILP semantics.

    Bases open when accumulated client dual contributions cover the base-open
    cost. Clients connect only along edges with D[i,j] <= d_max, and self-bases
    are forced when no feasible candidate exists.

    Distances are normalized in-place by sqrt(#elements) when metadata is
    provided through `tensor_ids` + (`tensor_shapes` | `tensor_num_elements` |
    `per_item_num_elements`). Missing metadata leaves those rows/columns raw.
    """
    if model is None:
        raise ValueError("MicroCompressionModel is required for primal-dual solver.")

    total_start = time.perf_counter()
    tensor_ids = list(tensor_ids)
    D = np.asarray(D, dtype=np.float64)
    sizes = np.asarray(sizes, dtype=np.float64)
    n = len(sizes)

    if len(tensor_ids) != n:
        raise ValueError(
            f"tensor_ids length {len(tensor_ids)} must match sizes length {n}"
        )

    if D.shape != (n, n):
        raise ValueError(f"Distance matrix shape {D.shape} does not match sizes length {n}")

    D = normalize_distance_matrix_by_size(
        tensor_ids,
        D,
        tensor_shapes=tensor_shapes,
        tensor_num_elements=tensor_num_elements,
        per_item_num_elements=per_item_num_elements,
    )

    if n == 0:
        return {
            "bases": [],
            "assignment": {},
            "costs": {
                "objective_total_bytes": 0.0,
                "base_storage_total_bytes": 0.0,
                "sum_Rd_si_bytes": 0.0,
            },
            "meta": {
                "num_targets": 0,
                "num_bases": 0,
                "runtime_sec": 0.0,
                "candidate_selection_sec": 0.0,
                "pd_solve_sec": 0.0,
                "candidate_indices": [],
                "num_candidates": 0,
            },
        }

    if params.verbose:
        print(f"[primal-dual] n={n}, candidate_reduction={params.candidate_reduction}")

    candidate_selection_start = time.perf_counter()
    if params.candidate_reduction and n > params.small_ilp_cutoff:
        if params.candidate_topk is not None:
            k = min(params.candidate_topk, n)
        else:
            k = min(params.max_ilp_items, n)
        candidate_indices = select_candidate_indices(D, sizes, params, model, k)
        if params.verbose:
            print(f"[primal-dual] candidate reduction enabled: {len(candidate_indices)} / {n}")
    else:
        candidate_indices = list(range(n))
        if params.verbose:
            if params.candidate_reduction:
                print(f"[primal-dual] candidate reduction skipped (n={n} <= cutoff={params.small_ilp_cutoff})")
            else:
                print("[primal-dual] candidate reduction disabled via params.")
    candidate_selection_sec = time.perf_counter() - candidate_selection_start

    pd_solve_start = time.perf_counter()

    if ratio_override is not None:
        ratio_full = np.asarray(ratio_override, dtype=np.float64)
        if ratio_full.shape != (n, n):
            raise ValueError(
                f"ratio_override shape {ratio_full.shape} does not match sizes length {n}"
            )
        ratio_full = ratio_full.copy()
        np.fill_diagonal(ratio_full, 0.0)
        feasible = np.isfinite(ratio_full)
        if params.d_max is not None:
            feasible &= np.isfinite(D) & (D <= params.d_max)
    else:
        ratio_full = compute_ratio_matrix_from_D(D, model, params.d_max)
        feasible = np.isfinite(D) & (D <= params.d_max)
    np.fill_diagonal(feasible, True)
    connection_cost = np.asarray(ratio_full * sizes[:, None], dtype=np.float64)
    connection_cost[~feasible] = np.inf
    np.fill_diagonal(connection_cost, 0.0)

    base_open_cost = params.base_storage_multiplier * (sizes + params.base_meta_bytes)

    allowed_bases = sorted({int(idx) for idx in candidate_indices if 0 <= int(idx) < n})
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

    if params.verbose and forced_indices:
        print(f"[primal-dual] forced {len(forced_indices)} self-bases due to d_max feasibility.")

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

    pd_solve_sec = time.perf_counter() - pd_solve_start
    runtime_sec = time.perf_counter() - total_start

    if params.verbose:
        print(
            "[primal-dual] timing: candidates={:.3f}s pd_solve={:.3f}s total={:.3f}s | "
            "opened {} bases, objective {:.3f}".format(
                candidate_selection_sec,
                pd_solve_sec,
                runtime_sec,
                len(bases),
                objective_total,
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
            "candidate_selection_sec": candidate_selection_sec,
            "pd_solve_sec": pd_solve_sec,
            "candidate_indices": candidate_indices,
            "num_candidates": len(candidate_indices),
        },
    }

