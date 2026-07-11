"""
Greedy facility location algorithm for TensorDex clustering and compression.
"""

from __future__ import annotations

import heapq
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np

try:
    from numba import njit, prange
    NUMBA_AVAILABLE = True
    # print("Numba available") # Optional: avoid spamming prints on import
except ImportError:
    # Fallback decorators if numba is not installed
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    def prange(*args):
        return range(*args)
    NUMBA_AVAILABLE = False
    # print("Numba not available")

from .micro_algorithms import (
    FacilityParams,
    MicroCompressionModel,
    compute_ratio_matrix_from_D,
    normalize_distance_matrix_by_size,
    select_candidate_indices,
)


# ============================================================================
# Utility functions for Greedy
# ============================================================================

def _compute_objective_and_assignment(
    RdSi_full: np.ndarray,
    base_open_cost: np.ndarray,
    bases: np.ndarray,
) -> Tuple[np.ndarray, float]:
    """
    Compute total objective and assignments for a given base set.
    """
    n = RdSi_full.shape[0]
    bases_arr = np.asarray(bases, dtype=int)
    assignment = np.arange(n, dtype=int)

    if bases_arr.size == 0:
        # No bases open ⇒ every item stores itself and pays full base cost.
        return assignment, float(base_open_cost.sum())

    bases_arr = np.unique(bases_arr)
    base_mask = np.zeros(n, dtype=bool)
    base_mask[bases_arr] = True
    assignment[bases_arr] = bases_arr

    non_base_idx = np.nonzero(~base_mask)[0]
    if non_base_idx.size > 0:
        non_base_costs = RdSi_full[np.ix_(non_base_idx, bases_arr)]
        closest_idx = np.argmin(non_base_costs, axis=1)
        assignment[non_base_idx] = bases_arr[closest_idx]

        conn_cost = float(
            RdSi_full[non_base_idx, assignment[non_base_idx]].sum()
        )
    else:
        conn_cost = 0.0

    base_cost = float(base_open_cost[bases_arr].sum())
    return assignment, base_cost + conn_cost


def _select_local_candidates(
    cluster_indices: np.ndarray,
    candidate_array: np.ndarray,
    RdSi_full: np.ndarray,
    max_candidates: int,
) -> np.ndarray:
    """
    Select promising replacement bases for a cluster using mean connection cost.
    """
    if cluster_indices.size == 0 or candidate_array.size == 0:
        return np.empty(0, dtype=int)

    cluster_costs = RdSi_full[np.ix_(cluster_indices, candidate_array)]
    mean_cost = cluster_costs.mean(axis=0)

    if max_candidates is not None and max_candidates > 0 and candidate_array.size > max_candidates:
        idx = np.argpartition(mean_cost, kth=max_candidates - 1)[:max_candidates]
        idx = idx[np.argsort(mean_cost[idx])]
    else:
        idx = np.argsort(mean_cost)

    return candidate_array[idx]


def _run_local_swap_refinement(
    RdSi_full: np.ndarray,
    base_open_cost: np.ndarray,
    candidate_indices: List[int],
    initial_bases: np.ndarray,
    params: FacilityParams,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    """
    Perform Teitz–Bart style local search with ADD/DROP/SWAP moves using
    Numba-optimized kernels.
    """
    n = RdSi_full.shape[0]
    
    # Setup bases
    bases_arr = np.asarray(initial_bases, dtype=np.int32)
    bases_arr = np.unique(bases_arr)
    bases_arr.sort()
    
    # Candidates
    candidates_arr = np.array(sorted(list(set(candidate_indices))), dtype=np.int32)
    
    # Ensure float64 for costs
    base_open_cost = np.asarray(base_open_cost, dtype=np.float64)
    
    stats = {"swaps": 0, "iterations": 0}
    max_swaps = params.local_search_max_swaps
    eps = params.local_search_eps
    
    for _ in range(max_swaps):
        stats["iterations"] += 1
        
        # 1. Compute current state (assignments, costs)
        current_costs, second_costs, assignments = _compute_state_numba(
            RdSi_full, bases_arr, n
        )
        
        # 2. Compute DROP deltas
        drop_deltas = _compute_drop_deltas_numba(
            current_costs, second_costs, assignments, bases_arr, base_open_cost
        )
        
        # Find best DROP
        best_delta = 0.0
        best_action_type = 0 # 0: None, 1: ADD, 2: SWAP, 3: DROP
        best_cand_idx = -1
        best_base_idx_in_arr = -1
        
        if len(bases_arr) > 0:
            min_drop_idx = np.argmin(drop_deltas)
            min_drop_val = drop_deltas[min_drop_idx]
            if min_drop_val < -eps:
                best_delta = min_drop_val
                best_action_type = 3
                best_base_idx_in_arr = min_drop_idx
        
        # 3. Compute ADD/SWAP deltas
        if len(candidates_arr) > 0:
            c_deltas, c_types, c_params = _find_best_swap_numba(
                RdSi_full,
                current_costs,
                second_costs,
                assignments,
                bases_arr,
                candidates_arr,
                base_open_cost
            )
            
            min_c_idx = np.argmin(c_deltas)
            min_c_val = c_deltas[min_c_idx]
            
            if min_c_val < best_delta - eps:
                best_delta = min_c_val
                best_action_type = c_types[min_c_idx]
                best_cand_idx = min_c_idx
                best_base_idx_in_arr = c_params[min_c_idx]

        if best_action_type == 0:
            break
            
        # Apply move
        stats["swaps"] += 1
        
        if best_action_type == 3: # DROP
            bases_arr = np.delete(bases_arr, best_base_idx_in_arr)
            
        elif best_action_type == 1: # ADD
            cand = candidates_arr[best_cand_idx]
            bases_arr = np.append(bases_arr, cand)
            bases_arr.sort()
            
        elif best_action_type == 2: # SWAP
            cand = candidates_arr[best_cand_idx]
            # Replace
            bases_arr[best_base_idx_in_arr] = cand
            bases_arr.sort()
            
    # Final assignment
    assignment, _ = _compute_objective_and_assignment(
        RdSi_full, base_open_cost, bases_arr
    )
    
    return assignment, bases_arr, stats


# ============================================================================
# Numba Helpers for Greedy Algorithm
# ============================================================================

@njit(parallel=True, fastmath=True)
def _compute_gains_batch_numba(
    RdSi_full: np.ndarray,
    current_costs: np.ndarray,
    candidates: np.ndarray,
    base_open_cost: np.ndarray,
) -> np.ndarray:
    """
    Compute initial marginal gains for a batch of candidates in parallel.
    Uses O(1) memory per thread instead of broadcasting.
    """
    n_cands = len(candidates)
    n_items = RdSi_full.shape[0]
    gains = np.zeros(n_cands, dtype=np.float64)
    
    for i in prange(n_cands):
        c_idx = candidates[i]
        base_cost_c = base_open_cost[c_idx]
        gain = 0.0
        
        for j in range(n_items):
            # Enforce base self-storage cost logic
            if j == c_idx:
                cost_j = base_cost_c
            else:
                cost_j = RdSi_full[j, c_idx]
            
            curr = current_costs[j]
            # equivalent to max(curr - cost_j, 0)
            if cost_j < curr:
                gain += (curr - cost_j)
        
        gains[i] = gain
    return gains


@njit(parallel=True, fastmath=True)
def _compute_state_numba(RdSi_full, bases_indices, n_items):
    """
    Compute current_costs, second_costs, and assignment indices for all items.
    """
    n_bases = len(bases_indices)
    
    # Pre-allocate output arrays
    current_costs = np.empty(n_items, dtype=np.float64)
    second_costs = np.empty(n_items, dtype=np.float64)
    assignments = np.empty(n_items, dtype=np.int32)
    
    # Large finite number for infinity to avoid NaNs
    INF_VAL = 1e30 
    
    for i in prange(n_items):
        min1 = INF_VAL
        min2 = INF_VAL
        idx1 = -1
        
        for k in range(n_bases):
            base_id = bases_indices[k]
            # Cost for item i to connect to base_id
            if i == base_id:
                cost = 0.0
            else:
                cost = RdSi_full[i, base_id]
            
            # Handle NaNs or Infs in RdSi_full
            if not np.isfinite(cost):
                cost = INF_VAL
            
            if cost < min1:
                min2 = min1
                min1 = cost
                idx1 = k
            elif cost < min2:
                min2 = cost
        
        current_costs[i] = min1
        second_costs[i] = min2
        assignments[i] = idx1
        
    return current_costs, second_costs, assignments


@njit(parallel=True, fastmath=True)
def _compute_drop_deltas_numba(current_costs, second_costs, assignments, bases, base_open_cost):
    """
    Compute deltas for DROP moves (removing a base).
    """
    n_bases = len(bases)
    n_items = len(current_costs)
    
    drop_deltas = np.zeros(n_bases, dtype=np.float64)
    
    # Parallelize over bases (O(K*N))
    for k in prange(n_bases):
        base_id = bases[k]
        penalty_sum = 0.0
        for i in range(n_items):
            if assignments[i] == k:
                penalty_sum += (second_costs[i] - current_costs[i])
        
        drop_deltas[k] = -base_open_cost[base_id] + penalty_sum
        
    return drop_deltas


@njit(parallel=True, fastmath=True)
def _find_best_swap_numba(
    RdSi_full,
    current_costs,
    second_costs,
    assignments,
    bases,
    candidates,
    base_open_cost
):
    """
    Find best ADD or SWAP move for each candidate in parallel.
    Returns best_deltas, best_types, best_params arrays (size n_candidates).
    """
    n_candidates = len(candidates)
    n_bases = len(bases)
    n_items = len(current_costs)
    INF_VAL = 1e30
    
    # Store best result for each candidate
    # best_deltas[c_idx]
    # best_types[c_idx]: 1=ADD, 2=SWAP
    # best_params[c_idx]: base_idx_to_remove (for SWAP, index in bases array)
    c_best_deltas = np.full(n_candidates, INF_VAL, dtype=np.float64)
    c_best_types = np.zeros(n_candidates, dtype=np.int32)
    c_best_params = np.full(n_candidates, -1, dtype=np.int32)
    
    for c_idx in prange(n_candidates):
        cand_id = candidates[c_idx]
        base_cost_c = base_open_cost[cand_id]
        
        # 1. Calculate ADD Delta
        # Gain from items switching to c
        gain_add = 0.0
        
        # Accumulate correction terms for SWAP
        # swap_corrections[k] stores the extra cost if base k is removed
        # (relative to just adding c)
        swap_corrections = np.zeros(n_bases, dtype=np.float64)
        
        for i in range(n_items):
            # Cost if connected to c
            if i == cand_id:
                cost_c = 0.0
            else:
                cost_c = RdSi_full[i, cand_id]
            
            if not np.isfinite(cost_c):
                cost_c = INF_VAL
                
            curr = current_costs[i]
            
            # Add Gain
            # If cost_c < curr, i switches to c. Gain is cost_c - curr.
            if cost_c < curr:
                gain_val = cost_c - curr
                gain_add += gain_val
                
                # SWAP Correction:
                # If i was assigned to k, and k is removed:
                # i moves to c (since cost_c < curr <= second).
                # Cost change is cost_c - curr.
                # This exactly matches gain_val.
                # So correction is 0.
            else:
                # cost_c >= curr. i stays with curr (unless curr is removed).
                # gain_add contribution is 0.
                
                # SWAP Correction:
                # If i was assigned to k, and k is removed:
                k = assignments[i]
                if k != -1:
                    # i moves to min(cost_c, second).
                    sec = second_costs[i]
                    fallback = min(cost_c, sec)
                    
                    # Cost change: fallback - curr
                    correction = fallback - curr
                    swap_corrections[k] += correction
        
        # Total ADD Delta
        add_delta = base_cost_c + gain_add
        
        # Track best for this candidate
        best_d = add_delta
        best_t = 1 # ADD
        best_p = -1
        
        # Check SWAP Deltas
        # SWAP(k -> c) Delta = AddDelta - BaseCost(k) + SwapCorrection(k)
        for k in range(n_bases):
            base_id = bases[k]
            swap_delta = add_delta - base_open_cost[base_id] + swap_corrections[k]
            
            if swap_delta < best_d:
                best_d = swap_delta
                best_t = 2 # SWAP
                best_p = k # Index in bases array
        
        c_best_deltas[c_idx] = best_d
        c_best_types[c_idx] = best_t
        c_best_params[c_idx] = best_p
        
    return c_best_deltas, c_best_types, c_best_params


@njit(fastmath=True)
def _compute_marginal_gain_numba(
    RdSi_col: np.ndarray,
    current_costs: np.ndarray,
    cand_idx: int,
    base_cost_c: float,
) -> float:
    """
    Compute marginal gain for a single candidate against current state.
    """
    n_items = len(current_costs)
    gain = 0.0
    for j in range(n_items):
        if j == cand_idx:
            cost_j = base_cost_c
        else:
            cost_j = RdSi_col[j]
        
        curr = current_costs[j]
        if cost_j < curr:
            gain += (curr - cost_j)
            
    return gain


@njit(fastmath=True)
def _update_state_numba(
    RdSi_col: np.ndarray,
    current_costs: np.ndarray,
    assignment: np.ndarray,
    cand_idx: int,
    base_cost_c: float,
):
    """
    Update costs and assignment in-place for the selected candidate.
    """
    n_items = len(current_costs)
    for j in range(n_items):
        if j == cand_idx:
            # Force base to store itself
            current_costs[j] = base_cost_c
            assignment[j] = cand_idx
            continue
            
        cost_j = RdSi_col[j]
        if cost_j < current_costs[j]:
            current_costs[j] = cost_j
            assignment[j] = cand_idx


def greedy_facility_location(
    tensor_ids: Sequence[str],
    D: np.ndarray,
    sizes: np.ndarray,
    params: FacilityParams,
    model: MicroCompressionModel,
    *,
    tensor_shapes: Optional[Mapping[str, Sequence[int]]] = None,
    tensor_num_elements: Optional[Mapping[str, int]] = None,
    per_item_num_elements: Optional[Sequence[Optional[int]]] = None,
    pre_open_bases: Optional[Sequence[int]] = None,
    external_min_costs: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Multi-base greedy facility location aligned with ILP objective semantics.
    Distances are normalized internally via sqrt(#elements) whenever per-item
    shape metadata (or precomputed element counts) is provided.

    Optimized with Lazy Greedy (CELF) and Numba parallelization.

    Args:
        tensor_ids: Ordered IDs that align with rows/columns of `D`.
        D: Raw pairwise distance matrix.
        sizes: Tensor sizes in bytes (same ordering as tensor_ids).
        params: Facility-location hyperparameters.
        model: Compression model used to convert distances to ratios.
        tensor_shapes: Optional id→shape metadata for distance normalization.
        tensor_num_elements: Optional id→numel metadata (overrides shapes).
        per_item_num_elements: Optional iterable of numel counts when callers
            already computed them while building `sizes`.
        pre_open_bases: Optional indices that correspond to already-open bases.
    """
    tensor_ids = list(tensor_ids)
    n = len(sizes)
    if len(tensor_ids) != n:
        raise ValueError(
            f"tensor_ids length {len(tensor_ids)} must match sizes length {n}"
        )
    if D.shape != (n, n):
        raise ValueError(f"Distance matrix shape {D.shape} does not match n={n}")
    sizes = np.asarray(sizes, dtype=np.float64)

    D = normalize_distance_matrix_by_size(
        tensor_ids,
        D,
        tensor_shapes=tensor_shapes,
        tensor_num_elements=tensor_num_elements,
        per_item_num_elements=per_item_num_elements,
    )

    total_start = time.perf_counter()

    # ------------------------------------------------------------------
    # Candidate selection (reuse ILP reduction logic)
    # ------------------------------------------------------------------
    candidate_selection_start = time.perf_counter()
    if params.candidate_reduction and n > params.small_ilp_cutoff:
        if params.candidate_topk is not None:
            k = min(params.candidate_topk, n)
        else:
            k = min(params.max_ilp_items, n)
        candidate_indices = select_candidate_indices(D, sizes, params, model, k)
    else:
        candidate_indices = list(range(n))
    candidate_selection_sec = time.perf_counter() - candidate_selection_start

    # ------------------------------------------------------------------
    # Ratio / residual matrix with d_max feasibility
    # ------------------------------------------------------------------
    ratio_build_start = time.perf_counter()
    ratio_full = compute_ratio_matrix_from_D(D, model, params.d_max)
    feasible = np.isfinite(D) & (D <= params.d_max)
    np.fill_diagonal(feasible, True)
    RdSi_full = np.asarray(ratio_full * sizes[:, None], dtype=np.float64)
    RdSi_full[~feasible] = np.inf
    ratio_build_sec = time.perf_counter() - ratio_build_start

    alpha = params.base_storage_multiplier
    meta = params.base_meta_bytes
    base_open_cost = alpha * (sizes + meta)
    pre_open_set: Set[int] = {
        int(idx)
        for idx in (pre_open_bases or [])
        if 0 <= int(idx) < n
    }
    pre_open_arr = (
        np.asarray(sorted(pre_open_set), dtype=int)
        if pre_open_set
        else np.empty(0, dtype=int)
    )
    if pre_open_arr.size > 0:
        base_open_cost = np.asarray(base_open_cost, dtype=np.float64)
        base_open_cost[pre_open_arr] = 0.0

    # ------------------------------------------------------------------
    # Greedy marginal-gain loop (Lazy Greedy / CELF)
    # ------------------------------------------------------------------
    assignment = np.arange(n, dtype=int)  # all items self-store initially
    current_costs = base_open_cost.copy()

    if external_min_costs is not None:
        if external_min_costs.shape != current_costs.shape:
            raise ValueError(
                f"external_min_costs shape {external_min_costs.shape} must match n={n}"
            )
        current_costs = np.minimum(current_costs, external_min_costs)

    # Apply pre-open bases
    if pre_open_arr.size > 0 and n > 0:
        pre_open_cols = np.asarray(RdSi_full[:, pre_open_arr], dtype=np.float64)
        if np.isnan(pre_open_cols).any():
            np.nan_to_num(pre_open_cols, nan=np.inf, copy=False)
        best_idx = np.argmin(pre_open_cols, axis=1)
        best_costs = pre_open_cols[np.arange(n), best_idx]
        improved_mask = np.isfinite(best_costs) & (best_costs < current_costs)
        if np.any(improved_mask):
            assignment[improved_mask] = pre_open_arr[best_idx[improved_mask]]
            current_costs[improved_mask] = best_costs[improved_mask]
        # Ensure pre-open bases remain explicit bases.
        assignment[pre_open_arr] = pre_open_arr
        current_costs[pre_open_arr] = base_open_cost[pre_open_arr]

    # Filter candidates: exclude already pre-opened bases
    remaining_candidates = [
        int(idx)
        for idx in candidate_indices
        if 0 <= int(idx) < n and int(idx) not in pre_open_set
    ]

    greedy_loop_start = time.perf_counter()
    eps = 1e-12

    # Priority queue storing (-gain, candidate_index)
    pq: List[Tuple[float, int]] = []

    if remaining_candidates:
        # Numba optimization: batch compute initial gains
        candidates_arr = np.array(remaining_candidates, dtype=np.int64)
        gains = _compute_gains_batch_numba(
            RdSi_full, 
            current_costs, 
            candidates_arr, 
            base_open_cost
        )
        
        for idx, gain in zip(remaining_candidates, gains):
            if gain > eps:
                heapq.heappush(pq, (-float(gain), int(idx)))

    while pq:
        # Pop best candidate (upper bound gain)
        neg_gain, best_cand = heapq.heappop(pq)
        
        # If the best possible gain is negligible, we are done
        if -neg_gain <= eps:
            break

        # Re-calculate exact marginal gain against CURRENT solution state
        # Only extract ONE column -> O(N) memory.
        # Numba handles strided access efficiently.
        col = RdSi_full[:, best_cand]
        
        real_gain = _compute_marginal_gain_numba(
            col, 
            current_costs, 
            int(best_cand), 
            float(base_open_cost[best_cand])
        )

        # Check against the next best candidate in the heap
        if not pq:
            # No competitors, if gain is positive, take it
            if real_gain > eps:
                _update_state_numba(
                    col, 
                    current_costs, 
                    assignment, 
                    int(best_cand), 
                    float(base_open_cost[best_cand])
                )
            break

        # Lazy Greedy condition:
        # If real_gain >= upper_bound_of_next, then this is truly the best.
        next_neg_gain = pq[0][0]
        if real_gain >= -next_neg_gain:
             if real_gain > eps:
                # SELECT THIS CANDIDATE
                _update_state_numba(
                    col, 
                    current_costs, 
                    assignment, 
                    int(best_cand), 
                    float(base_open_cost[best_cand])
                )
             else:
                 # Best gain is negligible, stopping early
                 break
        else:
            # Push back with updated gain
            heapq.heappush(pq, (-float(real_gain), int(best_cand)))

    greedy_loop_sec = time.perf_counter() - greedy_loop_start

    # ------------------------------------------------------------------
    # Optional local search refinement
    # ------------------------------------------------------------------
    finalize_start = time.perf_counter()
    base_mask = assignment == np.arange(n)
    bases_arr = np.nonzero(base_mask)[0]
    assignment_vec = assignment.copy()

    local_search_sec = 0.0
    if params.use_local_swap:
        local_search_start = time.perf_counter()
        assignment_vec, bases_arr, _ = _run_local_swap_refinement(
            RdSi_full,
            base_open_cost,
            candidate_indices,
            bases_arr,
            params,
        )
        local_search_sec = time.perf_counter() - local_search_start

    bases_arr = np.asarray(bases_arr, dtype=int)
    bases_arr.sort()
    bases = bases_arr.tolist()

    # ------------------------------------------------------------------
    # Final cost accounting (align exactly with _compute_objective semantics)
    # ------------------------------------------------------------------
    assignment_vec, objective_total = _compute_objective_and_assignment(
        RdSi_full,
        base_open_cost,
        bases_arr,
    )
    base_storage_total = float(base_open_cost[bases_arr].sum())
    sum_Rd_si = float(objective_total - base_storage_total)
    if sum_Rd_si < 0.0 and abs(sum_Rd_si) <= 1e-9 * max(1.0, objective_total):
        sum_Rd_si = 0.0
    finalization_sec = time.perf_counter() - finalize_start
    total_sec = time.perf_counter() - total_start

    if params.verbose:
        if params.use_local_swap:
            print(
                "[greedy] timing: candidates={:.3f}s ratio_matrix={:.3f}s "
                "greedy_loop={:.3f}s local_search={:.3f}s "
                "finalization={:.3f}s total={:.3f}s".format(
                    candidate_selection_sec,
                    ratio_build_sec,
                    greedy_loop_sec,
                    local_search_sec,
                    finalization_sec,
                    total_sec,
                )
            )
        else:
            print(
                "[greedy] timing: candidates={:.3f}s ratio_matrix={:.3f}s "
                "greedy_loop={:.3f}s finalization={:.3f}s total={:.3f}s".format(
                    candidate_selection_sec,
                    ratio_build_sec,
                    greedy_loop_sec,
                    finalization_sec,
                    total_sec,
                )
            )

    assignment_dict: Dict[int, int] = {int(i): int(j) for i, j in enumerate(assignment_vec)}

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
            "candidate_indices": candidate_indices,
            "num_candidates": len(candidate_indices),
        },
    }

