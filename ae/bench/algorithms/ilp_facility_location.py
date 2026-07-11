"""
ILP facility location solver using Gurobi for TensorDex clustering and compression.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np

from .micro_algorithms import (
    FacilityParams,
    MicroCompressionModel,
    normalize_distance_matrix_by_size,
    select_candidate_indices,
)


def ilp_facility_location_gurobi(
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
    Formulate and solve facility location with Gurobi.

    Variables:
        - Binary x_j for base open
        - Binary y_ij for assignment (y_ij=0 if D[i,j] > d_max)

    Constraints:
        - sum_j y_ij = 1 for all i
        - y_ij <= x_j

    Objective:
        base_storage_multiplier * sum_j (x_j * size_j + x_j * base_meta_bytes)
        + sum_i sum_j (y_ij * R(D[i,j]) * size_i)

    Args:
        tensor_ids: IDs used to align sizes, distances, and metadata.
        D: Raw distance matrix of shape (n, n)
        sizes: Tensor sizes in bytes, array of shape (n,)
        params: Facility location parameters
        model: Compression model for R(d)
        tensor_shapes / tensor_num_elements / per_item_num_elements:
            Optional metadata that enables sqrt(#elements) distance scaling.

    Returns:
        Dictionary with same shape as greedy solver

    Raises:
        ImportError: If gurobipy is not available
    """
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except Exception as e:
        raise ImportError(
            "Gurobi not available. Install gurobipy and ensure a valid license."
        ) from e

    tensor_ids = list(tensor_ids)
    start_time = time.perf_counter()
    n = len(sizes)
    if len(tensor_ids) != n:
        raise ValueError(
            f"tensor_ids length {len(tensor_ids)} must match sizes length {n}"
        )

    sizes = np.asarray(sizes, dtype=np.float64)
    D = np.asarray(D, dtype=np.float64)

    D = normalize_distance_matrix_by_size(
        tensor_ids,
        D,
        tensor_shapes=tensor_shapes,
        tensor_num_elements=tensor_num_elements,
        per_item_num_elements=per_item_num_elements,
    )

    # Verbose logging for candidate reduction configuration
    if params.verbose:
        print(f"  [distance ILP] n={n}, small_ilp_cutoff={params.small_ilp_cutoff}")
        print(f"  [distance ILP] candidate_reduction={params.candidate_reduction}")
        print(f"  [distance ILP] candidate_strategy={params.candidate_strategy}")
        print(f"  [distance ILP] candidate_topk={params.candidate_topk}")

    # Candidate reduction if needed
    candidate_selection_start = time.perf_counter()
    candidate_indices = None
    if params.candidate_reduction and n > params.small_ilp_cutoff:
        # Determine how many candidates to keep
        if params.candidate_topk is not None:
            k = min(params.candidate_topk, n)
        else:
            # Fallback to existing default
            k = min(params.max_ilp_items, n)

        if params.verbose:
            print(f"  [distance ILP] Candidate reduction ENABLED: selecting {k} from {n}")

        # Use unified candidate selection dispatcher
        candidate_indices = select_candidate_indices(D, sizes, params, model, k)

        if params.verbose:
            print(f"  [distance ILP] Candidate selection: strategy={params.candidate_strategy}, "
                  f"selected={len(candidate_indices)} candidates")
            print(f"  [distance ILP] First 10 candidates: {candidate_indices[:10]}")
    else:
        candidate_indices = list(range(n))
        if params.verbose:
            if params.candidate_reduction:
                print(f"  [distance ILP] Candidate reduction DISABLED (n={n} <= small_ilp_cutoff={params.small_ilp_cutoff})")
            else:
                print(f"  [distance ILP] Candidate reduction DISABLED (candidate_reduction=False)")
            print(f"  [distance ILP] Using all {n} items as candidates")

    candidate_selection_sec = time.perf_counter() - candidate_selection_start
    if params.verbose:
        print(f"  [distance ILP] Candidate selection took {candidate_selection_sec:.3f}s")

    J = candidate_indices  # Candidate base indices
    I = list(range(n))  # All items

    # Create model
    if ratio_override is not None:
        ratio_matrix = np.asarray(ratio_override, dtype=np.float64)
        if ratio_matrix.shape != (n, n):
            raise ValueError(
                f"ratio_override shape {ratio_matrix.shape} does not match number of tensors {n}"
            )
        ratio_matrix = ratio_matrix.copy()
        np.fill_diagonal(ratio_matrix, 0.0)
    else:
        ratio_matrix = model.R(D)
        np.fill_diagonal(ratio_matrix, 0.0)

    m = gp.Model("facility_location")

    if not params.verbose:
        m.setParam('OutputFlag', 0)
        m.setParam('TimeLimit', 3600)

    m.setParam('MIPFocus', 1)

    # Variables
    x = {}  # x[j] = 1 if base j is opened
    for j in J:
        x[j] = m.addVar(vtype=GRB.BINARY, name=f"x_{j}")

    y = {}  # y[i,j] = 1 if item i assigned to base j
    for i in I:
        for j in J:
            if D[i, j] <= params.d_max and np.isfinite(ratio_matrix[i, j]):
                y[i, j] = m.addVar(vtype=GRB.BINARY, name=f"y_{i}_{j}")

    # Constraints: each item assigned to exactly one base
    for i in I:
        valid_bases = [j for j in J if (i, j) in y]
        if valid_bases:
            m.addConstr(gp.quicksum(y[i, j] for j in valid_bases) == 1, name=f"assign_{i}")
        else:
            # No valid base for i within d_max, force i to be its own base
            # Ensure x[i] exists (may not be in candidate set J)
            if i not in x:
                x[i] = m.addVar(vtype=GRB.BINARY, name=f"x_{i}")
            # Create y[i,i] for self-assignment
            if (i, i) not in y:
                y[i, i] = m.addVar(vtype=GRB.BINARY, name=f"y_{i}_{i}")
            # Force both to 1
            m.addConstr(x[i] == 1, name=f"force_base_{i}")
            m.addConstr(y[i, i] == 1, name=f"force_assign_{i}")

    # Constraints: y[i,j] <= x[j]
    for i in I:
        for j in J:
            if (i, j) in y:
                m.addConstr(y[i, j] <= x[j], name=f"link_{i}_{j}")

    # Objective
    # Include all opened bases (both from candidate set J and forced self-storage)
    base_cost = gp.quicksum(
        x[j] * (params.base_storage_multiplier * (sizes[j] + params.base_meta_bytes))
        for j in x.keys()
    )

    compressed_cost = gp.LinExpr()
    for i in I:
        for j in J:
            if (i, j) in y:
                ratio = float(ratio_matrix[i, j])
                compressed_cost += y[i, j] * ratio * sizes[i]
        # Include self-storage for forced assignments (i,i) where i not in J
        # if i not in J and (i, i) in y:
        #     # Self-storage has ratio=1.0 and D[i,i]=0
        #     compressed_cost += y[i, i] * 1.0 * sizes[i]

    m.setObjective(base_cost + compressed_cost, GRB.MINIMIZE)

    # Solve
    ilp_solve_start = time.perf_counter()
    m.optimize()
    ilp_solve_sec = time.perf_counter() - ilp_solve_start

    if params.verbose:
        print(f"  [distance ILP] ILP solve took {ilp_solve_sec:.3f}s")

    if m.status != GRB.OPTIMAL and m.status != GRB.TIME_LIMIT:
        raise RuntimeError(f"Gurobi optimization failed with status {m.status}")

    # Extract solution
    # Include both candidate bases from J and any forced self-storage bases
    bases = [j for j in x.keys() if x[j].X > 0.5]
    assignment = {}
    for i in I:
        # Check candidates in J
        for j in J:
            if (i, j) in y and y[i, j].X > 0.5:
                assignment[i] = j
                break
        # Check self-assignment for items not in J
        if i not in assignment and (i, i) in y and y[i, i].X > 0.5:
            assignment[i] = i

    # Ensure all items are assigned (fallback for infeasible items)
    for i in I:
        if i not in assignment:
            assignment[i] = i
            if i not in bases:
                bases.append(i)

    # Compute costs
    base_storage_total = sum(
        params.base_storage_multiplier * (sizes[j] + params.base_meta_bytes)
        for j in bases
    )

    sum_Rd_si = 0.0
    for i in I:
        j = assignment[i]
        if i == j:
            # self -> no residual
            continue
        ratio = float(ratio_matrix[i, j])
        sum_Rd_si += ratio * sizes[i]

    objective_total = base_storage_total + sum_Rd_si

    runtime_sec = time.perf_counter() - start_time

    if params.verbose:
        print(f"  [distance ILP] Total runtime: {runtime_sec:.3f}s "
              f"(candidate={candidate_selection_sec:.3f}s, ilp={ilp_solve_sec:.3f}s)")

    return {
        'bases': bases,
        'assignment': assignment,
        'costs': {
            'objective_total_bytes': float(objective_total),
            'base_storage_total_bytes': float(base_storage_total),
            "sum_Rd_si_bytes": float(sum_Rd_si),
        },
        'meta': {
            'num_targets': n,
            'num_bases': len(bases),
            'runtime_sec': runtime_sec,
            'candidate_selection_sec': candidate_selection_sec,
            'ilp_solve_sec': ilp_solve_sec,
            'candidate_indices': candidate_indices if candidate_indices else list(range(n)),
            'num_candidates': len(candidate_indices) if candidate_indices else n,
        }
    }
