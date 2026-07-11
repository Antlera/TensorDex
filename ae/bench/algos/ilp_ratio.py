#!/usr/bin/env python3
"""
ILP facility location solver that operates directly on a ratio matrix.

Inputs:
- tensor_ids: list of ids (used only for alignment/bookkeeping)
- sizes: 1D array of tensor sizes (bytes)
- ratio_matrix: NxN matrix where ratio[i, j] * sizes[i] is the connect cost i -> base j

Candidate reduction can be emulated by passing candidate_indices.
"""

from __future__ import annotations

import time
from typing import Dict, Optional, Sequence

import numpy as np


def ilp_facility_location_ratio_gurobi(
    tensor_ids: Sequence[str],
    ratio_matrix: np.ndarray,
    sizes: Sequence[float],
    *,
    base_meta_bytes: float = 0.0,
    base_storage_multiplier: float = 1.0,
    candidate_indices: Optional[Sequence[int]] = None,
    time_limit_sec: Optional[int] = 3600,
    verbose: bool = False,
) -> Dict:
    """Solve facility location using Gurobi on a ratio matrix."""
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except Exception as e:
        raise ImportError("Gurobi not available. Install gurobipy and ensure a valid license.") from e

    tensor_ids = list(tensor_ids)
    sizes = np.asarray(sizes, dtype=np.float64)
    ratio_full = np.asarray(ratio_matrix, dtype=np.float64)
    n = len(sizes)
    if ratio_full.shape != (n, n):
        raise ValueError(f"ratio_matrix shape {ratio_full.shape} does not match sizes length {n}")
    if len(tensor_ids) != n:
        raise ValueError(f"tensor_ids length {len(tensor_ids)} must match sizes length {n}")

    ratio_full = ratio_full.copy()
    np.fill_diagonal(ratio_full, 0.0)
    feasible = np.isfinite(ratio_full)

    allowed_bases = (
        sorted({int(idx) for idx in candidate_indices if 0 <= int(idx) < n})
        if candidate_indices is not None
        else list(range(n))
    )
    I = list(range(n))
    J = allowed_bases

    model = gp.Model("facility_location_ratio")
    if not verbose:
        model.setParam("OutputFlag", 0)
    if time_limit_sec:
        model.setParam("TimeLimit", int(time_limit_sec))
    model.setParam("MIPFocus", 1)

    x = {j: model.addVar(vtype=GRB.BINARY, name=f"x_{j}") for j in J}
    y = {}
    for i in I:
        for j in J:
            if feasible[i, j]:
                y[i, j] = model.addVar(vtype=GRB.BINARY, name=f"y_{i}_{j}")

    # Assignment constraints
    for i in I:
        valid_bases = [j for j in J if (i, j) in y]
        if valid_bases:
            model.addConstr(gp.quicksum(y[i, j] for j in valid_bases) == 1, name=f"assign_{i}")
        else:
            # Force self-base if no feasible candidate
            if i not in x:
                x[i] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}")
            y[i, i] = model.addVar(vtype=GRB.BINARY, name=f"y_{i}_{i}")
            model.addConstr(x[i] == 1, name=f"force_base_{i}")
            model.addConstr(y[i, i] == 1, name=f"force_assign_{i}")

    # Link constraints
    for (i, j), var in y.items():
        model.addConstr(var <= x[j], name=f"link_{i}_{j}")

    # Objective
    base_cost = gp.quicksum(x[j] * (base_storage_multiplier * (sizes[j] + base_meta_bytes)) for j in x.keys())
    compressed_cost = gp.LinExpr()
    for (i, j), var in y.items():
        compressed_cost += var * float(ratio_full[i, j]) * sizes[i]
    model.setObjective(base_cost + compressed_cost, GRB.MINIMIZE)

    solve_start = time.perf_counter()
    model.optimize()
    solve_sec = time.perf_counter() - solve_start

    if model.status not in (GRB.OPTIMAL, GRB.TIME_LIMIT):
        raise RuntimeError(f"Gurobi optimization failed with status {model.status}")

    bases = [j for j in x.keys() if x[j].X > 0.5]
    assignment = {}
    for i in I:
        for j in J:
            if (i, j) in y and y[i, j].X > 0.5:
                assignment[i] = j
                break
        if i not in assignment and (i, i) in y and y[i, i].X > 0.5:
            assignment[i] = i
    for i in I:
        if i not in assignment:
            assignment[i] = i
            if i not in bases:
                bases.append(i)

    base_storage_total = sum(base_storage_multiplier * (sizes[j] + base_meta_bytes) for j in bases)
    sum_Rd_si = 0.0
    for i in I:
        j = assignment[i]
        if i == j:
            continue
        sum_Rd_si += float(ratio_full[i, j]) * sizes[i]
    objective_total = base_storage_total + sum_Rd_si

    runtime_sec = solve_sec
    return {
        "bases": bases,
        "assignment": assignment,
        "costs": {
            "objective_total_bytes": float(objective_total),
            "base_storage_total_bytes": float(base_storage_total),
            "sum_Rd_si_bytes": float(sum_Rd_si),
        },
        "meta": {
            "num_targets": n,
            "num_bases": len(bases),
            "runtime_sec": runtime_sec,
            "ilp_solve_sec": solve_sec,
            "candidate_indices": J,
            "num_candidates": len(J),
            "status": int(model.status),
        },
    }


__all__ = ["ilp_facility_location_ratio_gurobi"]

