"""
Micro-scale algorithms for TensorDex clustering and compression experiments.

This module provides standalone implementations of:
- Fingerprint normalization and distance computation
- Candidate selection for facility location
- Compression ratio modeling
- Shared data structures (FacilityParams)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch


# ============================================================================
# Data classes
# ============================================================================

@dataclass
class FacilityParams:
    """
    Parameters for facility location problem.

    Attributes:
        seed: Random seed for reproducibility
        r_min: Minimum compression ratio (at d=0)
        r_max: Maximum compression ratio (asymptotic limit)
        scale: Scale parameter controlling the distance growth rate
        d_max: Maximum allowed distance for ILP assignment
        base_storage_multiplier: Base storage cost multiplier
        base_meta_bytes: Base metadata overhead in bytes
        normalize: Deprecated. Fingerprints are consumed in raw form.
        candidate_reduction: Enable candidate reduction before ILP
        candidate_strategy: Selection strategy for facility candidates
            - "kmeans++": K-means++ initialization (probabilistic, only supported option)
        candidate_topk: Number of facility candidates to keep before solving ILP (None = use default)
        neighbor_radius: Legacy compatibility parameter (unused when only using K-means++)
        max_ilp_items: Max items for ILP candidate reduction (fallback default)
        small_ilp_cutoff: Problem size threshold for candidate reduction
        verbose: Enable verbose output
        local_search_k: maximum swap group size for local search (k-swap neighborhood).
    """
    seed: int = 1234
    r_min: float = 0.05
    r_max: float = 0.60
    scale: float = 1e-3
    d_max: float = 100.0
    base_storage_multiplier: float = 1.0
    base_meta_bytes: int = 0
    normalize: str = "distance_sqrt_numel"
    candidate_reduction: bool = False
    candidate_strategy: str = "kmeans++"
    candidate_topk: Optional[int] = None
    neighbor_radius: float = 0.01
    max_ilp_items: int = 5000000
    small_ilp_cutoff: int = 200000
    verbose: bool = True
    use_local_swap: bool = True
    local_search_max_iters: int = 4000
    local_search_max_swaps: int = 4000
    local_search_k: int = 1
    local_search_max_candidates_per_base: int = 64
    local_search_eps: float = 1e-6


# ============================================================================
# Utility functions
# ============================================================================

def dtype_bytes(dtype_str: str) -> int:
    """Map dtype string to bytes per element."""
    dtype_map = {
        "float32": 4, "torch.float32": 4,
        "float16": 2, "torch.float16": 2,
        "bfloat16": 2, "torch.bfloat16": 2,
        "float64": 8, "torch.float64": 8,
        "int8": 1, "torch.int8": 1,
        "uint8": 1, "torch.uint8": 1,
        "int16": 2, "torch.int16": 2,
        "int32": 4, "torch.int32": 4,
        "int64": 8, "torch.int64": 8,
        "bool": 1, "torch.bool": 1,
    }
    return dtype_map.get(dtype_str.lower(), 4)


def tensor_nbytes(shape: Tuple[int, ...], dtype_str: str) -> int:
    """Compute tensor size in bytes."""
    numel = int(np.prod(shape))
    return numel * dtype_bytes(dtype_str)


# ============================================================================
# Normalization & distances
# ============================================================================

def normalize_fingerprints(
    ids: Sequence[str],
    fingerprint_db: Mapping[str, torch.Tensor],
) -> np.ndarray:
    """
    Stack raw fingerprints into an array of shape (n, d).

    Args:
        ids: Ordered tensor identifiers whose fingerprints are required.
        fingerprint_db: Mapping from tensor ID to fingerprint tensor.

    Returns:
        Raw fingerprint matrix of shape (n, d).

    Notes:
        Fingerprint values are no longer scaled or normalized. Callers must
        supply any downstream normalization logic explicitly.
    """
    fps = np.stack([fingerprint_db[tid].numpy() for tid in ids])
    return fps


def pairwise_distance_matrix(X: np.ndarray) -> np.ndarray:
    """
    Compute pairwise distance matrix.
    Args:
        X: Feature matrix of shape (n, d)

    Returns:
        Distance matrix of shape (n, n)
    """
    # Optimization: Use ||a-b||^2 = ||a||^2 + ||b||^2 - 2<a,b>
    # This avoids creating the intermediate (n, n, d) difference tensor
    # which causes O(n^2 * d) memory explosion.
    
    # Ensure float32 at least for precision in dot product
    if X.dtype == np.float16 or X.dtype == np.int8 or X.dtype == np.uint8:
        X = X.astype(np.float32)
        
    # (n, n) dot product
    # Note: matmul is generally efficient
    dot_prod = np.dot(X, X.T)
    
    # (n,) squared norms
    sq_norm = np.sum(X * X, axis=1)
    
    # dist^2 = norm^2 + norm^2 - 2*dot
    # Broadcasting: (n, 1) + (1, n) - (n, n)
    dist_sq = sq_norm[:, None] + sq_norm[None, :] - 2 * dot_prod
    
    # Clip negative values due to numerical precision
    np.maximum(dist_sq, 0.0, out=dist_sq)
    
    return np.sqrt(dist_sq)


def normalize_distance_matrix_by_size(
    ids: Sequence[str],
    D: np.ndarray,
    *,
    tensor_shapes: Optional[Mapping[str, Sequence[int]]] = None,
    tensor_num_elements: Optional[Mapping[str, int]] = None,
    per_item_num_elements: Optional[Sequence[Optional[int]]] = None,
) -> np.ndarray:
    """
    Normalize distances by sqrt(#elements) to compare tensors of different sizes.

    Args:
        ids: Ordered tensor identifiers that define the row/column order in `D`.
        D: Raw distance matrix of shape (n, n).
        tensor_shapes: Optional mapping from tensor id to its shape tuple.
        tensor_num_elements: Optional mapping from tensor id to a precomputed
            number of elements (∏ dims). Takes precedence over tensor_shapes.
        per_item_num_elements: Optional iterable of precomputed element counts
            aligned with `ids`. Useful when counts are already known.

    Returns:
        Distance matrix divided by sqrt(max(numel_i, numel_j)) for each pair.
        Entries lacking valid shape/size metadata are left unmodified.
    """
    ids = list(ids)
    n = len(ids)
    D = np.asarray(D, dtype=np.float64)
    if D.shape != (n, n):
        raise ValueError(
            f"Distance matrix shape {D.shape} does not match ids length {n}"
        )

    counts = np.full(n, np.nan, dtype=np.float64)

    def _safe_to_int(value: Optional[Any]) -> Optional[float]:
        if value is None:
            return None
        try:
            num = float(value)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(num) or num <= 0:
            return None
        return num

    if per_item_num_elements is not None:
        per_item_num_elements = list(per_item_num_elements)
        if len(per_item_num_elements) != n:
            raise ValueError(
                "per_item_num_elements must match ids length "
                f"(got {len(per_item_num_elements)} vs {n})"
            )
        for idx, value in enumerate(per_item_num_elements):
            counts[idx] = _safe_to_int(value) or np.nan
    else:
        tensor_shapes = tensor_shapes or {}
        tensor_num_elements = tensor_num_elements or {}
        for idx, tid in enumerate(ids):
            numel = tensor_num_elements.get(tid)
            if numel is None:
                shape = tensor_shapes.get(tid)
                if shape is not None:
                    try:
                        shape_vals = tuple(int(dim) for dim in shape)
                        numel = int(np.prod(shape_vals))
                    except (TypeError, ValueError, OverflowError):
                        numel = None
            counts[idx] = _safe_to_int(numel) or np.nan

    valid_mask = np.isfinite(counts) & (counts > 0)
    if not np.any(valid_mask):
        return np.array(D, copy=True)

    with np.errstate(invalid="ignore"):
        denom = np.sqrt(np.maximum.outer(counts, counts))

    normed = np.array(D, copy=True)
    valid_pairs = np.isfinite(denom) & (denom > 0)
    normed[valid_pairs] = normed[valid_pairs] / denom[valid_pairs]
    return normed


# ============================================================================
# Candidate reduction helpers
# ============================================================================

def fast_kmeanspp_candidates(D: np.ndarray, k: int, seed: int = 1234) -> List[int]:
    """
    Vectorized O(k·n) KMeans++ from precomputed distance matrix D (n×n).

    This is the optimized implementation that maintains a running min_d2 array
    and only updates it incrementally with each new center, avoiding the O(k²·n)
    recomputation of the naive approach.

    Assumes D[i, j] is the distance between items i and j.

    Args:
        D: Distance matrix of shape (n, n)
        k: Number of candidates to select
        seed: Random seed for reproducibility

    Returns:
        List of selected candidate indices (sorted)

    Raises:
        ValueError: If D is not a square matrix or contains invalid values
    """
    # Guardrails: validate input
    if D.ndim != 2:
        raise ValueError(f"D must be 2D, got shape {D.shape}")

    n = D.shape[0]
    if D.shape[1] != n:
        raise ValueError(f"D must be square, got shape {D.shape}")

    if k >= n:
        return list(range(n))

    # Ensure D is C-contiguous and float64 for stability and performance
    if not D.flags['C_CONTIGUOUS'] or D.dtype != np.float64:
        D = np.ascontiguousarray(D, dtype=np.float64)

    # Check for NaNs and handle them
    if np.any(np.isnan(D)):
        # Replace NaNs with large finite values for sampling purposes
        # This ensures we don't sample infeasible pairs
        D = D.copy()
        D[np.isnan(D)] = np.nanmax(D) * 10.0 if not np.all(np.isnan(D)) else 1e10

    rng = np.random.default_rng(seed)

    # Select first center uniformly at random
    first = rng.integers(n)
    centers = [first]

    # Initialize min squared distances to nearest center
    # Start with distances to the first center
    min_d2 = (D[first] ** 2).astype(np.float64, copy=True)

    # K-means++ loop: select k-1 more centers
    for _ in range(1, k):
        # Sample proportional to squared distance
        prob_sum = np.sum(min_d2)

        if prob_sum <= 0 or not np.isfinite(prob_sum):
            # Fallback: all remaining points are equidistant or invalid
            # Sample uniformly from unselected points
            remaining = [i for i in range(n) if i not in centers]
            if remaining:
                nxt = rng.choice(remaining)
            else:
                break
        else:
            probs = min_d2 / prob_sum
            nxt = rng.choice(n, p=probs)

        centers.append(nxt)

        # Update nearest-center squared distances (only with new center!)
        # This is the key optimization: O(n) update instead of O(k·n) recomputation
        np.minimum(min_d2, D[nxt] ** 2, out=min_d2)

    return sorted(centers)


def select_candidate_indices(
    D: np.ndarray,
    sizes: np.ndarray,
    params: FacilityParams,
    model: MicroCompressionModel,
    k: int
) -> List[int]:
    """
    Unified dispatcher for distance-based candidate selection strategies.

    Args:
        D: Distance matrix of shape (n, n)
        sizes: Tensor sizes in bytes, array of shape (n,)
        params: Facility location parameters
        model: Compression model for R(d)
        k: Number of candidates to select

    Returns:
        List of candidate indices (ordering depends on strategy)
    """
    n = D.shape[0]
    if k >= n:
        return list(range(n))

    strat = params.candidate_strategy

    if strat != "kmeans++":
        raise ValueError(
            f"candidate_strategy must be 'kmeans++' after simplification, got: {strat}"
        )

    return fast_kmeanspp_candidates(D, k, params.seed)


def compute_ratio_matrix_from_D(
    D: np.ndarray,
    model: MicroCompressionModel,
    d_max: float
) -> np.ndarray:
    """
    Build ratio matrix from distance matrix with clamping and d_max logic.

    For each entry D[i,j]:
    - If NaN or D[i,j] > d_max: ratio = 1.0 (infeasible)
    - Otherwise: ratio = R(D[i,j]), clamped to [0,1]
    - Diagonal forced to 1.0

    Args:
        D: Distance matrix of shape (n, n). NaN entries represent infeasible pairs.
        model: Compression model with R(d) function
        d_max: Maximum allowed distance for feasibility

    Returns:
        Ratio matrix of shape (n, n) with values in [0,1]
    """
    ratio = np.ones(D.shape, dtype=np.float32)

    # Identify feasible entries: finite and <= d_max
    feasible = np.isfinite(D) & (D <= d_max)

    # Evaluate compression model only on feasible entries
    if np.any(feasible):
        ratio_vals = np.asarray(model.R(D[feasible]), dtype=np.float32)
        np.clip(ratio_vals, 0.0, 1.0, out=ratio_vals)
        ratio[feasible] = ratio_vals

    # Force diagonal to 1.0 (self-storage / no compression)
    np.fill_diagonal(ratio, 1.0)

    return ratio


# ============================================================================
# Compression ratio model
# ============================================================================

class MicroCompressionModel:
    """
    Piecewise PowerExp|Exponential (numpy only), continuous at breakpoint t.

    Right side is anchored for continuity:
      Right(d) = Exp(d; θ_r) - Exp(t; θ_r) + Left(t; θ_l)

    ⚠️ Backward-compatible constructor signature; args are ignored.
       Hard-coded to your latest best-fit params.

    Best-fit (from 2025-11-20 run on
      per_pair_distances_none_euclidean_per_sqrt_elements.csv):

      Model: Piecewise[PowerExp|Exponential]
      Param order: (t, l_rmin, l_rmax, l_scale, l_p, r_rmin, r_rmax, r_scale)

      t        = 8.75e-04
      Left (PowerExp):      r_min=0.0,       r_max=0.653618,
                            scale=1.34e-04,  p=0.496238
      Right (Exponential):  r_min=0.742603,  r_max=1.037701,
                            scale=2.942e-03
    """

    def __init__(self,
                 r_min=0.0,
                 r_max=1.0,
                 scale=1.0,
                 p=1.0,
                 _eps=1e-12):
        self._eps = float(_eps)

        # --- hard-coded best-fit params (2025-11-20) ---
        self.t        = 8.75e-04

        # Left: PowerExp
        self.l_rmin   = 0.0
        self.l_rmax   = 0.653618
        self.l_scale  = 1.34e-04
        self.l_p      = 0.496238

        # Right: Exponential
        self.r_rmin   = 0.742603
        self.r_rmax   = 1.037701
        self.r_scale  = 2.942e-03

        # Expose legacy fields
        self.r_min = self.l_rmin
        self.r_max = self.r_rmax
        self.scale = self.l_scale
        self.p     = self.l_p

    # ---------- base models ----------
    def _power_exp(self, d, r_min, r_max, scale, p):
        r_min, r_max = (r_min, r_max) if r_min <= r_max else (r_max, r_min)
        scale = max(scale, self._eps)
        p     = max(p, self._eps)
        x = np.maximum(np.asarray(d, dtype=np.float64), 0.0)
        z = (x / scale) ** p
        return r_min + (r_max - r_min) * (1.0 - np.exp(-z))

    def _exp(self, d, r_min, r_max, scale):
        r_min, r_max = (r_min, r_max) if r_min <= r_max else (r_max, r_min)
        scale = max(scale, self._eps)
        x = np.maximum(np.asarray(d, dtype=np.float64), 0.0)
        return r_min + (r_max - r_min) * (1.0 - np.exp(-x / scale))

    # ---------- public API ----------
    def R(self, d):
        x = np.asarray(d, dtype=np.float64)
        x = np.maximum(x, 0.0)  # d < 0 clipped to 0
        t = float(self.t)

        # Left (d <= t): PowerExp
        mL = x <= t
        y  = np.empty_like(x, dtype=np.float64)
        if np.any(mL):
            y[mL] = self._power_exp(
                x[mL], self.l_rmin, self.l_rmax, self.l_scale, self.l_p
            )
            y_t   = float(self._power_exp(
                t, self.l_rmin, self.l_rmax, self.l_scale, self.l_p
            ))
        else:
            y_t   = float(self._power_exp(
                t, self.l_rmin, self.l_rmax, self.l_scale, self.l_p
            ))

        # Right (d > t): anchored Exponential
        mR = ~mL
        if np.any(mR):
            base_at = self._exp(
                x[mR], self.r_rmin, self.r_rmax, self.r_scale
            )
            base_t  = float(self._exp(
                t, self.r_rmin, self.r_rmax, self.r_scale
            ))
            y[mR]   = base_at - base_t + y_t

        # Sane global clipping using asymptotes
        left_lo = min(self.l_rmin, self.l_rmax)
        left_hi = max(self.l_rmin, self.l_rmax)

        # Right-side asymptote as d -> +inf:
        #   Right_inf = Exp(+inf) - Exp(t) + y_t
        #             = r_rmax - Exp(t) + y_t
        #             = (r_rmax - r_rmin) * exp(-t/scale) + y_t
        right_inf = (
            (self.r_rmax - self.r_rmin)
            * np.exp(-t / max(self.r_scale, self._eps))
            + y_t
        )

        # Anchored right branch at d = 0
        right_at0 = float(
            self._exp(0.0, self.r_rmin, self.r_rmax, self.r_scale)
            - self._exp(t, self.r_rmin, self.r_rmax, self.r_scale)
            + y_t
        )

        g_lo = min(left_lo, y_t, right_at0)
        g_hi = max(left_hi, float(right_inf))
        y    = np.clip(y, g_lo, g_hi)

        return float(y) if np.isscalar(d) else y

    __call__ = R
