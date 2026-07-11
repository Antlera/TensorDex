"""FlexSplit — adaptive multi-base planner for a checkpoint series.

The bundle *star* planner anchors every checkpoint on the earliest one, so a
long run's late checkpoints delta against a far-away base and compress poorly.
**FlexSplit** recursively divides the series, opening a new raw base only where
the predicted byte gain exceeds the cost of keeping that base raw — so each
checkpoint attaches to a *nearby* base, still at reconstruction depth 1
(every member deltas against a raw base, never a chain).

This ports the FlexSplit recursive split (``find_best_split`` /
``_try_split_subcluster``, mirrored in the landing-page ``splitRec`` demo)
and predicts pairwise compressibility from TensorSketch fingerprints via the
Hybrid CR model — no tensor bytes are read while planning.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

Coeffs = Tuple[float, float, float, float]


def predict_cr(p: float, coeffs: Coeffs) -> float:
    """Predicted delta ratio (compressed / raw) for normalized distance ``p``."""
    p = min(max(p, 0.0), 0.5)
    h = 0.0 if p <= 1e-15 else -p * np.log2(p) - (1 - p) * np.log2(1 - p)
    t = 8.0 * h
    a, b, c, d = coeffs
    return float(min(max(a * p + b * t + c * p * t + d, 0.0), 1.0))


def pairwise_cr(fps: np.ndarray, n_bits: float, coeffs: Coeffs, bcs_d: int) -> np.ndarray:
    """``R[i, j]`` = predicted ratio of deltaing tensor ``i`` against base ``j``."""
    f = np.asarray(fps).astype(np.int64)
    n = len(f)
    R = np.zeros((n, n), dtype=np.float64)
    denom = float(bcs_d) * max(float(n_bits), 1.0)
    for i in range(n):
        diff = f - f[i]
        d2 = (diff * diff).sum(axis=1) / denom
        for j in range(n):
            if i != j:
                R[i, j] = predict_cr(float(d2[j]), coeffs)
    return R


def _best_open(members: Sequence[int], base: int, R: np.ndarray) -> Tuple[float, int]:
    """Member that earns the most as a new raw base; (gain, idx) or (<=0, -1)."""
    best_gain, best = 0.0, -1
    for cand in members:
        if cand == base:
            continue
        gain = -(1.0 - R[cand, base])  # cand reverts to raw (size 1.0 in L units)
        for m in members:
            if m == cand:
                continue
            if R[m, cand] < R[m, base]:
                gain += R[m, base] - R[m, cand]
        if gain > best_gain:
            best_gain, best = gain, cand
    return best_gain, best


def _split(
    members: List[int], base: int, R: np.ndarray, bases: List[int], min_ratio: float
) -> None:
    """Recursively open raw bases; append each discovered base to ``bases``."""
    if len(members) < 1:
        return
    star_cost = 1.0 + sum(R[m, base] for m in members)
    gain, cand = _best_open(members, base, R)
    if cand == -1 or gain <= 0 or gain / max(star_cost, 1e-9) < min_ratio:
        return
    bases.append(cand)
    stay: List[int] = []
    move: List[int] = []
    for m in members:
        if m == cand:
            continue
        (move if R[m, cand] < R[m, base] else stay).append(m)
    if len(stay) >= 2:
        _split(stay, base, R, bases, min_ratio)
    if len(move) >= 2:
        _split(move, cand, R, bases, min_ratio)


def plan_group(
    fps: np.ndarray,
    n_bits: float,
    *,
    coeffs: Coeffs,
    bcs_d: int = 2,
    min_ratio: float = 0.0,
) -> Tuple[List[int], List[Tuple[int, int, float]]]:
    """Plan one tensor's checkpoint series (fingerprints in checkpoint order).

    Returns ``(base_indices, attach)`` where ``attach`` is a list of
    ``(target_idx, base_idx, predicted_cr)`` — every non-base member attached
    to its nearest opened base, at reconstruction depth 1.
    """
    n = len(fps)
    if n <= 1:
        return list(range(n)), []
    R = pairwise_cr(fps, n_bits, coeffs, bcs_d)
    bases: List[int] = [0]
    _split(list(range(1, n)), 0, R, bases, min_ratio)
    bases = sorted(set(bases))
    base_set = set(bases)
    attach: List[Tuple[int, int, float]] = []
    for i in range(n):
        if i in base_set:
            continue
        b = min(bases, key=lambda bb: R[i, bb])
        attach.append((i, b, float(R[i, b])))
    return bases, attach
