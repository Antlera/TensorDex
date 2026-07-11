"""FlexSplit adaptive planner: opens nearby bases and beats the star on drift."""

from __future__ import annotations

import numpy as np

from tensordex.compression import flexsplit as fd
from tensordex.core.engine import HYBRID_COEFFS_V3

N_BITS = 81_000  # tuned so monotonic drift spans predict_cr's useful p range


def _drift_fps(n: int, k: int = 8, step: int = 10) -> np.ndarray:
    """A monotonic-drift series: distance grows with checkpoint separation."""
    rng = np.random.default_rng(0)
    base = rng.integers(0, 100, size=k).astype(np.int64)
    return np.stack([base + i * step for i in range(n)])


def _cost(R: np.ndarray, bases: list[int]) -> float:
    n = len(R)
    bset = set(bases)
    return len(bases) + sum(
        min(R[i, b] for b in bases) for i in range(n) if i not in bset
    )


def test_flexsplit_opens_multiple_bases_and_beats_star() -> None:
    n = 10
    fps = _drift_fps(n)
    bases, attach = fd.plan_group(fps, N_BITS, coeffs=HYBRID_COEFFS_V3, bcs_d=2)
    R = fd.pairwise_cr(fps, N_BITS, HYBRID_COEFFS_V3, 2)

    assert 0 in bases
    assert len(bases) >= 2  # drift forces more than one base

    # every non-base attaches to its NEAREST opened base
    for i, b, cr in attach:
        assert b in bases
        assert abs(cr - min(R[i, bb] for bb in bases)) < 1e-9

    # FlexSplit strictly beats anchoring everything on the earliest (star-first)
    star_first = 1.0 + sum(R[i, 0] for i in range(1, n))
    assert _cost(R, bases) < star_first


def test_flexsplit_trivial_series() -> None:
    fps = _drift_fps(1)
    bases, attach = fd.plan_group(fps, N_BITS, coeffs=HYBRID_COEFFS_V3, bcs_d=2)
    assert bases == [0]
    assert attach == []
