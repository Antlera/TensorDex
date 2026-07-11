#!/usr/bin/env python3
"""Compare base-selection strategies for a long checkpoint chain.

Given a hub holding a time-ordered series of checkpoints (e.g. Pythia
``stepN`` revisions), this predicts — per parameter, from the BCS
fingerprints + the Hybrid CR model, no actual compression — the storage
each strategy would achieve:

  - star-first   : one raw base = the earliest ckpt; everyone deltas to it
  - star-mid     : one raw base = the middle ckpt
  - chain        : ckpt_i deltas against ckpt_{i-1} (min storage, depth N)
  - k-center     : K evenly-spaced raw bases; attach to nearest
  - split (adapt): FlexSplit recursive split — open a raw base only where
                   the predicted byte gain exceeds its cost (the base
                   reverting to full size). How many splits is decided by
                   the fingerprints, not a fixed K.

All non-chain strategies reconstruct in depth 1 (every member deltas
against a *raw* base). Reported numbers are predicted; validate the
winner with `tensordex compress`.

    python examples/benchmark_chain_strategies.py <hub> "<model-glob>"
"""

from __future__ import annotations

import fnmatch
import json
import sqlite3
import sys
from math import prod

import numpy as np

HYBRID = (-23.727944, 0.522466, 1.966862, -0.043132)
BCS_W = 1024  # d=2 rows of w=1024 → 2048 int32
DT_BITS = {
    "torch.float32": 32, "torch.float16": 16, "torch.bfloat16": 16,
    "torch.float64": 64, "torch.int64": 64, "torch.int8": 8, "torch.bool": 8,
}


def predict_cr(p: float) -> float:
    p = min(max(p, 0.0), 0.5)
    h = 0.0 if p <= 1e-15 else -p * np.log2(p) - (1 - p) * np.log2(1 - p)
    t = 8.0 * h
    a, b, c, d = HYBRID
    return float(min(max(a * p + b * t + c * p * t + d, 0.0), 1.0))


def bcs_dist(fa: np.ndarray, fb: np.ndarray, n_bits: float) -> float:
    diff = fa.astype(np.int64) - fb.astype(np.int64)
    return float((diff * diff).sum()) / 2.0 / max(n_bits, 1.0)


# --- FlexSplit recursive split (find_best_split + _try_split_subcluster) ----

def _best_open(members, base, R):
    """Best member to open as a new raw base; returns (gain, idx) or (<=0,-1)."""
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


def _split(members, base, R, bases, min_ratio):
    """Recursively split; append discovered raw bases to `bases`."""
    if len(members) < 1:
        return
    star_cost = 1.0 + sum(R[m, base] for m in members)  # in units of L
    gain, cand = _best_open(members, base, R)
    if cand == -1 or gain <= 0 or gain / max(star_cost, 1e-9) < min_ratio:
        return
    bases.append(cand)
    stay, move = [], []
    for m in members:
        if m == cand:
            continue
        (move if R[m, cand] < R[m, base] else stay).append(m)
    if len(stay) >= 2:
        _split(stay, base, R, bases, min_ratio)
    if len(move) >= 2:
        _split(move, cand, R, bases, min_ratio)


def strat_cost(R, n, strategy, *, k=4, min_ratio=0.0):
    """Cost in units of L (one tensor's logical size) for a param's chain.

    Returns (cost, n_bases). Lower cost = better; baseline (all raw) = n.
    """
    if strategy == "chain":
        return 1.0 + sum(R[i, i - 1] for i in range(1, n)), 1
    if strategy == "star-first":
        return 1.0 + sum(R[i, 0] for i in range(1, n)), 1
    if strategy == "star-mid":
        m = n // 2
        return 1.0 + sum(R[i, m] for i in range(n) if i != m), 1
    if strategy == "k-center":
        bases = sorted(set(round(j * (n - 1) / (k - 1)) for j in range(k))) if n >= k else list(range(n))
        cost = len(bases) + sum(
            min(R[i, b] for b in bases) for i in range(n) if i not in bases
        )
        return cost, len(bases)
    if strategy == "split":
        bases = [0]
        _split(list(range(1, n)), 0, R, bases, min_ratio)
        bases = sorted(set(bases))
        cost = len(bases) + sum(
            min(R[i, b] for b in bases) for i in range(n) if i not in bases
        )
        return cost, len(bases)
    raise ValueError(strategy)


def main() -> None:
    hub, glob = sys.argv[1], sys.argv[2]
    db = sqlite3.connect(f"{hub}/metadata.db")
    models = [
        r[0] for r in db.execute("SELECT model_name FROM model_meta ORDER BY created_at")
        if fnmatch.fnmatchcase(r[0], glob)
    ]
    n = len(models)
    if n < 3:
        sys.exit(f"need >=3 checkpoints, found {n} matching {glob!r}")
    print(f"chain: {n} checkpoints matching {glob!r}\n")

    # param -> ordered list of (fp int32 array, n_bits, logical_bytes)
    chains: dict = {}
    for ckpt in models:
        rows = db.execute(
            """SELECT mm.param_name, t.shape, t.dtype, t.fingerprint
               FROM model_mappings mm JOIN tensors t ON t.id = mm.tensor_id
               WHERE mm.model_name = ?""",
            (ckpt,),
        ).fetchall()
        for param, shape_json, dtype, fp_blob in rows:
            if fp_blob is None:
                continue
            shape = json.loads(shape_json) or [1]
            numel = prod(shape)
            bits = DT_BITS.get(dtype, 32)
            fp = np.frombuffer(fp_blob, dtype=np.int32)
            chains.setdefault(param, []).append((fp, numel * bits, numel * bits / 8.0))

    strategies = ["chain", "star-first", "star-mid", "k-center", "split"]
    totals = {s: 0.0 for s in strategies}
    nbases = {s: 0 for s in strategies}
    baseline = 0.0

    for param, seq in chains.items():
        if len(seq) != n:  # only params present in every checkpoint
            continue
        fps = [s[0] for s in seq]
        n_bits = seq[0][1]
        Lbytes = seq[0][2]
        # pairwise pred_cr matrix
        R = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    R[i, j] = predict_cr(bcs_dist(fps[i], fps[j], n_bits))
        baseline += n * Lbytes
        for s in strategies:
            cost_units, nb = strat_cost(R, n, s)
            totals[s] += cost_units * Lbytes
            nbases[s] += nb

    print(f"{'strategy':12s} {'stored':>10s} {'reduce-to':>10s} {'saved':>7s} "
          f"{'rawBases':>9s} {'depth':>6s}")
    gb = 1e9
    for s in strategies:
        depth = "N" if s == "chain" else "1"
        print(f"{s:12s} {totals[s]/gb:9.2f}G {totals[s]/baseline*100:9.1f}% "
              f"{(1-totals[s]/baseline)*100:6.1f}% {nbases[s]:9d} {depth:>6s}")
    print(f"\nbaseline (all {n} ckpts raw) = {baseline/gb:.2f} GB")


if __name__ == "__main__":
    main()
