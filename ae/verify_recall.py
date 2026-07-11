#!/usr/bin/env python3
"""Tier 1 (experiment) — TensorSketch Recall@1 (paper Fig 12a).

Reproduces the paper's similarity-search result: TensorSketch + an approximate
HNSW index picks the **same delta base** for each tensor as exact brute-force
search over the sketches — i.e. Recall@1 ≈ 1.0 — so the planner can select bases
from 8 KB fingerprints instead of reading full tensors.

This is a faithful port of the authors' internal ANN-vs-BCS benchmark (its
constants are shared with `tests/test_flexsplit.py`): greedy base-selection (attach a tensor to the
nearest existing base when the predicted CR clears the threshold, else open a
new base), run twice — once with a brute-force BCS index (ground truth) and once
with an hnswlib HNSW index — then measure the fraction of tensors assigned to the
**same base**. BCS fingerprints are recomputed from the shipped blobs with the
package's own `tensordex._ops`, so nothing is trusted from a table.

Usage:
    python ae/verify_recall.py [--blobs ...] [--min-recall 0.99]
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict

import numpy as np
import torch
from safetensors import safe_open

_AE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _AE_DIR)
from _blobs import available_ids, blob_path, real_tensor_key  # noqa: E402
from tensordex import _ops  # noqa: E402

DEFAULT_BLOBS = os.path.join(_AE_DIR, "cache", "sample_blobs")

# Constants from tests/test_flexsplit.py (BCS d=2,w=1024; TensorPred CR model).
# Note: these coefficients only steer greedy ATTACH decisions inside this
# experiment; the Fig 13 prediction claim is re-fit from scratch by
# fit_predict.py with NO stored coefficients.
BCS_W = 1024
HYBRID_COEFFS = np.array([-23.727944, 0.522466, 1.966862, -0.043132], dtype=np.float64)
STANDALONE_ZSTD_CR = 0.70  # attach threshold


def binary_entropy(p: float) -> float:
    if p <= 1e-15 or p >= 1.0 - 1e-15:
        return 0.0
    return -p * np.log2(p) - (1.0 - p) * np.log2(1.0 - p)


def predict_cr_hybrid(p: float, c=HYBRID_COEFFS) -> float:
    p = max(0.0, min(p, 0.5))
    t = 8.0 * binary_entropy(p)
    return max(0.0, min(c[0] * p + c[1] * t + c[2] * p * t + c[3], 1.0))


def bcs_norm_hamming(a, b, n_bits) -> float:
    d0 = a[:BCS_W].astype(np.int64) - b[:BCS_W].astype(np.int64)
    d1 = a[BCS_W:].astype(np.int64) - b[BCS_W:].astype(np.int64)
    return ((d0 * d0).sum() + (d1 * d1).sum()) / 2.0 / max(n_bits, 1)


def bcs_dist_one_vs_many(q, M, n_bits):
    d0 = M[:, :BCS_W].astype(np.int64) - q[:BCS_W].astype(np.int64)
    d1 = M[:, BCS_W:].astype(np.int64) - q[BCS_W:].astype(np.int64)
    return ((d0 * d0).sum(1) + (d1 * d1).sum(1)) / 2.0 / max(n_bits, 1)


class BruteForceBCSIndex:
    """Exact nearest base by BCS distance — the ground-truth index."""
    def __init__(self):
        self._ids, self._fps, self._n = [], [], 1
    def set_n_bits(self, n): self._n = n
    @property
    def size(self): return len(self._ids)
    def add_one(self, v, t): self._ids.append(t); self._fps.append(v.copy())
    def query(self, v):
        d = bcs_dist_one_vs_many(v, np.stack(self._fps), self._n)
        i = int(np.argmin(d)); return self._ids[i]


class HNSWLibIndex:
    """Approximate nearest base via hnswlib (L2 over 2048-dim fingerprints)."""
    def __init__(self, dim, m=32, efc=200, ef=64, maxel=100_000):
        import hnswlib
        self._ix = hnswlib.Index(space="l2", dim=dim)
        self._ix.init_index(max_elements=maxel, ef_construction=efc, M=m)
        self._ix.set_ef(ef); self._max = maxel; self._ids = []
    @property
    def size(self): return len(self._ids)
    def add_one(self, v, t):
        lbl = len(self._ids); self._ids.append(t)
        if lbl >= self._max:
            self._ix.resize_index(self._max * 2); self._max *= 2
        self._ix.add_items(v.astype(np.float32).reshape(1, -1), [lbl])
    def query(self, v):
        lbl, _ = self._ix.knn_query(v.astype(np.float32).reshape(1, -1), k=1)
        return self._ids[int(lbl[0, 0])]


def greedy_cluster(ordered, fp_db, shapes, nbits, make_index, thr=STANDALONE_ZSTD_CR):
    """Incremental greedy base-selection (FlexSplit Phase I) with a pluggable index."""
    groups = defaultdict(list)
    for t in ordered:
        groups[shapes[t]].append(t)
    tensor_to_base = {}
    for shp, tids in groups.items():
        nb = nbits[shp]
        idx = make_index(2 * BCS_W)
        if hasattr(idx, "set_n_bits"):
            idx.set_n_bits(nb)
        for t in tids:
            fp = fp_db[t]
            if idx.size == 0:
                tensor_to_base[t] = t; idx.add_one(fp.astype(np.float32), t); continue
            nt = idx.query(fp.astype(np.float32))
            p = bcs_norm_hamming(fp, fp_db[nt], nb)
            if predict_cr_hybrid(p) <= thr:               # attach to nearest base
                tensor_to_base[t] = nt
            else:                                          # open a new base
                tensor_to_base[t] = t; idx.add_one(fp.astype(np.float32), t)
    return tensor_to_base


def load_u16(path):
    with safe_open(path, framework="pt") as f:
        t = f.get_tensor(real_tensor_key(list(f.keys())))
    return t, t.view(torch.uint16).numpy().reshape(-1)


def main() -> int:
    ap = argparse.ArgumentParser(description="TensorDex AE — TensorSketch Recall@1")
    ap.add_argument("--blobs", default=DEFAULT_BLOBS)
    ap.add_argument("--min-recall", type=float, default=0.99)
    args = ap.parse_args()
    try:
        import hnswlib  # noqa: F401
    except ImportError:
        print("ERROR: pip install hnswlib  (see ae/requirements-ae.txt)")
        return 2
    if not os.path.isdir(args.blobs):
        print(f"ERROR: blob store not found at {args.blobs} — run `make ae-cache`.")
        return 2

    tids = sorted(available_ids(args.blobs))
    print(f"Recomputing BCS fingerprints (d=2, w=1024) for {len(tids)} tensors …")
    fp_db, shapes, nbits = {}, {}, {}
    for t in tids:
        tensor, u16 = load_u16(blob_path(args.blobs, t))
        fp_db[t] = np.asarray(_ops.compute_bcs_fingerprint_u16_py(u16), dtype=np.int32)
        shp = tuple(tensor.shape)
        shapes[t] = shp
        nbits[shp] = int(np.prod(shp)) * 16

    print("Greedy base-selection: brute-force (ground truth) vs hnswlib (HNSW) …")
    gt = greedy_cluster(tids, fp_db, shapes, nbits, lambda d: BruteForceBCSIndex())
    tt = greedy_cluster(tids, fp_db, shapes, nbits, lambda d: HNSWLibIndex(d))

    same = sum(1 for t in tids if gt[t] == tt[t])
    recall = same / len(tids)
    n_bases_gt = len(set(gt.values()))
    n_attached = len(tids) - n_bases_gt
    print("\n" + "=" * 64)
    print(f"tensors {len(tids)}  ·  bases {n_bases_gt}  ·  attached {n_attached}")
    print(f"TensorSketch Recall@1 (HNSW base == brute-force base) : {recall:.4f}")
    ok = recall >= args.min_recall
    print("\nRESULT:", f"PASS ✅  matches the paper's ~1.0 recall (≥{args.min_recall})"
          if ok else f"FAIL ❌  recall {recall:.4f} < {args.min_recall}")
    print("=" * 64)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
