#!/usr/bin/env python3
"""Fig 13 — reproduce the reduction-ratio prediction accuracy from the cache.

A prediction figure is, by its nature, *fit a model on the computed results and
measure it there*. TensorPred's hybrid model is linear in its four coefficients

    cr = c0*p + c1*t + c2*(p*t) + c3,   p = clip(bcs_dist, 0, 0.5),  t = 8*H(p)

so this script re-derives it from the cache by ordinary least squares — **no
pre-recorded coefficients are used**. It:

  1. loads every cached (bcs_dist, aratio) pair under the paper's Fig 13 filter;
  2. fits the model on a random *train* half and evaluates on the held-out half
     (proves the accuracy generalises, not an in-sample artefact);
  3. reports MAE / median AE / Pearson r in reduction-ratio space; and
  4. cross-checks the fit against the stored `pred_ratio` column for provenance,
     then re-renders Fig 13 from the freshly fit predictions.

Expected (matches the paper, 5.77 M pairs): held-out MAE ≈ 1.11 %, median ≈
0.79 %, Pearson ≈ 99.3 %; recovered predictions match the stored column to
max|Δ| ≈ 1e-4.

    python ae/fit_predict.py [--db ae/cache/results.db] [--seed 0]
                             [--test-frac 0.5] [--no-render] [--format png]
"""
from __future__ import annotations

import argparse
import os
import sqlite3
import sys

import numpy as np

_AE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _AE_DIR)
DEFAULT_DB = os.path.join(_AE_DIR, "cache", "results.db")


def _rr_stats(pred_cr, real_cr):
    """MAE, median AE, Pearson r in reduction-ratio (1-cr) space, as %/%."""
    pred_rr, real_rr = 1.0 - pred_cr, 1.0 - real_cr
    m = np.isfinite(pred_rr) & np.isfinite(real_rr)
    pred_rr, real_rr = pred_rr[m], real_rr[m]
    ae = np.abs(pred_rr - real_rr)
    r = np.corrcoef(pred_rr, real_rr)[0, 1]
    return ae.mean() * 100, np.median(ae) * 100, r * 100, int(m.sum())


def main() -> int:
    ap = argparse.ArgumentParser(description="TensorDex AE — Fig 13 prediction experiment")
    ap.add_argument("--db", default=DEFAULT_DB)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--test-frac", type=float, default=0.5)
    ap.add_argument("--no-render", action="store_true", help="skip re-rendering Fig 13")
    ap.add_argument("--format", default="png", choices=["pdf", "png", "svg"])
    args = ap.parse_args()

    if not os.path.exists(args.db):
        print(f"ERROR: cache DB not found at {args.db}\n"
              f"       run `make ae-cache-figures` first (downloads results.db).")
        return 2

    os.environ.setdefault("TENSORDEX_AE_CACHE", os.path.join(_AE_DIR, "cache"))
    from charts import flexsplit_analysis as fa  # noqa: E402 (after sys.path/env)
    fa._RESULTS_DB = args.db  # honour --db

    print("Fig 13 — re-fitting the TensorPred reduction-ratio predictor from cache")
    print(f"  db      : {args.db}")
    bcs, real, sizes = fa.load_pred_pairs()
    print(f"  pairs   : {len(bcs):,}  (bcs_dist > 0, aratio not null, bytes_in > 100KB)")

    # ── Held-out split: fit on train half, evaluate on the untouched half ──
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(bcs))
    cut = int(len(bcs) * (1.0 - args.test_frac))
    tr, te = idx[:cut], idx[cut:]

    coef_tr = fa.fit_hybrid_predictor(bcs[tr], real[tr])
    pred_te = fa.predict_hybrid(bcs[te], coef_tr)
    mae, med, r, n = _rr_stats(pred_te, real[te])

    # Full-data fit (this is what the figure shows) + provenance check.
    coef_full = fa.fit_hybrid_predictor(bcs, real)
    pred_full = fa.predict_hybrid(bcs, coef_full)
    mae_f, med_f, r_f, _ = _rr_stats(pred_full, real)

    print("\n  recovered hybrid coefficients (train split):")
    print(f"    c0={coef_tr[0]:+.4f}  c1={coef_tr[1]:+.4f}  "
          f"c2={coef_tr[2]:+.4f}  c3={coef_tr[3]:+.4f}")
    print("\n  {:<24s}{:>8s}{:>10s}{:>12s}".format("", "MAE", "Med.AE", "Pearson"))
    print("  {:<24s}{:>7.2f}%{:>9.2f}%{:>11.1f}%   (n={:,})".format(
        f"held-out ({int(args.test_frac*100)}% test)", mae, med, r, n))
    print("  {:<24s}{:>7.2f}%{:>9.2f}%{:>11.1f}%   (n={:,})".format(
        "full-data (figure)", mae_f, med_f, r_f, len(bcs)))
    print("  {:<24s}{:>7s} {:>9s} {:>10s}".format("paper (Fig 13)", "1.11%", "0.79%", "99.3%"))

    # ── Provenance: does the re-fit recover the stored pred_ratio column? ──
    con = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    stored = con.execute(
        "SELECT bcs_dist, pred_ratio FROM compression_results "
        "WHERE bcs_dist > 0 AND aratio IS NOT NULL AND pred_ratio IS NOT NULL "
        "AND bytes_in > ?", (100 * 1024,),
    ).fetchall()
    con.close()
    if stored:
        s = np.asarray(stored, dtype=float)
        refit = fa.predict_hybrid(s[:, 0], coef_full)
        d = np.abs(refit - s[:, 1])
        pr = np.corrcoef(refit, s[:, 1])[0, 1] * 100
        print(f"\n  provenance vs stored pred_ratio ({len(s):,} rows): "
              f"max|Δ|={d.max():.4f}  mean|Δ|={d.mean():.4f}  Pearson={pr:.3f}%")

    ok = (mae < 1.5 and r > 99.0)
    print(f"\n  {'PASS' if ok else 'FAIL'}: held-out prediction MAE {mae:.2f}% "
          f"(< 1.5%), Pearson {r:.1f}% (> 99%) — reproduces the paper's Fig 13.")

    # ── Re-render Fig 13 from the freshly fit predictions ──
    if not args.no_render:
        import matplotlib
        matplotlib.use("Agg")
        import render as _render  # noqa: E402
        from charts import CHARTS  # noqa: E402
        fa._pred_override = (pred_full, real, sizes)  # figure uses THIS fit
        rc = dict(_render.DEFAULT_RC, **_render.CHART_RC.get("pred_vs_real_ratio", {}))
        _render._apply_rc(rc)
        out_dir = os.path.join(_AE_DIR, "figures")
        os.makedirs(out_dir, exist_ok=True)
        fig = CHARTS["pred_vs_real_ratio"]["fn"](rc)
        path = os.path.join(out_dir, f"pred_vs_real_ratio.{args.format}")
        fig.savefig(path, format=args.format, dpi=150, bbox_inches="tight", facecolor="white")
        print(f"  rendered {path}")

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
