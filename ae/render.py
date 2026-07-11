#!/usr/bin/env python3
"""Tier 0 — render paper figures from the cached results.db (+ staged aux data).

Headless twin of the interactive `plots/serve.py`: it discovers every chart in
the `charts/` package, applies the publication rcParams, and writes each one to
`ae/figures/`. Charts whose data isn't staged are skipped with a reason rather
than aborting the run, so a reviewer always gets every satisfiable figure.

Usage:
    python ae/render.py [--db ae/cache/results.db] [--out ae/figures]
                        [--format pdf|png] [--only id1,id2,...]
"""
from __future__ import annotations

import argparse
import os
import sys
import traceback

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_AE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _AE_DIR)

DEFAULT_DB = os.path.join(_AE_DIR, "cache", "results.db")
DEFAULT_OUT = os.path.join(_AE_DIR, "figures")

# Publication baseline (mirrors plots/serve.py DEFAULT_RC).
DEFAULT_RC = {
    "figsize_w": 12.5, "figsize_h": 12.5, "dpi": 100,
    "font_size": 42, "font_family": "sans-serif", "font_sans_serif": "DejaVu Sans",
    "font_weight": "normal", "title_size": 42, "label_size": 42,
    "tick_label_size": 42, "ytick_label_size": 42, "legend_size": 32,
    "annotation_size": 20, "line_width": 8, "marker_size": 30,
    "axes_linewidth": 2, "tick_major_width": 2, "tick_major_size": 12,
    "tick_minor_width": 1.0, "tick_minor_size": 4, "grid": False,
    "grid_linewidth": 1, "grid_color": "#e1e1e1", "grid_alpha": 0.7,
}


# Per-chart figure-size overrides. The square 12.5×12.5 default squishes
# multi-panel charts (labels overlap); give side-by-side panels a wide aspect
# and stacked panels more height. Single-panel charts keep the square default.
# 2 panels side by side: wide aspect + smaller ticks/labels so the long
# category names and bar value-labels stop overlapping.
_WIDE2 = {"figsize_w": 26, "figsize_h": 11, "tick_label_size": 27,
          "ytick_label_size": 32, "annotation_size": 16, "legend_size": 28,
          "label_size": 34}
_WIDE3 = {"figsize_w": 38, "figsize_h": 12, "tick_label_size": 26,
          "ytick_label_size": 30, "annotation_size": 15, "legend_size": 28,
          "label_size": 32}
CHART_RC = {
    "bcs_recall": _WIDE2, "bcs_reduction": _WIDE2, "bcs_qps": _WIDE2,
    "bcs_memory": _WIDE2,
    "throughput_and_bar": dict(_WIDE2, figsize_h=12),
    "flexsplit_cluster_size": _WIDE2, "flexsplit_split_overview": _WIDE2,
    "flexsplit_split_effect": _WIDE2, "flexsplit_post_split_cr": _WIDE2,
    "flexsplit_pred_vs_real": _WIDE2, "pred_vs_real_ratio": _WIDE2,
    "modelhub_growth": _WIDE2,
    "bcs_overview": _WIDE3,                                       # 3 panels
    "reduction_combined": dict(_WIDE3, figsize_w=40),            # 3 panels
    "algo_bench_q_proj": {"figsize_w": 14, "figsize_h": 18},     # 2 rows stacked
    "algo_bench_v_proj": {"figsize_w": 14, "figsize_h": 18},
    "rr_qps_comparison": {"figsize_w": 12, "figsize_h": 5},
}

# Bar charts whose x-axis holds long model-family names — rotate the tick labels
# so the three names don't run together.
ROTATE_XTICKS = {"bcs_recall", "bcs_reduction", "bcs_qps", "bcs_memory", "bcs_overview"}


def _rotate_xticks(fig, deg=20):
    for ax in fig.axes:
        labels = ax.get_xticklabels()
        if labels and any(lbl.get_text() for lbl in labels):
            ax.set_xticklabels(labels, rotation=deg, ha="right")


def _apply_rc(rc):
    plt.rcParams.update({
        "figure.figsize": (rc["figsize_w"], rc["figsize_h"]),
        "figure.dpi": rc["dpi"], "font.size": rc["font_size"],
        "font.family": rc["font_family"], "font.sans-serif": [rc["font_sans_serif"]],
        "font.weight": rc["font_weight"], "axes.titlesize": rc["title_size"],
        "axes.labelsize": rc["label_size"], "axes.labelweight": rc["font_weight"],
        "axes.titleweight": rc["font_weight"], "axes.linewidth": rc["axes_linewidth"],
        "xtick.labelsize": rc["tick_label_size"], "ytick.labelsize": rc["ytick_label_size"],
        "xtick.major.width": rc["tick_major_width"], "ytick.major.width": rc["tick_major_width"],
        "xtick.major.size": rc["tick_major_size"], "ytick.major.size": rc["tick_major_size"],
        "lines.linewidth": rc["line_width"], "lines.markersize": rc["marker_size"],
        "legend.fontsize": rc["legend_size"], "axes.grid": rc["grid"],
        "pdf.fonttype": 42,
    })


def main() -> int:
    ap = argparse.ArgumentParser(description="TensorDex AE — render figures from cache")
    ap.add_argument("--db", default=DEFAULT_DB)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--format", default="pdf", choices=["pdf", "png", "svg"])
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--only", default=None, help="comma-separated chart ids")
    args = ap.parse_args()

    if not os.path.exists(args.db):
        print(f"ERROR: cache DB not found at {args.db}\n"
              f"       run `make ae-cache` first (downloads the HF dataset).")
        return 2

    os.environ.setdefault("TENSORDEX_AE_CACHE", os.path.join(_AE_DIR, "cache"))
    from charts import CHARTS, init_db  # noqa: E402  (after sys.path/env setup)
    init_db(args.db)
    os.makedirs(args.out, exist_ok=True)

    ids = list(CHARTS)
    if args.only:
        want = [x.strip() for x in args.only.split(",") if x.strip()]
        ids = [i for i in want if i in CHARTS]
        missing = [i for i in want if i not in CHARTS]
        for m in missing:
            print(f"  ?? unknown chart id: {m}")
        if not ids:
            print("ERROR: none of the requested chart ids exist "
                  "(see ae/FIGURE_MAP.md for the list)")
            return 2

    print(f"Rendering {len(ids)} charts from {args.db}\n")
    ok, skipped = [], []
    for cid in sorted(ids):
        try:
            rc = dict(DEFAULT_RC, **CHART_RC.get(cid, {}))
            _apply_rc(rc)
            fig = CHARTS[cid]["fn"](rc)
            if cid in ROTATE_XTICKS:
                _rotate_xticks(fig)
            path = os.path.join(args.out, f"{cid}.{args.format}")
            fig.savefig(path, format=args.format, dpi=args.dpi,
                        bbox_inches="tight", facecolor="white")
            plt.close(fig)
            ok.append(cid)
            print(f"  ok   {cid}.{args.format}")
        except Exception as e:  # noqa: BLE001 — report & continue
            plt.close("all")
            reason = str(e).splitlines()[0][:80] if str(e) else type(e).__name__
            skipped.append((cid, reason))
            print(f"  SKIP {cid:32} — {reason}")
            if os.environ.get("AE_RENDER_TRACE"):
                traceback.print_exc()

    print(f"\n{'='*60}\nrendered {len(ok)} figures -> {args.out}")
    if skipped:
        print(f"skipped {len(skipped)} (missing staged data or measurement-only):")
        for cid, why in skipped:
            print(f"   - {cid}: {why}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
