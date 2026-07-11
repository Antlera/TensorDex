"""
Reduction-rate & QPS comparison bar charts.

Charts:
    - rr_qps_comparison: Side-by-side bar charts comparing RR and QPS across methods
"""

import matplotlib.pyplot as plt
import numpy as np


# ── Data ──────────────────────────────────────────────────────────────

METHODS = ["TensorSketch", "Raw+RandomProj", "Raw+SVD", "Raw+PQ"]
REDUCTION_RATIO = [0.5472, 0.1717, 0.1717, 0.1772]
QPS     = [59_026, 9_174, 9_653, 11_659]

# BCS (TensorSketch) = red (#fb8072), others from the palette
BAR_COLORS = ["#fb8072", "#8dd3c7", "#ffffb3", "#bebada"]


# ── Chart function ────────────────────────────────────────────────────

def chart_rr_qps_comparison(rc):
    """Side-by-side bar chart: Reduction Rate and QPS by method."""
    fig, (ax_rr, ax_qps) = plt.subplots(
        1, 2,
        figsize=(rc.get("figsize_w", 10), rc.get("figsize_h", 4.5)),
    )

    x = np.arange(len(METHODS))
    label_size = rc.get("label_size", 12)
    tick_size  = rc.get("tick_label_size", 10)

    legend_size = rc.get("legend_size", 10)

    # ── Left: Reduction Rate ──
    for i, (method, rr, color) in enumerate(zip(METHODS, REDUCTION_RATIO, BAR_COLORS)):
        ax_rr.bar(x[i], rr, color=color, edgecolor="black", linewidth=0.6, label=method)
    ax_rr.set_xticks([])
    ax_rr.set_ylabel("Reduction Ratio", fontsize=label_size)
    ax_rr.set_xlabel("Reduction Ratio", fontsize=label_size)
    ax_rr.set_ylim(0, max(REDUCTION_RATIO) * 1.25)
    for i, v in enumerate(REDUCTION_RATIO):
        ax_rr.text(x[i], v + max(REDUCTION_RATIO) * 0.02, f"{v:.4f}",
                   ha="center", va="bottom", fontsize=tick_size)
    ax_rr.legend(fontsize=legend_size)

    # ── Right: QPS ──
    for i, (method, qps, color) in enumerate(zip(METHODS, QPS, BAR_COLORS)):
        ax_qps.bar(x[i], qps, color=color, edgecolor="black", linewidth=0.6, label=method)
    ax_qps.set_xticks([])
    ax_qps.set_ylabel("QPS", fontsize=label_size)
    ax_qps.set_xlabel("Queries Per Second", fontsize=label_size)
    ax_qps.set_ylim(0, max(QPS) * 1.25)
    for i, v in enumerate(QPS):
        ax_qps.text(x[i], v + max(QPS) * 0.02, f"{v:,}",
                    ha="center", va="bottom", fontsize=tick_size)
    ax_qps.legend(fontsize=legend_size)

    fig.tight_layout()
    return fig


# ── Chart Registry ────────────────────────────────────────────────────

CHARTS = {
    "rr_qps_comparison": {
        "name": "RR & QPS Method Comparison",
        "category": "Bar / Comparison",
        "desc": "Side-by-side bar charts comparing Reduction Rate and QPS across TensorSketch vs baseline methods",
        "fn": chart_rr_qps_comparison,
    },
}
