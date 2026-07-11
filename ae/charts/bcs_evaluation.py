"""
BCS (BitCountSketch) Evaluation Charts.

Visualizes BCS advantages: high recall, fast speed, small memory, zero I/O.
Data from tests/output/multi_family_2026-03-29/EVALUATION_DATA.md.

Charts:
    - bcs_recall: Recall@1 comparison across methods and families
    - bcs_reduction: FlexSplit Reduction Ratio comparison
    - bcs_qps: End-to-end QPS comparison (log scale)
    - bcs_memory: Memory footprint comparison
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── Data from EVALUATION_DATA.md ──────────────────────────────────────

FAMILIES_100 = ["Qwen2.5-7B", "Llama-3.1-8B", "Gemma-2-9b"]
METHODS_4 = ["BCS HNSW", "Raw Tensor", "Direct HNSW", "LSH", "RandProj 2048d"]
METHODS_3 = ["BCS HNSW", "Direct HNSW", "RandProj 2048d"]

# Section 2.1: Recall@1 (100-model)
RECALL_100 = {
    "BCS HNSW":     [1.0000, 1.0000, 1.0000],
    "Direct HNSW":  [0.5990, 0.5992, 0.7330],
    "Raw Tensor":   [1.0000, 1.0000, 1.0000],
    "LSH":          [0.6137, 0.6641, 0.7718],
    "RandProj 2048d":[0.4130, 0.3935, 0.3155],
}

# Section 1.3: Reduction Ratio (100-model, FlexSplit)
REDUCTION_FLEXSPLIT_100 = {
    "BCS HNSW":     [59.9, 50.1, 64.3],
    "Direct HNSW":  [57.4, 48.2, 63.9],
    "LSH":          [59.3, 49.7, 64.1],
    "RandProj 2048d":[38.3, 41.4, 28.7],
}

# Section 1.1: Greedy Reduction Ratio (100-model, initial clustering = 1 - CR)
REDUCTION_GREEDY_100 = {
    "BCS HNSW":     [100*(1-0.4398), 100*(1-0.5773), 100*(1-0.4042)],
    "Direct HNSW":  [100*(1-0.4889), 100*(1-0.5719), 100*(1-0.4681)],
    "LSH":          [100*(1-0.4924), 100*(1-0.5946), 100*(1-0.4273)],
    "RandProj 2048d":[100*(1-0.6834), 100*(1-0.6415), 100*(1-0.7489)],
}

# Section 4.3: E2E QPS (100-model)
QPS_100 = {
    "BCS HNSW":     [25056, 26107, 26990],
    "Direct HNSW":  [44.6,  30.6,  40.6],
    "Raw Tensor":   [2.2,   1.3,   2.2],
    "LSH":          [2.4,   1.4,   2.4],
    "RandProj 2048d":[24312, 25480, 26105],
}

# Section 3.1: Memory per vector (bytes) — INDEX only (downsampled)
MEMORY_PER_VEC_INDEX = {
    "BCS HNSW":      8_448,       # 2048 * 4 + HNSW overhead
    "Direct HNSW":   40_256,      # 10000 * 4 + HNSW overhead
    "LSH":           1_266,       # 10000 / 8 + overhead
    "RandProj 2048d": 8_448,       # 2048 * 4 + HNSW overhead
}

# Per-tensor representation cost (what must be stored/precomputed per tensor)
# BCS: 2048 x int32 fingerprint, precomputed at ingest
# Direct: no precomputation — must load full raw tensor (~37 MB avg BF16)
# LSH: precomputed binary hash (10k bits)
# RandProj: precomputed projection vector (256 x float32)
AVG_TENSOR_BYTES = 37_000_000   # ~37 MB average raw tensor (BF16, Qwen2.5-7B layer)
MEMORY_PER_VEC = {
    "BCS HNSW":      8_192,          # 2048 x 4 bytes (precomputed fingerprint)
    "Direct HNSW":   AVG_TENSOR_BYTES,  # full tensor, no precomputation possible
    "Raw Tensor":    AVG_TENSOR_BYTES,  # full tensor, brute-force scan
    "LSH":           1_250,          # 10000 bits = 1.22 KB (precomputed hash)
    "RandProj 2048d": 8_192,          # 2048 x 4 bytes (precomputed projection)
}

# Section 3.2: Total index memory (MB) at different scales
MEMORY_SCALES = [1_000, 5_000, 10_000, 50_000]
MEMORY_TOTAL = {
    "BCS HNSW":      [8, 40, 81, 404],
    "Direct HNSW":   [38, 192, 384, 1920],
    "LSH":           [1.2, 6, 12, 60],
    "RandProj 2048d": [1.2, 6, 12, 62],
}

# Section 4.4: Index-only QPS (100-model)
INDEX_QPS_100 = {
    "BCS HNSW":     [88527, 95169, 98954],
    "Direct HNSW":  [15662, 14022, 15489],
    "LSH":          [15, 16, 16],
    "RandProj 2048d":[12904, 11127, 10708],
}

# Full-scale data (Section 1.6, 2.2, 4.2)
FAMILIES_FULL = ["Llama-3.1-8B", "Gemma-2-9b", "Llama-3.2-3B"]
FULL_TENSORS = [14277, 3513, 939]

RECALL_FULL = {
    "BCS HNSW":     [1.0000, 1.0000, 1.0000],
    "Direct HNSW":  [0.9551, 0.9294, 0.8285],
    "RandProj 2048d":[0.1394, 0.3074, 0.2726],
}

REDUCTION_FULL = {
    "BCS HNSW":     [71.6, 73.1, 51.4],
    "Direct HNSW":  [71.5, 73.1, 52.5],
    "RandProj 2048d":[15.7, 32.3, 11.4],
}

QPS_FULL = {
    "BCS HNSW":     [0.84, 0.21, 0.04],  # seconds (will convert)
    "Direct HNSW":  [526, 109, 15],
    "RandProj 2048d":[14661, 3711, 594],
}

# Colors
METHOD_COLORS = {
    "BCS HNSW":      "#fb8072",  # salmon (ours)
    "Direct HNSW":   "#80b1d3",  # blue
    "Raw Tensor":    "#fdb462",  # orange
    "LSH":           "#bebada",  # lavender
    "RandProj 2048d": "#d9d9d9",  # gray
}

# Display names for legend / axis labels
METHOD_DISPLAY = {
    "BCS HNSW":      "TensorSketch",
    "Direct HNSW":   "Raw Tensor HNSW",
    "Raw Tensor":    "Bit Distance",
    "LSH":           "LSH",
    "RandProj 2048d": "RandProj 2048d",
}


def _grouped_bar(ax, families, methods, data, colors, ylabel, fmt="%.1f",
                 log=False, annotate=True, ann_fontsize=28, xtick_fontsize=None,
                 ann_rotation=0, ann_inside=False):
    """Draw a grouped bar chart."""
    # Some panels list methods that legitimately lack data (e.g. brute-force
    # "Raw Tensor" has no index to build → absent from the index-QPS dict).
    # Plot only the methods that actually have values.
    methods = [m for m in methods if m in data]
    n_fam = len(families)
    n_meth = len(methods)
    x = np.arange(n_fam)
    width = 0.8 / n_meth

    for i, m in enumerate(methods):
        vals = data[m]
        offset = (i - n_meth / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width * 0.9, label=m,
                      color=colors[m], edgecolor="white", linewidth=0.5)
        if annotate:
            for bar, v in zip(bars, vals):
                if log:
                    txt = f"{v:,.0f}" if v >= 1 else f"{v:.1f}"
                else:
                    txt = fmt % v
                fw = "bold" if m in ("BCS HNSW", "TensorSketch") else "normal"
                if ann_inside and ann_rotation == 90:
                    # Text inside bar, vertical, near top
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() * 0.99,
                            txt, ha="center", va="top", fontsize=ann_fontsize,
                            fontweight=fw, rotation=90, color="black")
                elif ann_rotation == 90:
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height(),
                            " " + txt, ha="center", va="bottom", fontsize=ann_fontsize,
                            fontweight=fw, rotation=90)
                else:
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                            txt, ha="center", va="bottom", fontsize=ann_fontsize,
                            fontweight=fw)

    ax.set_xticks(x)
    ax.set_xticklabels(families, fontsize=xtick_fontsize)
    ax.set_ylabel(ylabel)
    if log:
        ax.set_yscale("log")


# ── Chart 1: Recall@1 ─────────────────────────────────────────────────

def chart_bcs_recall(rc):
    """Recall@1: BCS HNSW achieves perfect recall across all families."""
    FS = rc.get("tick_label_size", 42)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(rc.get("figsize_w", 25), rc.get("figsize_h", 10)))

    # Left: 100-model scale
    _grouped_bar(ax1, FAMILIES_100, METHODS_4, RECALL_100, METHOD_COLORS,
                 "Recall@1", fmt="%.2f", ann_fontsize=FS * 0.5)
    ax1.set_ylim(0, 1.15)
    ax1.set_xlabel("(a) 100-Model Scale", fontsize=FS)
    ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))

    # Right: Full scale
    _grouped_bar(ax2, FAMILIES_FULL, METHODS_3, RECALL_FULL, METHOD_COLORS,
                 "Recall@1", fmt="%.2f", ann_fontsize=FS * 0.5)
    ax2.set_ylim(0, 1.15)
    ax2.set_xlabel("(b) Full Scale", fontsize=FS)
    ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))

    ax1.legend(loc="upper center", bbox_to_anchor=(1.1, 1.15),
               ncol=4, frameon=False, fontsize=FS * 0.6)

    fig.tight_layout()
    return fig


# ── Chart 2: Reduction Ratio ──────────────────────────────────────────

def chart_bcs_reduction(rc):
    """Reduction Ratio: Greedy (initial) and FlexSplit (post-optimization)."""
    FS = rc.get("tick_label_size", 42)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(rc.get("figsize_w", 25), rc.get("figsize_h", 10)))

    # Left: Greedy (100-model)
    _grouped_bar(ax1, FAMILIES_100, METHODS_4, REDUCTION_GREEDY_100, METHOD_COLORS,
                 "Reduction Ratio (%)", fmt="%.0f%%", ann_fontsize=FS * 0.5)
    ax1.set_ylim(0, 75)
    ax1.set_xlabel("(a) Greedy Clustering (100-Model)", fontsize=FS)
    ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))

    # Right: Full scale
    _grouped_bar(ax2, FAMILIES_FULL, METHODS_3, REDUCTION_FULL, METHOD_COLORS,
                 "Reduction Ratio (%)", fmt="%.1f%%", ann_fontsize=FS * 0.5)
    ax2.set_ylim(0, 90)
    ax2.set_xlabel("(b) Full Scale", fontsize=FS)
    ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))

    ax1.legend(loc="upper center", bbox_to_anchor=(1.1, 1.15),
               ncol=4, frameon=False, fontsize=FS * 0.6)

    fig.tight_layout()
    return fig


# ── Chart 3: QPS (speed) ──────────────────────────────────────────────

def chart_bcs_qps(rc):
    """E2E QPS: BCS HNSW is 100-10000x faster (log scale)."""
    FS = rc.get("tick_label_size", 42)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(rc.get("figsize_w", 25), rc.get("figsize_h", 10)))

    # Left: E2E QPS (100-model)
    _grouped_bar(ax1, FAMILIES_100, METHODS_4, QPS_100, METHOD_COLORS,
                 "Queries per Second", log=True, ann_fontsize=FS * 0.45)
    ax1.set_xlabel("(a) End-to-End QPS (100-model)", fontsize=FS)

    # Right: Index-only QPS (100-model)
    _grouped_bar(ax2, FAMILIES_100, METHODS_4, INDEX_QPS_100, METHOD_COLORS,
                 "Queries per Second", log=True, ann_fontsize=FS * 0.45)
    ax2.set_xlabel("(b) Index-Only QPS (100-model)", fontsize=FS)

    ax1.legend(loc="upper center", bbox_to_anchor=(1.1, 1.15),
               ncol=4, frameon=False, fontsize=FS * 0.6)

    fig.tight_layout()
    return fig


# ── Chart 4: Memory ───────────────────────────────────────────────────

def chart_bcs_memory(rc):
    """Memory footprint: BCS HNSW balances size and recall."""
    FS = rc.get("tick_label_size", 42)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(rc.get("figsize_w", 25), rc.get("figsize_h", 10)))

    # Left: Per-vector memory
    methods = list(MEMORY_PER_VEC.keys())
    mem_vals = [MEMORY_PER_VEC[m] / 1024 for m in methods]  # KB
    colors = [METHOD_COLORS[m] for m in methods]
    bars = ax1.bar(methods, mem_vals, color=colors, edgecolor="white", linewidth=0.5)
    for bar, v in zip(bars, mem_vals):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 f"{v:.1f} KB", ha="center", va="bottom",
                 fontsize=FS * 0.6, fontweight="bold")
    ax1.set_ylabel("Memory per Vector (KB)")
    ax1.set_xlabel("(a) Per-Vector Memory", fontsize=FS)

    # Right: Total memory at scale
    x = np.arange(len(MEMORY_SCALES))
    scale_labels = [f"{s//1000}k" for s in MEMORY_SCALES]
    n_meth = len(MEMORY_TOTAL)
    width = 0.8 / n_meth
    for i, (m, vals) in enumerate(MEMORY_TOTAL.items()):
        offset = (i - n_meth / 2 + 0.5) * width
        ax2.bar(x + offset, vals, width * 0.9, label=m,
                color=METHOD_COLORS[m], edgecolor="white", linewidth=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(scale_labels)
    ax2.set_ylabel("Total Index Memory (MB)")
    ax2.set_xlabel("(b) Total Memory by Scale", fontsize=FS)
    ax2.set_yscale("log")
    ax2.legend(loc="upper left", frameon=False, fontsize=FS * 0.6)

    fig.tight_layout()
    return fig


# ── Chart 5: Combined 4-panel (100-model scale) ─────────────────────

def chart_bcs_overview(rc):
    """Combined 3-panel (horizontal): Recall, QPS, Memory (100-model scale)."""
    FS = rc.get("tick_label_size", 42)
    LS = rc.get("legend_size", 32)
    LBL = rc.get("label_size", 42)
    ANN = rc.get("annotation_size", FS * 0.45)
    XTFS = FS
    fig, axes = plt.subplots(1, 3, figsize=(rc.get("figsize_w", 36), rc.get("figsize_h", 12)))

    BANN = FS * 0.95

    # (a) Recall@1 — annotations inside bars (vertical)
    ax = axes[0]
    _grouped_bar(ax, FAMILIES_100, METHODS_4, RECALL_100, METHOD_COLORS,
                 "Recall@1", fmt="%.2f", ann_fontsize=BANN, xtick_fontsize=XTFS,
                 ann_rotation=90, ann_inside=True)
    ax.set_ylim(0, 1.15)
    ax.set_xlabel("(a) Recall@1", fontsize=LBL, labelpad=10)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))

    # (b) E2E QPS (log) — annotations inside bars (vertical) + speedup vs Bit Distance
    ax = axes[1]
    _grouped_bar(ax, FAMILIES_100, METHODS_4, QPS_100, METHOD_COLORS,
                 "Queries per Second", log=True, ann_fontsize=BANN, xtick_fontsize=XTFS,
                 ann_rotation=90, ann_inside=True)
    ax.set_ylim(bottom=0.5, top=ax.get_ylim()[1] * 50)
    # # Add speedup vs Bit Distance for TensorSketch (BCS HNSW) only
    # bt_qps = QPS_100["Raw Tensor"]
    # n_fam = len(FAMILIES_100)
    # n_meth = len(METHODS_4)
    # width = 0.8 / n_meth
    # mi_bcs = METHODS_4.index("BCS HNSW")
    # for fi in range(n_fam):
    #     qps_val = QPS_100["BCS HNSW"][fi]
    #     speedup = qps_val / bt_qps[fi]
    #     txt = f"{speedup:.0f}x\nFaster"
    #     offset = (mi_bcs - n_meth / 2 + 0.5) * width
    #     ax.text(fi + offset, qps_val * 1.3, txt,
    #             ha="center", va="bottom", fontsize=FS * 1.0,
    #             color=METHOD_COLORS["BCS HNSW"], fontweight="bold")
    ax.set_xlabel("(b) End-to-End QPS", fontsize=LBL, labelpad=10)

    # (c) Per-query data footprint (log scale)
    ax = axes[2]
    methods = [m for m in METHODS_4 if m in MEMORY_PER_VEC]
    mem_kb = [MEMORY_PER_VEC[m] / 1024 for m in methods]  # KB for log scale
    bar_colors = [METHOD_COLORS[m] for m in methods]
    bars = ax.bar(range(len(methods)), mem_kb, color=bar_colors, edgecolor="white", linewidth=0.5)
    DANN = FS * 1.0
    for bar, v, m in zip(bars, mem_kb, methods):
        if v >= 1024:
            label = f"{v/1024:.0f} MB"
        else:
            label = f"{v:.1f} KB"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.15,
                label, ha="center", va="bottom",
                fontsize=DANN, fontweight="bold")
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels([" "] * len(methods), fontsize=XTFS)
    ax.set_ylabel("Avg. Query IO")
    ax.set_yscale("log")
    ax.set_ylim(top=ax.get_ylim()[1] * 80)
    ax.set_xlabel("(c) Average Per-Tensor Query IO", fontsize=LBL, labelpad=10)
    # # Add IO reduction vs Bit Distance for TensorSketch (BCS HNSW) only
    # bt_io = MEMORY_PER_VEC["Raw Tensor"] / 1024  # KB
    # for bar, v, m in zip(bars, mem_kb, methods):
    #     if m != "BCS HNSW":
    #         continue
    #     reduction = bt_io / v
    #     txt = f"{reduction:.0f}x\nLess IO"
    #     ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 3.5,
    #             txt, ha="center", va="bottom",
    #             fontsize=FS * 1.0, color=METHOD_COLORS["BCS HNSW"], fontweight="bold")

    # Shared legend at top
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=METHOD_COLORS[m], edgecolor="white")
               for m in METHODS_4]
    display_names = [METHOD_DISPLAY.get(m, m) for m in METHODS_4]
    fig.legend(handles, display_names, loc="upper center",
               bbox_to_anchor=(0.5, 1.0), ncol=len(METHODS_4),
               frameon=False, fontsize=LS)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    plt.subplots_adjust(wspace=0.15)
    return fig


# ── Chart Registry ────────────────────────────────────────────────────

CHARTS = {
    "bcs_recall": {
        "name": "BCS Recall@1",
        "category": "BCS Evaluation",
        "desc": "Recall@1 comparison: BCS HNSW vs Direct HNSW, LSH, RandProj",
        "fn": chart_bcs_recall,
    },
    "bcs_reduction": {
        "name": "BCS Reduction Ratio",
        "category": "BCS Evaluation",
        "desc": "FlexSplit Reduction Ratio comparison across methods",
        "fn": chart_bcs_reduction,
    },
    "bcs_qps": {
        "name": "BCS QPS",
        "category": "BCS Evaluation",
        "desc": "End-to-end and index-only QPS comparison (log scale)",
        "fn": chart_bcs_qps,
    },
    "bcs_memory": {
        "name": "BCS Memory",
        "category": "BCS Evaluation",
        "desc": "Memory footprint comparison: per-vector and total at scale",
        "fn": chart_bcs_memory,
    },
    "bcs_overview": {
        "name": "BCS Overview (4-panel)",
        "category": "BCS Evaluation",
        "desc": "Combined view: Recall, Reduction Ratio, QPS, Memory (100-model scale)",
        "fn": chart_bcs_overview,
    },
}
