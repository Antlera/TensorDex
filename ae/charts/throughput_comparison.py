"""
Compression & Decompression Throughput Scatter Plot.

Data: throughput with 192 threads, all data in memory (no storage I/O).
X-axis: compression speed, Y-axis: decompression speed.

Charts:
    - throughput_comparison: scatter plot, each method as a point
"""

import numpy as np
import matplotlib.pyplot as plt


# ── Data ─────────────────────────────────────────────────────────────────

METHODS = {
    "ZipNN":          {"comp": 1473,  "decomp": 9575,  "color": "#b0b0b0", "marker": "o"},
    "ZipLLM":         {"comp": 5942,  "decomp": 7981,  "color": "#80b1d3", "marker": "s"},
    "OpenZL":         {"comp": 753,   "decomp": 19068, "color": "#bebada", "marker": "^"},
    "TensorDex-FM++": {"comp": 9821,  "decomp": 8494,  "color": "#fb8072", "marker": "D"},
    "TensorDex-TX":   {"comp": 22916, "decomp": 28393, "color": "#e31a1c", "marker": "*"},
}


# ── Chart ────────────────────────────────────────────────────────────────

def chart_throughput(rc):
    """Compression vs decompression throughput scatter."""
    FS = rc.get("tick_label_size", 42)
    MS = rc.get("marker_size", 30)
    ANN = FS * 0.55

    fig, ax = plt.subplots(figsize=(rc.get("figsize_w", 14),
                                    rc.get("figsize_h", 12)))

    for name, d in METHODS.items():
        is_ours = "TensorDex" in name
        ax.scatter(d["comp"], d["decomp"],
                   s=MS ** 2 * (1.5 if is_ours else 1),
                   c=d["color"], marker=d["marker"],
                   edgecolors="black" if is_ours else "#666",
                   linewidths=2 if is_ours else 1,
                   zorder=10 if is_ours else 5,
                   label=name)

        # Label offset
        ox, oy = 0, 0
        if name == "TensorDex-TX":
            ox, oy = -800, -2200
        elif name == "TensorDex-FM++":
            ox, oy = 600, -1800
        elif name == "OpenZL":
            ox, oy = 600, -500
        elif name == "ZipNN":
            ox, oy = 600, -500
        elif name == "ZipLLM":
            ox, oy = 600, -500

        ax.annotate(name, (d["comp"], d["decomp"]),
                    xytext=(d["comp"] + ox, d["decomp"] + oy),
                    fontsize=ANN,
                    fontweight="bold" if is_ours else "normal")

    ax.set_xlabel("Compression Throughput (MB/s)")
    ax.set_ylabel("Decompression Throughput (MB/s)")
    ax.set_xlim(0, max(d["comp"] for d in METHODS.values()) * 1.15)
    ax.set_ylim(0, max(d["decomp"] for d in METHODS.values()) * 1.15)

    # Format tick labels with commas
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:,.0f}"))

    fig.tight_layout()
    return fig


# ── Registry ─────────────────────────────────────────────────────────────

CHARTS = {
    "throughput_comparison": {
        "name": "Throughput Comparison",
        "category": "Performance",
        "desc": "Compression vs decompression throughput scatter (MB/s, 192 threads, in-memory)",
        "fn": chart_throughput,
    },
}
