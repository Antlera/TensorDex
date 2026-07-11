"""
System-level codec comparison charts.

Charts:
    - codec_throughput: compression vs decompression throughput scatter
    - codec_storage_reduction: normalized storage reduction bar chart
"""

import os
from collections import OrderedDict

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from . import _db as _db_mod
from .model_level_reduction import (
    _sim_load_reductions, _METHOD_STYLES, _BAR_LABELS_MAP, _FULL_GRAY,
    _ZIPLLM_CSV, _METADATA_DB, _CACHE_JSON,
)


# ── Throughput data ───────────────────────────────────────────────────────

METHODS = {
    "OpenZL":         {"comp":  0.74, "decomp": 18.62, "color": "#bc80bd", "marker": "^"},
    "ZipNN":          {"comp":  1.44, "decomp":  9.35, "color": "#b3de69", "marker": "o"},
    "ZipLLM":         {"comp":  5.94, "decomp":  7.98, "color": "#bebada", "marker": "p"},
    "FM-Delta":       {"comp":  0.10, "decomp":  0.10, "color": "#8dd3c7", "marker": "s"},
    "TensorDex":      {"comp": 28.57, "decomp": 58.93, "color": "#fb8072", "marker": "*"},
}

METHOD_ORDER = ["OpenZL", "ZipNN", "ZipLLM", "FM-Delta", "TensorDex"]

# Aggregated ours storage ratio (hardcoded — not from DB)
_TENSORDEX_STORAGE = 0.29


# ── Chart 1: Throughput ───────────────────────────────────────────────────

def chart_codec_throughput(rc):
    """Scatter: X = compression throughput, Y = decompression throughput."""

    MS = rc.get("marker_size", 30)
    ANN = rc.get("tick_label_size", 42) * 0.9

    fig, ax = plt.subplots(figsize=(rc["figsize_w"], rc["figsize_h"]))

    for name in METHOD_ORDER:
        d = METHODS[name]
        is_ours = name == "TensorDex"
        point_s = (MS * 2.0) ** 2 if is_ours else (MS * 1.4) ** 2
        ax.scatter(d["comp"], d["decomp"],
                   s=point_s,
                   c=d["color"], marker=d["marker"],
                   edgecolors="black" if is_ours else "#666",
                   linewidths=2 if is_ours else 1,
                   zorder=10 if is_ours else 5,
                   label=name)

        # TensorDex label on left, others on right
        if name == "TensorDex":
            ox, oy = -1.5, -1.2
            ha = "right"
        elif name == "FM-Delta":
            ox, oy = 1.0, -1.0
            ha = "left"
        elif name == "ZipNN":
            ox, oy = 1.0, 1.0
            ha = "left"
        elif name == "ZipLLM":
            ox, oy = 1.2, -1.5
            ha = "left"
        else:
            ox, oy = 0.6, -0.5
            ha = "left"

        ax.annotate(name, (d["comp"], d["decomp"]),
                    xytext=(d["comp"] + ox, d["decomp"] + oy),
                    fontsize=ANN, ha=ha,
                    fontweight="bold" if is_ours else "normal")

    ax.set_xlabel("Compression Throughput (GB/s)")
    ax.set_ylabel("Decompression Throughput (GB/s)")
    ax.set_xlim(-1.5, max(METHODS[m]["comp"] for m in METHOD_ORDER) * 1.15)
    ax.set_ylim(-3, max(METHODS[m]["decomp"] for m in METHOD_ORDER) * 1.15)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}"))

    # "Better" arrow pointing toward upper-right
    ax.annotate("", xy=(0.85, 0.75), xytext=(0.50, 0.40),
                xycoords="axes fraction", textcoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", color="black", lw=12,
                                mutation_scale=65))
    ax.text(0.60, 0.67, "Better", transform=ax.transAxes,
            fontsize=ANN * 1.3, fontstyle="italic", color="black",
            fontweight="bold", ha="left", va="top", rotation=45)

    fig.tight_layout()
    return fig


# ── Chart 2: Storage Reduction ────────────────────────────────────────────
# Same style as zipllm_real_bar, but replaces TensorDex-FM/TX with TensorDex aggregate.

def chart_codec_storage_reduction(rc):
    """Normalized storage reduction bar chart (Full = 1.0x baseline)."""
    from matplotlib.patches import Patch

    for path in [_ZIPLLM_CSV, _METADATA_DB, _CACHE_JSON]:
        if not os.path.exists(path):
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, f"Data not found:\n{path}",
                    ha="center", va="center", transform=ax.transAxes)
            return fig

    reductions = _sim_load_reductions()

    first_key = list(reductions.keys())[0]
    orig_gb = reductions[first_key]["orig_gb"]

    # Bar order: Full, OpenZL, ZipNN, FM-Delta, ZipLLM, TensorDex
    _bar_order = ["OpenZL", "ZipNN", "ZipLLM", "ZipLLM-Oracle"]
    ordered_keys = [k for k in _bar_order if k in reductions]

    all_labels  = ['Full']
    all_heights = [1.0]
    all_colors  = [_FULL_GRAY]

    # Consistent colors matching throughput scatter
    _BAR_COLOR_MAP = {
        "OpenZL":        "#bc80bd",
        "ZipNN":         "#b3de69",
        "ZipLLM":        "#8dd3c7",   # FM-Delta display
        "ZipLLM-Oracle": "#bebada",   # ZipLLM display
    }

    for key in ordered_keys:
        display = _BAR_LABELS_MAP.get(key, key)
        norm = reductions[key]["stored_gb"] / orig_gb
        all_labels.append(display)
        all_heights.append(norm)
        all_colors.append(_BAR_COLOR_MAP.get(key, _FULL_GRAY))

    # Append TensorDex (hardcoded aggregate bar)
    all_labels.append("TensorDex")
    all_heights.append(_TENSORDEX_STORAGE)
    all_colors.append("#fb8072")

    fig, ax = plt.subplots(figsize=(rc["figsize_w"], rc["figsize_h"]))

    x = range(len(all_labels))
    bars = ax.bar(x, all_heights, color=all_colors, edgecolor='white',
                  linewidth=0.5, width=0.7)

    ann_sz = 42
    for bar, h, label in zip(bars, all_heights, all_labels):
        cx = bar.get_x() + bar.get_width() / 2
        ax.text(cx, 0.05, f'{h:.2f}x',
                ha='center', va='bottom', rotation=90,
                fontsize=ann_sz, color='black',
                fontweight='bold' if label == 'Full' else 'normal')

    ax.set_xticks(list(x))
    ax.set_xticklabels(all_labels, rotation=45, ha="center")
    ax.set_ylabel('Normalized Storage (x)')
    ax.set_ylim(0, 1.18)
    ax.axhline(1.0, color='#aaaaaa', linestyle='--', linewidth=1.5, alpha=0.6, zorder=1)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:.2f}x'))

    # # Legend
    # legend_handles = [Patch(facecolor=_FULL_GRAY, edgecolor='white', label='Full')]
    # for key in ordered_keys:
    #     lbl = _BAR_LABELS_MAP.get(key, key.replace("\n", " "))
    #     legend_handles.append(Patch(facecolor=_BAR_COLOR_MAP.get(key, _FULL_GRAY), edgecolor='white', label=lbl))
    # legend_handles.append(Patch(facecolor="#fb8072", edgecolor='white', label='TensorDex'))
    # ax.legend(handles=legend_handles, loc="upper right", ncol=2,
    #           fontsize=rc["legend_size"], handlelength=1.0, handletextpad=0.4,
    #           columnspacing=0.5)

    fig.tight_layout()
    return fig


# ── Chart Registry ────────────────────────────────────────────────────────

CHARTS = {
    "codec_throughput": {
        "name": "Codec Throughput Comparison",
        "category": "Compression",
        "desc": "Compression & decompression throughput (MB/s) across methods",
        "fn": chart_codec_throughput,
    },
    "codec_storage_reduction": {
        "name": "Codec Storage Reduction",
        "category": "Compression",
        "desc": "Normalized storage ratio (compressed/original) across methods",
        "fn": chart_codec_storage_reduction,
    },
}
