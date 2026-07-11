"""
Model Hub Growth — Cumulative size & count for base vs fine-tuned models.

Charts:
    - modelhub_growth: Stacked area chart with two subplots (size & count)
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path
from collections import defaultdict

from ._root import PROJECT_ROOT as _PROJECT_ROOT
_BASE_JSON = str(_PROJECT_ROOT / "model_hub_crawl" / "base_monthly_stats.json")
_TYPE_JSON = str(_PROJECT_ROOT / "model_hub_crawl" / "model_type_monthly_stats.json")


def _load_monthly(json_path, key=None):
    """Load monthly_data from JSON. If key is given, use data[key]; else sum all keys."""
    with open(json_path) as f:
        data = json.load(f)

    merged = defaultdict(lambda: {"count": 0, "size": 0})
    if key:
        for month, vals in data[key]["monthly_data"].items():
            merged[month]["count"] += vals["count"]
            merged[month]["size"] += vals["size"]
    else:
        for k, v in data.items():
            for month, vals in v["monthly_data"].items():
                merged[month]["count"] += vals["count"]
                merged[month]["size"] += vals["size"]

    months = sorted(merged.keys())
    counts = np.array([merged[m]["count"] for m in months])
    sizes = np.array([merged[m]["size"] for m in months])
    return months, np.cumsum(counts), np.cumsum(sizes)


def _month_to_year(months):
    return np.array([int(m[:4]) + (int(m[5:7]) - 1) / 12 for m in months])


def _align_series(base_months, base_vals, ft_months, ft_vals):
    """Align base and fine-tuned series to a common month grid."""
    all_months = sorted(set(base_months) | set(ft_months))
    x = _month_to_year(all_months)

    base_map = dict(zip(base_months, base_vals))
    ft_map = dict(zip(ft_months, ft_vals))

    base_aligned = np.zeros(len(all_months))
    ft_aligned = np.zeros(len(all_months))

    last_b, last_f = 0.0, 0.0
    for i, m in enumerate(all_months):
        last_b = base_map.get(m, last_b)
        last_f = ft_map.get(m, last_f)
        base_aligned[i] = last_b
        ft_aligned[i] = last_f

    return x, base_aligned, ft_aligned


def chart_modelhub_growth(rc):
    """Stacked area: cumulative size (TB) and count (K) for base vs fine-tuned."""
    base_months, base_cum_count, base_cum_size = _load_monthly(_BASE_JSON, key="base")
    ft_months, ft_cum_count, ft_cum_size = _load_monthly(_TYPE_JSON)

    # Align to common time axis
    x, base_tb, ft_tb = _align_series(
        base_months, base_cum_size / 1e12, ft_months, ft_cum_size / 1e12)
    _, base_k, ft_k = _align_series(
        base_months, base_cum_count / 1e3, ft_months, ft_cum_count / 1e3)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(rc["figsize_w"], rc["figsize_h"]))

    color_base = "#d76b5a"
    color_ft = "#5680eb"
    lw = rc["line_width"]
    fs = rc.get("label_size", 42)

    # ── Left: Cumulative Size (TB) — log scale ──
    total_tb = base_tb + ft_tb
    ax1.plot(x, total_tb, color=color_ft, linewidth=0.5)
    ax1.plot(x, ft_tb, color=color_ft, linewidth=0.5, linestyle="--")
    ax1.fill_between(x, ft_tb, total_tb, color=color_base, label="Base")
    ax1.fill_between(x, 0, ft_tb, color=color_ft, label="Fine-tuned")
    ax1.set_ylim(bottom=-20)
    ax1.set_ylabel("Cumulative Size (TB)", fontsize=fs)
    ax1.set_xlabel("Year", fontsize=fs)
    ax1.tick_params(axis="both", labelsize=rc["tick_label_size"])
    ax1.legend(loc="upper left", fontsize=rc["legend_size"],
               handlelength=1.0, handletextpad=0.4)

    # ── Right: Cumulative Count (K) — log scale ──
    total_k = base_k + ft_k
    ax2.plot(x, total_k, color=color_ft, linewidth=0.5)
    ax2.plot(x, ft_k, color=color_ft, linewidth=0.5, linestyle="--")
    ax2.fill_between(x, ft_k, total_k, color=color_base, label="Base")
    ax2.fill_between(x, 0, ft_k, color=color_ft, label="Fine-tuned")
    ax2.set_ylim(bottom=-5)
    ax2.set_ylabel("Cumulative Count (K)", fontsize=fs)
    ax2.set_xlabel("Year", fontsize=fs)
    ax2.tick_params(axis="both", labelsize=rc["tick_label_size"])
    ax2.legend(loc="upper left", fontsize=rc["legend_size"],
               handlelength=1.0, handletextpad=0.4)

    # ── Time marker annotations ──
    ann_sz = rc["legend_size"] * 0.8
    time_markers = [2021, 2023, 2025]
    for tm in time_markers:
        # Find closest index
        idx = np.argmin(np.abs(x - tm))
        total_tb_val = base_tb[idx] + ft_tb[idx]
        total_k_val = base_k[idx] + ft_k[idx]
        ft_tb_pct = ft_tb[idx] / total_tb_val * 100 if total_tb_val > 0 else 0
        ft_k_pct = ft_k[idx] / total_k_val * 100 if total_k_val > 0 else 0

        # Vertical lines
        for ax in (ax1, ax2):
            ax.axvline(tm, color="#aaaaaa", linestyle="--", linewidth=1.5, alpha=0.6, zorder=1)

        # Left: size ratio
        max_tb = max(base_tb + ft_tb)
        ax1.text(tm - 0.05, total_tb_val + max_tb * 0.03,
                 f"FT {ft_tb_pct:.1f}%",
                 fontsize=ann_sz, color=color_ft, fontweight="bold", ha="right")
        ax1.text(tm - 0.05, total_tb_val + max_tb * 0.10,
                 f"{total_tb_val:.0f} TB",
                 fontsize=ann_sz * 0.85, color="#333333", ha="right")

        # Right: count ratio
        max_k = max(base_k + ft_k)
        ax2.text(tm - 0.05, total_k_val + max_k * 0.03,
                 f"FT {ft_k_pct:.1f}%",
                 fontsize=ann_sz, color=color_ft, fontweight="bold", ha="right")
        ax2.text(tm - 0.05, total_k_val + max_k * 0.10,
                 f"{total_k_val:.0f}K",
                 fontsize=ann_sz * 0.85, color="#333333", ha="right")

    # ── Zoom-in inset: show the FT/Base boundary around 2023 ──
    idx_2023 = np.argmin(np.abs(x - 2023))
    ft_val = ft_tb[idx_2023]
    total_val = (base_tb + ft_tb)[idx_2023]
    margin = (total_val - ft_val) * 1.5

    axins = ax1.inset_axes([0.16, 0.35, 0.3, 0.3])
    axins.fill_between(x, ft_tb, base_tb + ft_tb, color=color_base)
    axins.fill_between(x, 0, ft_tb, color=color_ft)
    axins.set_xlim(2022.7, 2023.3)
    axins.set_ylim(ft_val - margin, total_val + margin)
    axins.xaxis.set_major_locator(mticker.FixedLocator([2023]))
    axins.yaxis.set_major_locator(mticker.MaxNLocator(3))
    axins.tick_params(labelsize=rc["legend_size"] * 0.7)
    axins.set_ylabel("TB", fontsize=rc["legend_size"] * 0.7)
    ax1.indicate_inset_zoom(axins, edgecolor="#666666", linewidth=1.5)

    # ── Zoom-in inset on right chart ──
    ft_val_k = ft_k[idx_2023]
    total_val_k = (base_k + ft_k)[idx_2023]
    margin_k = (total_val_k - ft_val_k) * 1.5

    axins2 = ax2.inset_axes([0.18, 0.35, 0.3, 0.3])
    axins2.fill_between(x, ft_k, base_k + ft_k, color=color_base)
    axins2.fill_between(x, 0, ft_k, color=color_ft)
    axins2.set_xlim(2022.7, 2023.3)
    axins2.set_ylim(ft_val_k - margin_k, total_val_k + margin_k)
    axins2.xaxis.set_major_locator(mticker.FixedLocator([2023]))
    axins2.yaxis.set_major_locator(mticker.MaxNLocator(3, integer=True))
    axins2.tick_params(labelsize=rc["legend_size"] * 0.7)
    axins2.set_ylabel("K", fontsize=rc["legend_size"] * 0.7)
    rect_patch, connectors = ax2.indicate_inset_zoom(axins2, edgecolor="#666666", linewidth=1.5)
    # Fix connector visibility: show lines 0 (top-left) and 3 (bottom-right)
    for c in connectors:
        c.set_visible(False)
    connectors[0].set_visible(True)  # top-left to top-left
    connectors[3].set_visible(True)  # bottom-right to bottom-right

    # Force x-axis ticks at integer years
    for ax in (ax1, ax2):
        ax.set_xticks([2019, 2021, 2023, 2025])
        ax.set_xticklabels(["2019", "2021", "2023", "2025"])
        ax.set_xlim(2019, 2025.5)

    fig.tight_layout()
    return fig


CHARTS = {
    "modelhub_growth": {
        "name": "Model Hub Growth",
        "category": "Model Hub",
        "desc": "Stacked area: cumulative size (TB) and count (K) for base vs fine-tuned models",
        "fn": chart_modelhub_growth,
    },
}
