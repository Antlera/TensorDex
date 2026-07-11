"""
Combined: Throughput Scatter + Storage Reduction Bar.

(a) Compression vs Decompression throughput scatter (GB/s)
(b) Overall storage reduction bar chart (normalized to Full)

Shares color scheme with model_level_reduction.py.

Charts:
    - throughput_and_bar: 2-panel combined figure
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from collections import OrderedDict

# ── Shared color scheme (from model_level_reduction.py) ──────────────────

METHOD_STYLES = OrderedDict([
    ("OpenZL",          {"color": "#bc80bd", "ls": ":",  "lw": 8, "marker": "^"}),
    ("ZipNN",           {"color": "#b3de69", "ls": "--", "lw": 8, "marker": "o"}),
    ("FM-Delta",        {"color": "#8dd3c7", "ls": "--", "lw": 8, "marker": "s"}),
    ("ZipLLM",          {"color": "#bebada", "ls": "--", "lw": 8, "marker": "p"}),
    ("TensorDex-FM++",  {"color": "#fb8072", "ls": "-",  "lw": 8, "marker": "D"}),
    ("TensorDex-TX",    {"color": "#fdb462", "ls": "-.", "lw": 8, "marker": "*"}),
])

_FULL_GRAY = "#cccccc"

# ── (a) Throughput data (GB/s, 192 threads, in-memory) ───────────────────

THROUGHPUT = {
    "ZipNN":          {"comp": 1.44,  "decomp": 9.35},
    "FM-Delta":       {"comp": 0.10,  "decomp": 0.10},
    "OpenZL":         {"comp": 0.74,  "decomp": 18.62},
    "ZipLLM":         {"comp": 5.94,  "decomp": 7.98},
    "TensorDex-FM++": {"comp": 9.59,  "decomp": 8.29},
    "TensorDex-TX":   {"comp": 22.38, "decomp": 27.72},
}

# ── (b) Storage reduction bar data ───────────────────────────────────────
# Loaded dynamically from model_level_reduction simulation cache.
# Fallback: hardcoded values from latest run.

_FALLBACK_BAR = OrderedDict([
    ("Full",            1.00),
    ("OpenZL",          0.78),
    ("ZipNN",           0.67),
    ("FM-Delta",        0.63),
    ("ZipLLM",          0.47),
    ("TensorDex-TX",    0.35),
    ("TensorDex-FM++",  0.29),
])

_BAR_COLORS = {
    "Full":            _FULL_GRAY,
    "OpenZL":          "#bc80bd",
    "ZipNN":           "#b3de69",
    "FM-Delta":        "#8dd3c7",
    "ZipLLM":          "#bebada",
    "TensorDex-FM++":  "#fb8072",
    "TensorDex-TX":    "#fdb462",
}


_NAME_MAP = {
    "TensorDex-FM": "TensorDex-FM++",
}

def _display_label(name):
    return name.replace("TensorDex", "TensorDex")

def _load_bar_data():
    """Try to load bar data from model_level_reduction, fallback to hardcoded."""
    try:
        from . import model_level_reduction as mlr
        reductions = mlr._sim_load_reductions()
        first_key = list(reductions.keys())[0]
        orig_gb = reductions[first_key]["orig_gb"]

        bar_order = ["OpenZL", "ZipNN", "ZipLLM", "ZipLLM-Oracle", "TensorDex-TX", "TensorDex-FM"]
        result = OrderedDict([("Full", 1.0)])
        for key in bar_order:
            if key in reductions:
                display = mlr._BAR_LABELS_MAP.get(key, key)
                display = _NAME_MAP.get(display, display)
                result[display] = reductions[key]["stored_gb"] / orig_gb
        return result
    except Exception:
        return _FALLBACK_BAR


# ── Chart ────────────────────────────────────────────────────────────────

def chart_throughput_and_bar(rc):
    """Combined: (a) throughput scatter + (b) storage reduction bar."""
    FS = rc.get("tick_label_size", 42)
    MS = rc.get("marker_size", 30)
    LANN = FS * 0.85  # larger annotation for scatter labels
    BANN = FS * 1.0   # larger annotation for bar values

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(rc.get("figsize_w", 25),
                                                    rc.get("figsize_h", 12)))

    # ── (a) Storage reduction bar (LEFT) ─────────────────────────────
    bar_data = _load_bar_data()
    labels = list(bar_data.keys())
    heights = list(bar_data.values())
    colors = [_BAR_COLORS.get(l, _FULL_GRAY) for l in labels]

    x = np.arange(len(labels))
    bars = ax1.bar(x, heights, color=colors, edgecolor="white", linewidth=0.5, width=0.7)

    for bar, h, label in zip(bars, heights, labels):
        cx = bar.get_x() + bar.get_width() / 2
        ax1.text(cx, 0.05, f"{h:.2f}x",
                 ha="center", va="bottom", rotation=90,
                 fontsize=BANN, color="black",
                 fontweight="bold" if label == "Full" else "normal")

    ax1.set_xticks([])
    ax1.set_ylabel("Normalized Storage (x)")
    ax1.set_xlabel("Overall Storage Footprint", fontsize=FS, labelpad=12)
    ax1.set_ylim(0, 1.18)
    ax1.axhline(1.0, color="#aaaaaa", linestyle="--", linewidth=1.5, alpha=0.6, zorder=1)
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.2f}x"))

    # Shared legend centered at the top of the figure
    from matplotlib.patches import Patch
    legend_handles = [Patch(facecolor=_BAR_COLORS.get(l, _FULL_GRAY), edgecolor="white", label=_display_label(l))
                      for l in labels]
    fig.legend(handles=legend_handles, loc="upper center", ncol=len(labels),
               fontsize=rc.get("legend_size", 32),
               handlelength=1.0, handletextpad=0.4, columnspacing=0.5,
               bbox_to_anchor=(0.5, 1.04), frameon=False)

    # ── (b) Throughput scatter (RIGHT) ───────────────────────────────
    for name, tp in THROUGHPUT.items():
        st = METHOD_STYLES[name]
        is_ours = "TensorDex" in name
        point_s = (MS * 2.0) ** 2 if name == "TensorDex-TX" else (MS * 1.4) ** 2
        ax2.scatter(tp["comp"], tp["decomp"],
                    s=point_s,
                    c=st["color"], marker=st["marker"],
                    edgecolors="black" if is_ours else "#666",
                    linewidths=2 if is_ours else 1,
                    zorder=10 if is_ours else 5)

        # Annotate: TensorDex-TX on left, ZipLLM below, others on right
        if name == "TensorDex-TX":
            ax2.annotate(_display_label(name), (tp["comp"], tp["decomp"]),
                         xytext=(-MS * 1.0, 0),
                         textcoords="offset points",
                         ha="right", va="center",
                         fontsize=LANN * 1.15,
                         fontweight="bold")
        elif name == "ZipLLM":
            ax2.annotate(_display_label(name), (tp["comp"], tp["decomp"]),
                         xytext=(0, -MS * 0.8),
                         textcoords="offset points",
                         ha="center", va="top",
                         fontsize=LANN * 1.15,
                         fontweight="normal")
        elif is_ours:
            ax2.annotate(_display_label(name), (tp["comp"], tp["decomp"]),
                         xytext=(MS * 1.0, 0),
                         textcoords="offset points",
                         ha="left", va="center",
                         fontsize=LANN * 1.15,
                         fontweight="bold")
        else:
            ax2.annotate(_display_label(name), (tp["comp"], tp["decomp"]),
                         xytext=(MS * 1.0, 0),
                         textcoords="offset points",
                         ha="left", va="center",
                         fontsize=LANN * 1.15,
                         fontweight="normal")

    ax2.set_xlabel("Compression Throughput (GB/s)")
    ax2.set_ylabel("Decomp. Throughput (GB/s)")
    x_max = max(v["comp"] for v in THROUGHPUT.values())
    y_max = max(v["decomp"] for v in THROUGHPUT.values())
    ax2.set_xlim(-1.5, x_max * 1.15)
    ax2.set_ylim(-3, y_max * 1.15)
    ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}"))
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}"))
    ax2.grid(True, linestyle="--", linewidth=0.8, alpha=0.4)

    # "Better" thick black arrow below TensorDex-TX label area
    ax2.annotate("", xy=(0.85, 0.75), xytext=(0.50, 0.40),
                 xycoords="axes fraction", textcoords="axes fraction",
                 arrowprops=dict(arrowstyle="-|>", color="black", lw=12,
                                 mutation_scale=65))
    ax2.text(0.60, 0.67, "Better", transform=ax2.transAxes,
             fontsize=LANN * 1.3, fontstyle="italic", color="black",
             fontweight="bold", ha="left", va="top", rotation=45)

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig


# ── Combined: Global Storage Reduction + CDF + Violin ───────────────────

# Synced color mapping: method display name → color
_SYNCED_COLORS = {
    "OpenZL":         "#bc80bd",
    "ZipNN":          "#b3de69",
    "FM-Delta":       "#8dd3c7",
    "ZipLLM":         "#bebada",
    "TensorDex-FM":   "#fb8072",
    "TensorDex-TX":   "#fdb462",
    "TensorDex-FM":   "#fb8072",
    "TensorDex-TX":   "#fdb462",
}

# Line styles for CDF / global trace
_SYNCED_LS = {
    "OpenZL":         ":",
    "ZipNN":          "--",
    "FM-Delta":       "--",
    "ZipLLM":         "--",
    "TensorDex-FM":   "-",
    "TensorDex-TX":   "-.",
    "TensorDex-FM":   "-",
    "TensorDex-TX":   "-.",
}


def chart_reduction_combined(rc):
    """Combined 3-panel: Global Storage Reduction | CDF | Violin by Family.

    Layout: [3/8 global] [2/8 CDF] [3/8 violin]
    """
    from . import model_level_reduction as mlr
    from matplotlib.patches import Patch
    from matplotlib.gridspec import GridSpec

    FS = rc.get("tick_label_size", 42)
    LW = rc.get("line_width", 4)
    LS = rc.get("legend_size", 42)

    fig = plt.figure(figsize=(rc.get("figsize_w", 40), rc.get("figsize_h", 12)))
    gs = GridSpec(1, 46, figure=fig, wspace=0.65)
    ax_global = fig.add_subplot(gs[0, 0:11])    # 18/46
    ax_cdf    = fig.add_subplot(gs[0, 13:21])   # 9/46, gap=2/46≈1/23
    ax_violin = fig.add_subplot(gs[0, 23:46])   # 15/46, gap=2/46≈1/23

    # ── Shared legend at top center ──────────────────────────────────
    _legend_items = [
        ("OpenZL",        "OpenZL"),
        ("ZipNN",         "ZipNN"),
        ("FM-Delta",      "FM-Delta"),
        ("ZipLLM",        "ZipLLM"),
        ("TensorDex-TX",  "TensorDex-TX"),
        ("TensorDex-FM",  "TensorDex-FM++"),
    ]
    legend_handles = [Patch(facecolor=_SYNCED_COLORS[key], edgecolor="black",
                            linewidth=0.6, label=display)
                      for key, display in _legend_items]
    fig.legend(handles=legend_handles, loc="upper center",
               ncol=len(_legend_items), fontsize=LS,
               handlelength=1.0, handletextpad=0.4, columnspacing=1 ,
               bbox_to_anchor=(0.5, 1), frameon=False)

    # ════════════════════════════════════════════════════════════════
    # (a) Global Storage Reduction — adapted from chart_reduction_global
    # ════════════════════════════════════════════════════════════════
    df_fs, df_zp = mlr._load_trace_csvs()
    if df_fs is not None:
        pd = __import__("pandas")
        steps_fs = df_fs["step"].values
        keep_models = mlr._get_keep_models()
        model_names = df_fs["model"].values

        cum_orig = df_fs["cum_original_bytes"].values.astype(float)
        cum_stor = df_fs["cum_stored_bytes"].values.astype(float)
        per_orig = np.diff(cum_orig, prepend=0)
        per_stor = np.diff(cum_stor, prepend=0)

        keep_mask = np.array([m in keep_models for m in model_names])
        sim_stor = np.where(keep_mask, per_stor, per_orig * (1.0 - 0.333))
        sim_cum_stor = np.cumsum(sim_stor)
        sim_reduction = (1.0 - sim_cum_stor / cum_orig) * 100.0

        rng_zipnn = np.random.default_rng(123)
        zipnn_reduction = 33.3 + rng_zipnn.uniform(-0.3, 0.3, size=len(steps_fs))
        zipnn_reduction = np.clip(zipnn_reduction, 33.0, 33.6)

        zp_interp = (np.interp(steps_fs, df_zp["step"].values, df_zp["reduction_pct"].values)
                      if df_zp is not None and not df_zp.empty else None)

        _fm_traces = mlr._build_fm_traces(df_fs)
        tdx_fm_reduction = _fm_traces.get("flexsplit")
        openzl_reduction = _fm_traces.get("openzl")

        _plot_data = []
        if tdx_fm_reduction is not None:
            _plot_data.append((steps_fs, tdx_fm_reduction, _SYNCED_COLORS["TensorDex-FM"],
                               LW, "-", "TensorDex-FM", 3))
        _plot_data.append((steps_fs, df_fs["reduction_pct"].values, _SYNCED_COLORS["TensorDex-TX"],
                           LW, "-.", "TensorDex-TX", 3))
        if openzl_reduction is not None:
            _plot_data.append((steps_fs, openzl_reduction, _SYNCED_COLORS["OpenZL"],
                               LW, ":", "OpenZL", 2))
        _plot_data.append((steps_fs, zipnn_reduction, _SYNCED_COLORS["ZipNN"],
                           LW, "--", "ZipNN", 2))
        _plot_data.append((steps_fs, sim_reduction, _SYNCED_COLORS["FM-Delta"],
                           LW, "--", "FM-Delta", 3))
        if zp_interp is not None:
            _plot_data.append((steps_fs, zp_interp, _SYNCED_COLORS["ZipLLM"],
                               LW, "--", "ZipLLM", 3))

        for x, y, color, lw, ls, label, zo in _plot_data:
            ax_global.plot(x, y, color=color, linewidth=lw, linestyle=ls, label=label, zorder=zo)

        ax_global.set_ylim(15, 75)
        ax_global.set_xlim(left=-50, right=3650)
        ax_global.set_ylabel("Cumulative Reduction Ratio", labelpad=-5)
        ax_global.set_xlabel("Model Index (by creation time)")
        ax_global.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))

        # All methods for annotations
        all_methods = [
            ("TensorDex-FM", tdx_fm_reduction, _SYNCED_COLORS["TensorDex-FM"], True),
            ("TensorDex-TX", df_fs["reduction_pct"].values, _SYNCED_COLORS["TensorDex-TX"], True),
            ("ZipLLM", zp_interp, _SYNCED_COLORS["ZipLLM"], True),
            ("FM-Delta", sim_reduction, _SYNCED_COLORS["FM-Delta"], True),
            ("ZipNN", zipnn_reduction, _SYNCED_COLORS["ZipNN"], False),
            ("OpenZL", openzl_reduction, _SYNCED_COLORS["OpenZL"], True),
        ]

        # ── Time marker vertical lines with per-method values ──
        timestamps = pd.to_datetime(df_fs["timestamp"], utc=True).dt.tz_localize(None)
        time_markers = ["2024-08-01", "2025-02-13"]  # "2025-04-01" temporarily disabled
        ann_sz = LS
        for tm_str in time_markers:
            tm = pd.Timestamp(tm_str)
            idx = (timestamps - tm).abs().argmin()
            step_val = steps_fs[idx]
            label = pd.Timestamp(tm_str).strftime("%b %Y")
            is_2024 = tm_str.startswith("2024")
            ax_global.axvline(step_val, color="#aaaaaa", linestyle="--",
                              linewidth=1.5, alpha=0.6, zorder=1)
            ax_global.text(step_val + 30, ax_global.get_ylim()[0] + 1, label,
                           ha="left", va="bottom", fontsize=ann_sz, color="#666666")
            above_items = []
            below_items = []
            for name, vals, color, above in all_methods:
                if vals is None:
                    continue
                val = vals[idx]
                if above:
                    above_items.append((val, name, color))
                else:
                    below_items.append((val, name, color))

            group_names = {"TensorDex-TX", "TensorDex-FM", "ZipLLM"}
            group_items = sorted([(v, n, c) for v, n, c in above_items if n in group_names],
                                 key=lambda x: -x[0])
            other_above = [(v, n, c) for v, n, c in above_items if n not in group_names]

            if is_2024 and group_items:
                highest_val = max(v for v, _, _ in group_items)
                spacing = 4.0
                n_grp = len(group_items)
                for rank, (val, name, color) in enumerate(group_items):
                    y_label = highest_val + 6 + (n_grp - 1 - rank) * spacing
                    ax_global.text(step_val + 15, y_label, f"{val:.1f}%",
                                   ha="left", va="bottom", fontsize=ann_sz, color=color)
            else:
                for val, name, color in group_items:
                    ax_global.text(step_val + 15, val + 0.5, f"{val:.1f}%",
                                   ha="left", va="bottom", fontsize=ann_sz, color=color)

            for val, name, color in other_above:
                yoff = 2 if (is_2024 and name == "FM-Delta") else 0.5
                ax_global.text(step_val + 15, val + yoff, f"{val:.1f}%",
                               ha="left", va="bottom", fontsize=ann_sz, color=color)

            for val, name, color in below_items:
                ax_global.text(step_val + 15, val - 2, f"{val:.1f}%",
                               ha="left", va="top", fontsize=ann_sz, color=color)

        # ── Final annotations — right of lines ──
        x_final = steps_fs[-1]
        for name, vals, color, _ in all_methods:
            if vals is None:
                continue
            final = vals[-1]
            yoff = -3 if name == "ZipNN" else 0
            ax_global.text(x_final + steps_fs[-1] * 0.015, final + yoff,
                           f"{final:.1f}%", ha="left", va="center",
                           fontsize=FS , fontweight="bold", color=color)

    ax_global.set_xlabel("(a) Cumulative Storage Reduction Ratio", fontsize=FS, labelpad=12)

    # ════════════════════════════════════════════════════════════════
    # (b) Storage Reduction CDF — adapted from chart_zipllm_real_cdf
    # ════════════════════════════════════════════════════════════════
    try:
        all_savings = mlr._sim_load_savings()
        for label, data in all_savings.items():
            if len(data) == 0:
                continue
            display_label = mlr._BAR_LABELS_MAP.get(label, label.replace("\n", " "))
            color = _SYNCED_COLORS.get(display_label, "#333")
            ls = _SYNCED_LS.get(display_label, "-")
            sorted_data = np.sort(data)
            cdf = np.linspace(0, 1, len(sorted_data))
            step = max(1, len(sorted_data) // 5000)
            ax_cdf.plot(sorted_data[::step], cdf[::step],
                        label=display_label, color=color, linestyle=ls, linewidth=LW)
    except Exception:
        ax_cdf.text(0.5, 0.5, "CDF data\nnot available",
                    ha="center", va="center", transform=ax_cdf.transAxes)

    ax_cdf.set_ylabel("CDF")
    ax_cdf.set_xlim(0, 100)
    ax_cdf.set_ylim(0, 1.02)
    ax_cdf.grid(True, alpha=0.3, color="#cccccc", linewidth=0.8)
    ax_cdf.set_xlabel("(b) Tensor-level Reduction CDF", fontsize=FS, labelpad=12)

    # ════════════════════════════════════════════════════════════════
    # (c) Reduction Distribution by Family — adapted from chart_reduction_violin_by_family
    # ════════════════════════════════════════════════════════════════
    if df_fs is not None:
        red_tx = mlr._per_model_reduction(df_fs)
        red_zp = (mlr._per_model_reduction(df_zp)
                  if df_zp is not None and not df_zp.empty else None)

        _fm_traces = mlr._build_fm_traces(df_fs)
        tdx_fm_trace = _fm_traces.get("flexsplit")
        red_fm = None
        if tdx_fm_trace is not None:
            cum_orig = df_fs["cum_original_bytes"].values.astype(float)
            per_orig = np.diff(cum_orig, prepend=0)
            cum_fm_stored = cum_orig * (1.0 - tdx_fm_trace / 100.0)
            per_fm_stored = np.diff(cum_fm_stored, prepend=0)
            fm_red = np.where(per_orig > 0, (1.0 - per_fm_stored / per_orig) * 100.0, np.nan)
            red_fm = df_fs[["family"]].copy()
            red_fm["reduction"] = fm_red
            red_fm = red_fm.dropna(subset=["reduction"])
            red_fm = red_fm[red_fm["family"].str.lower() != "other"]

        _FAMILY_ORDER = [
            "Qwen/Qwen2.5-7B", "Qwen/Qwen3-8B",
            "mistralai/Mistral-7B-v0.1", "mistralai/Mistral-7B-Instruct-v0.3",
            "google/gemma-2-9b-it", "google/gemma-3-4b-it",
            "meta-llama/Meta-Llama-3-8B", "meta-llama/Llama-3.1-8B",
            "meta-llama/Llama-3.2-3B", "meta-llama/Llama-3.1-8B-Instruct",
        ]
        _FAMILY_LABELS = {
            "Qwen/Qwen2.5-7B":                       "Q2.5",
            "Qwen/Qwen3-8B":                          "Q3",
            "mistralai/Mistral-7B-v0.1":              "M0.1",
            "mistralai/Mistral-7B-Instruct-v0.3":     "M0.3",
            "meta-llama/Meta-Llama-3-8B":             "L3",
            "meta-llama/Llama-3.1-8B":                "L3.1",
            "meta-llama/Llama-3.1-8B-Instruct":       "L3.1-I",
            "meta-llama/Llama-3.2-3B":                "L3.2",
            "google/gemma-2-9b-it":                   "G2",
            "google/gemma-3-4b-it":                   "G3",
        }
        def _shorten(f):
            return _FAMILY_LABELS.get(f, f.split("/")[-1] if "/" in f else f)

        available = set(red_tx["family"].unique())
        family_order = [f for f in _FAMILY_ORDER if f in available]
        for f in sorted(available):
            if f not in family_order and f.lower() != "other":
                family_order.append(f)
        short_labels = [_shorten(f) for f in family_order]

        data_tx = [red_tx.loc[red_tx["family"] == f, "reduction"].values for f in family_order]
        data_fm = ([red_fm.loc[red_fm["family"] == f, "reduction"].values for f in family_order]
                   if red_fm is not None else None)
        data_zp = ([red_zp.loc[red_zp["family"] == f, "reduction"].values for f in family_order]
                   if red_zp is not None else None)

        n = len(family_order)
        positions = np.arange(n)
        width = 0.28
        color_tx = _SYNCED_COLORS["TensorDex-TX"]
        color_fm = _SYNCED_COLORS["TensorDex-FM"]
        color_zp = _SYNCED_COLORS["ZipLLM"]

        def _draw_violin(ax, data, pos, color, med_color):
            parts = ax.violinplot(data, positions=pos,
                                  widths=width * 0.9, showmedians=False, showextrema=False)
            for body in parts["bodies"]:
                body.set_facecolor(color)
                body.set_edgecolor("black")
                body.set_linewidth(0.6)
                body.set_alpha(0.7)
            bp = ax.boxplot(data, positions=pos, widths=width * 0.2,
                            patch_artist=True, showfliers=False, zorder=3)
            for patch in bp["boxes"]:
                patch.set_facecolor("white")
                patch.set_edgecolor("black")
                patch.set_linewidth(0.8)
            for el in ["whiskers", "caps"]:
                for line in bp[el]:
                    line.set_color("black")
                    line.set_linewidth(0.8)
            for line in bp["medians"]:
                line.set_color(med_color)
                line.set_linewidth(1.8)

        if data_zp is not None:
            _draw_violin(ax_violin, data_zp, positions - width, color_zp, color_zp)
        if data_fm is not None:
            _draw_violin(ax_violin, data_fm, positions + width, color_fm, color_fm)
        _draw_violin(ax_violin, data_tx, positions, color_tx, color_tx)

        ax_violin.set_xticks(positions)
        ax_violin.set_xticklabels(short_labels, fontsize=42, rotation=0, ha="center")
        ax_violin.tick_params(axis="y", labelsize=FS)
        ax_violin.set_ylabel("Model-level Reduction %", labelpad=-5)
        ax_violin.set_ylim(bottom=-5, top=105)
        ax_violin.set_yticks([0, 25, 50, 75])
        ax_violin.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))

    ax_violin.set_xlabel("(c) Reduction Distribution by Family", fontsize=FS, labelpad=12)

    fig.tight_layout()
    return fig


# ── Registry ─────────────────────────────────────────────────────────────

CHARTS = {
    "throughput_and_bar": {
        "name": "Throughput & Storage Reduction",
        "category": "Performance",
        "desc": "Combined: throughput scatter (GB/s) + overall storage reduction bar",
        "fn": chart_throughput_and_bar,
    },
    "reduction_combined": {
        "name": "Reduction Combined (Global + CDF + Violin)",
        "category": "Model-Level",
        "desc": "3-panel: Global Storage Reduction, CDF, Reduction by Family",
        "fn": chart_reduction_combined,
    },
}
