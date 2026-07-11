"""
Entropy analysis charts — layer-type breakdowns and pipeline ablation.

Charts:
    - layer_entropy:  Avg byte entropy by layer type & pipeline stage
    - layer_ratio:    Avg compression ratio by layer type — theoretical vs actual
    - pipeline_bar:   Avg byte entropy & theoretical ratio at each pipeline stage
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from ._db import query, query_one, LAYER_TYPE_CASE
from ._colors import COLORS


def chart_layer_entropy(rc):
    """Avg byte entropy by layer type & pipeline stage (grouped bar)."""
    rows = query(f"""
        SELECT {LAYER_TYPE_CASE} as layer_type,
            COUNT(*), AVG(target_byte_H), AVG(xor_byte_H),
            AVG(sub_byte_H), AVG(sub_zz_byte_H),
            AVG(sub_zz_low_byte_H), AVG(sub_zz_high_byte_H)
        FROM compression_results
        WHERE target_byte_H IS NOT NULL AND ratio IS NOT NULL
        GROUP BY layer_type ORDER BY COUNT(*) DESC
    """)

    # 将 other 组移到最后，其余按数量降序排列
    rows = sorted(rows, key=lambda r: (str(r[0]).lower() == 'other', str(r[0]).lower() == 'norm', -r[1]))

    types    = [r[0] for r in rows]
    counts   = [r[1] for r in rows]
    target_H = [r[2] for r in rows]
    xor_H    = [r[3] for r in rows]
    sub_H    = [r[4] for r in rows]
    sub_zz_H = [r[5] for r in rows]
    lo_hi_avg = [(r[6] + r[7]) / 2.0 for r in rows]  # avg of low & high byte entropy

    x_labels = [f"{t}" for t in types]
    x = np.arange(len(types))
    n_groups = 5
    width = 0.8 / n_groups

    fig, ax = plt.subplots()

    series = [
        (target_H,  "Original",          COLORS["sub_zz"]),
        (xor_H,     "XOR Δ",             COLORS["xor"]),
        (sub_H,     "Sub Δ",             COLORS["sub"]),
        (sub_zz_H,  "Sub Δ + ZZ",            COLORS["target"]),
        (lo_hi_avg, "Sub Δ + ZZ + Split",  COLORS["hi"]),
    ]
    for i, (vals, name, color) in enumerate(series):
        offset = (i - n_groups / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=name, color=color,
                      edgecolor="white", linewidth=0.3)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                    f"{v:.2f}", ha="center", va="bottom", rotation=90,
                    fontsize=rc.get("annotation_size", 20))

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=25, ha='center')
    ax.set_ylabel("Byte Entropy (bits)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 0.98), ncol=5, frameon=False)
    ax.set_ylim(bottom=0, top=7.5)
    fig.tight_layout()
    return fig


def chart_layer_ratio(rc):
    """Avg compression ratio by layer type — theoretical vs actual.

    Uses bytes_in–weighted averages. For tensors < 10 KB or layer types
    norm/other, the delta column falls back to zratio instead of aratio.
    """
    _USE_ZRATIO = {"norm", "other"}

    # Weighted averages: SUM(col * bytes_in) / SUM(bytes_in)
    # For <10KB tensors, use zratio instead of aratio.
    _base_sql = f"""
        SELECT {LAYER_TYPE_CASE} as layer_type,
            COUNT(*),
            SUM(sub_zz_low_byte_H * bytes_in) * 1.0 / SUM(bytes_in),
            SUM(sub_zz_high_byte_H * bytes_in) * 1.0 / SUM(bytes_in),
            SUM(ratio * bytes_in) * 1.0 / SUM(bytes_in),
            SUM(COALESCE(
                CASE WHEN bytes_in < 10240 THEN zratio ELSE aratio END,
                aratio, zratio
            ) * bytes_in) * 1.0 / SUM(bytes_in),
            SUM(COALESCE(zratio, aratio) * bytes_in) * 1.0 / SUM(bytes_in),
            SUM(sub_zz_low_H1 * bytes_in) * 1.0 / SUM(bytes_in),
            SUM(sub_zz_high_H1 * bytes_in) * 1.0 / SUM(bytes_in)
        FROM compression_results
        WHERE target_byte_H IS NOT NULL AND ratio IS NOT NULL
            {{extra_filter}}
        GROUP BY layer_type ORDER BY COUNT(*) DESC
    """
    # cols: 0=layer_type, 1=count, 2=lo_H, 3=hi_H, 4=ratio,
    #       5=aratio_eff, 6=zratio_eff, 7=lo_H1, 8=hi_H1

    # Main query: all layer types except norm/other
    rows_main = query(_base_sql.format(extra_filter=""))
    # For norm/other: only use pairs that also have zratio (common with bitx)
    rows_zr = query(_base_sql.format(extra_filter="AND zratio IS NOT NULL"))

    # Build final rows: use zratio-filtered data for norm/other
    row_map_zr = {str(r[0]).lower(): r for r in rows_zr}
    rows = []
    for r in rows_main:
        lt = str(r[0]).lower()
        if lt in _USE_ZRATIO and lt in row_map_zr:
            rows.append(row_map_zr[lt])
        else:
            rows.append(r)

    rows = sorted(rows, key=lambda r: (str(r[0]).lower() == 'other', str(r[0]).lower() == 'norm', -r[1]))

    types  = [r[0] for r in rows]

    def _pick_delta(r):
        lt = str(r[0]).lower()
        if lt in _USE_ZRATIO:
            v = r[6]  # zratio-based weighted avg
        else:
            v = r[5]  # aratio-based weighted avg (with <10KB fallback)
        return v if v is not None else 0

    ratios_bitx   = [1.0 - (r[4] if r[4] else 0) for r in rows]
    ratios_delta  = [1.0 - _pick_delta(r) for r in rows]
    theo_vals     = [1.0 - ((r[7] or r[2]) + (r[8] or r[3])) / 16.0 for r in rows]

    x_labels = [f"{t}" for t in types]
    x = np.arange(len(types))
    n_groups = 3
    width = 0.8 / n_groups

    fig, ax = plt.subplots()
    series = [
        (ratios_bitx,    "BitX Ratio",    COLORS["xor"],     1.0),
        (ratios_delta,   "rANS Delta Ratio", COLORS["zratio"], 1.0),
        (theo_vals,      "Oracle Ratio",  COLORS["sub_zz"],  1.0),
    ]
    for i, (vals, name, color, alpha) in enumerate(series):
        offset = (i - n_groups / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=name, color=color,
                      alpha=alpha, edgecolor="white", linewidth=0.3)
        for bar, v in zip(bars, vals):
            label_text = f"{v:.3f}" if v > 0 else "N/A"
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    label_text, ha="center", va="bottom", rotation=90,
                    fontsize=rc.get("annotation_size", 20) + 4)

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=25, ha='center')
    ax.set_ylabel("Reduction Ratio")
    ax.set_ylim(0, 1.1)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 0.98), ncol=3, frameon=False)
    fig.tight_layout()
    return fig


def chart_pipeline_bar(rc):
    """Avg byte entropy & theoretical ratio at each pipeline stage."""
    agg = query_one("""
        SELECT AVG(target_byte_H), AVG(xor_byte_H), AVG(sub_byte_H), AVG(sub_zz_byte_H),
               AVG(sub_zz_low_byte_H), AVG(sub_zz_high_byte_H),
               AVG(ratio), AVG(zratio), COUNT(*), COUNT(zratio)
        FROM compression_results
        WHERE target_byte_H IS NOT NULL AND ratio IS NOT NULL
    """)

    stages = ["Original", "XOR Δ", "Sub Δ", "Sub Δ + ZZ", "Sub Δ + ZZ\n+ Split"]
    entropy_vals = [agg[0], agg[1], agg[2], agg[3], (agg[4] + agg[5]) / 2]
    theo_ratios = [agg[0] / 8.0, agg[1] / 8.0, agg[2] / 8.0,
                   agg[3] / 8.0, (agg[4] + agg[5]) / 16.0]

    colors = [COLORS["target"], COLORS["xor"], COLORS["sub"], COLORS["sub_zz"],
              COLORS["hi"]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(rc["figsize_w"], rc["figsize_h"] * 0.7))
    x = np.arange(len(stages))

    # Left: entropy
    bars1 = ax1.bar(x, entropy_vals, color=colors, edgecolor="white", linewidth=0.5)
    for bar, v in zip(bars1, entropy_vals):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                 f"{v:.3f}", ha="center", va="bottom", fontsize=rc.get("annotation_size", 20))
    ax1.set_xticks(x)
    ax1.set_xticklabels(stages, fontsize=rc.get("annotation_size", 20))
    ax1.set_ylabel("Byte Entropy (bits)")
    ax1.set_title("Avg Byte Entropy per Stage")
    ax1.set_ylim(0, max(entropy_vals) * 1.2)

    # Right: theoretical ratio
    bars2 = ax2.bar(x, theo_ratios, color=colors, edgecolor="white", linewidth=0.5)
    for bar, v in zip(bars2, theo_ratios):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                 f"{v:.1%}", ha="center", va="bottom", fontsize=rc.get("annotation_size", 20))
    ax2.set_xticks(x)
    ax2.set_xticklabels(stages, fontsize=rc.get("annotation_size", 20))
    ax2.set_ylabel("Ratio (H/8)")
    ax2.set_title("Theoretical Compression Ratio")
    ax2.set_ylim(0, max(theo_ratios) * 1.2)
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter(1.0, decimals=0))

    fig.tight_layout()
    return fig


# ── Chart Registry ────────────────────────────────────────────────────

CHARTS = {
    "layer_entropy": {
        "name": "Entropy by Layer Type",
        "category": "Entropy Analysis",
        "desc": "Avg byte entropy by layer type across pipeline stages",
        "fn": chart_layer_entropy,
    },
    "layer_ratio": {
        "name": "Compression Ratio by Layer Type",
        "category": "Entropy Analysis",
        "desc": "Theoretical vs actual compression ratio by layer type",
        "fn": chart_layer_ratio,
    },
    "pipeline_bar": {
        "name": "Pipeline Ablation Bar",
        "category": "Entropy Analysis",
        "desc": "Avg byte entropy & theoretical ratio at each pipeline stage",
        "fn": chart_pipeline_bar,
    },
}
