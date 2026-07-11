"""
Model Hub Metadata Quality — shows that most models lack metadata,
and low-download models are far more likely to be missing it.

Metadata = has BOTH model card AND base_model field.

Charts:
    - modelhub_metadata: Bar chart — metadata ratio by download bucket
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ._root import PROJECT_ROOT as _PROJECT_ROOT

_CSV = str(_PROJECT_ROOT / "model_hub_crawl" / "model_snapshot_merged_last_month.csv")


def _load():
    df = pd.read_csv(_CSV)
    df = df.dropna(subset=["downloads"])

    has_card = df["has_model_card"] == "yes"
    has_base = df["has_base_model"] == "yes"
    df["has_meta"] = has_card & has_base

    # Rank-based quintiles: sort by downloads descending, bucket by index position
    df = df.sort_values("downloads", ascending=False).reset_index(drop=True)
    n = len(df)
    q = 5
    labels = ["Top 20%", "20–40%", "40–60%", "60–80%", "Bottom 20%"]

    records = []
    for i in range(q):
        start = i * (n // q)
        end = (i + 1) * (n // q) if i < q - 1 else n
        sub = df.iloc[start:end]
        meta_pct = sub["has_meta"].mean() * 100
        records.append({
            "bucket": labels[i],
            "meta_pct": meta_pct,
            "no_meta_pct": 100 - meta_pct,
            "count": len(sub),
            "dl_min": int(sub["downloads"].min()),
            "dl_max": int(sub["downloads"].max()),
        })

    # Overall
    meta_pct_all = df["has_meta"].mean() * 100
    records.append({
        "bucket": "Overall",
        "meta_pct": meta_pct_all,
        "no_meta_pct": 100 - meta_pct_all,
        "count": len(df),
        "dl_min": int(df["downloads"].min()),
        "dl_max": int(df["downloads"].max()),
    })

    return pd.DataFrame(records)


def chart_modelhub_metadata(rc):
    stats = _load()
    n = len(stats)
    x = np.arange(n)

    fig, ax = plt.subplots(figsize=(rc["figsize_w"], rc["figsize_h"]))

    fs = rc.get("label_size", 42)
    ann_fs = rc.get("annotation_size", 20)

    c_meta = "#5680eb"
    c_no = "#d76b5a"

    meta_vals = stats["meta_pct"].values
    no_vals = stats["no_meta_pct"].values

    ax.bar(x, no_vals, color=c_no, width=0.65,
           edgecolor="white", linewidth=0.5, label="No metadata")
    ax.bar(x, meta_vals, bottom=no_vals, color=c_meta, width=0.65,
           edgecolor="white", linewidth=0.5, label="Has metadata")

    for j in range(n):
        nm = no_vals[j]
        ax.text(x[j], nm / 2, f"{nm:.1f}%",
                ha="center", va="center", fontsize=fs,
                fontweight="bold", color="white")

    # Separator before Overall
    ax.axvline(x[-1] - 0.5, color="#888888", linestyle="--", linewidth=2, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(stats["bucket"].values, fontsize=rc["tick_label_size"])
    ax.set_ylabel("Percentage of models (%)", fontsize=fs)
    ax.set_xlabel("Download rank (quintile)", fontsize=fs)
    ax.set_ylim(0, 105)
    ax.tick_params(axis="y", labelsize=rc["tick_label_size"])
    ax.legend(fontsize=rc["legend_size"], loc="upper left",
              handlelength=1.0, handletextpad=0.4)

    fig.tight_layout()
    return fig


CHARTS = {
    "modelhub_metadata": {
        "name": "Model Hub Metadata Quality",
        "category": "Model Hub",
        "desc": "Metadata completeness (card + base_model) by download percentile",
        "fn": chart_modelhub_metadata,
    },
}
