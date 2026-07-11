"""
Scatter / 2D histogram charts.

Charts:
    - lowhi_scatter: 2D histogram of low vs high byte-plane entropy
"""

import numpy as np
import matplotlib.pyplot as plt

from ._db import query, query_one
from ._colors import COLORS  # noqa: F401


def chart_lowhi_scatter(rc):
    """2D histogram: low-byte vs high-byte entropy."""
    total = query_one(
        "SELECT COUNT(*) FROM compression_results WHERE target_byte_H IS NOT NULL AND ratio IS NOT NULL"
    )[0]
    step = max(1, total // 100_000)

    rows = query(f"""
        SELECT sub_zz_low_byte_H, sub_zz_high_byte_H
        FROM compression_results
        WHERE target_byte_H IS NOT NULL AND ratio IS NOT NULL AND (rowid % {step}) = 0
    """)
    lo = np.array([r[0] for r in rows if r[0] is not None and r[1] is not None])
    hi = np.array([r[1] for r in rows if r[0] is not None and r[1] is not None])

    fig, ax = plt.subplots()
    h = ax.hist2d(lo, hi, bins=120, cmap="turbo", cmin=1)
    fig.colorbar(h[3], ax=ax, label="Count")
    ax.plot([0, 8], [0, 8], "r--", linewidth=1.5, alpha=0.7, label="x = y")
    ax.set_xlabel("Low Byte Plane Entropy (bits)")
    ax.set_ylabel("High Byte Plane Entropy (bits)")
    ax.set_title("Low-Byte vs. High-Byte Entropy (Sub+Zigzag)")
    ax.legend(framealpha=0.9)
    fig.tight_layout()
    return fig


# ── Chart Registry ────────────────────────────────────────────────────

CHARTS = {
    "lowhi_scatter": {
        "name": "Low vs High Byte Entropy",
        "category": "Scatter / 2D",
        "desc": "2D histogram of low vs high byte-plane entropy",
        "fn": chart_lowhi_scatter,
    },
}
