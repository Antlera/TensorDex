"""
CDF analysis charts — cumulative distribution function visualizations.

Charts:
    - entropy_cdf:        CDF of byte entropy at each pipeline stage
    - byteplane_cdf:      CDF of low vs high byte-plane entropy
    - theo_vs_actual_cdf: CDF of theoretical vs actual compression ratio
    - ratio_cdf:          CDF of ratio vs zratio
"""

import numpy as np
import matplotlib.pyplot as plt

from ._db import query, query_one
from ._colors import COLORS


def chart_entropy_cdf(rc):
    """CDF of byte entropy at each pipeline stage."""
    total = query_one(
        "SELECT COUNT(*) FROM compression_results WHERE target_byte_H IS NOT NULL AND ratio IS NOT NULL"
    )[0]
    step = max(1, total // 500_000)

    fig, ax = plt.subplots()

    for col, name, color, ls in [
        ("target_byte_H", "Original Target", COLORS["target"], "-"),
        ("xor_byte_H",    "XOR Delta",       COLORS["xor"],    "-"),
        ("sub_byte_H",    "Sub Delta",        COLORS["sub"],    "-"),
        ("sub_zz_byte_H", "Sub + Zigzag",     COLORS["sub_zz"], "-"),
    ]:
        rows = query(f"""
            SELECT {col} FROM compression_results
            WHERE target_byte_H IS NOT NULL AND ratio IS NOT NULL AND (rowid % {step}) = 0
            ORDER BY {col}
        """)
        vals = np.array([r[0] for r in rows if r[0] is not None])
        n = len(vals)
        ds = max(1, n // 3000)
        ax.plot(vals[::ds], np.linspace(0, 1, n)[::ds], label=name, color=color, linestyle=ls)

    ax.set_xlabel("Byte-level Shannon Entropy (bits)")
    ax.set_ylabel("CDF")
    ax.set_title(f"CDF of Byte Entropy — Pipeline Ablation (n={total:,})")
    ax.legend(loc="upper left", framealpha=0.9)
    ax.set_xlim(left=0)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    return fig


def chart_byteplane_cdf(rc):
    """CDF of low-byte vs high-byte entropy after Sub+Zigzag."""
    total = query_one(
        "SELECT COUNT(*) FROM compression_results WHERE target_byte_H IS NOT NULL AND ratio IS NOT NULL"
    )[0]
    step = max(1, total // 500_000)

    fig, ax = plt.subplots()

    for col, name, color in [
        ("sub_zz_byte_H",      "Sub+ZZ Combined", COLORS["sub_zz"]),
        ("sub_zz_low_byte_H",  "Low Byte Plane",  COLORS["lo"]),
        ("sub_zz_high_byte_H", "High Byte Plane",  COLORS["hi"]),
    ]:
        rows = query(f"""
            SELECT {col} FROM compression_results
            WHERE target_byte_H IS NOT NULL AND ratio IS NOT NULL AND (rowid % {step}) = 0
            ORDER BY {col}
        """)
        vals = np.array([r[0] for r in rows if r[0] is not None])
        n = len(vals)
        ds = max(1, n // 3000)
        ax.plot(vals[::ds], np.linspace(0, 1, n)[::ds], label=name, color=color)

    ax.set_xlabel("Byte-level Shannon Entropy (bits)")
    ax.set_ylabel("CDF")
    ax.set_title("CDF of Byte-Plane Entropy — Zeltax Byte Splitting")
    ax.legend(loc="center right", framealpha=0.9)
    ax.set_xlim(left=0)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    return fig


def chart_theo_vs_actual_cdf(rc):
    """CDF: theoretical compression ratio vs actual ratio/zratio."""
    total = query_one(
        "SELECT COUNT(*) FROM compression_results WHERE target_byte_H IS NOT NULL AND ratio IS NOT NULL"
    )[0]
    step = max(1, total // 300_000)

    rows = query(f"""
        SELECT xor_byte_H, sub_byte_H, sub_zz_byte_H,
               sub_zz_low_byte_H, sub_zz_high_byte_H, ratio, zratio
        FROM compression_results
        WHERE target_byte_H IS NOT NULL AND ratio IS NOT NULL AND (rowid % {step}) = 0
    """)

    xor_H  = np.array([r[0] or 0 for r in rows])
    sub_H  = np.array([r[1] or 0 for r in rows])
    zz_H   = np.array([r[2] or 0 for r in rows])
    lo_H   = np.array([r[3] or 0 for r in rows])
    hi_H   = np.array([r[4] or 0 for r in rows])
    ratio  = np.array([r[5] if r[5] is not None else np.nan for r in rows])
    zratio = np.array([r[6] if r[6] is not None else np.nan for r in rows])

    fig, ax = plt.subplots()

    def _plot_cdf(arr, name, color, ls="-"):
        valid = arr[~np.isnan(arr)]
        s = np.sort(valid)
        n = len(s)
        ds = max(1, n // 3000)
        ax.plot(s[::ds], np.linspace(0, 1, n)[::ds], label=name, color=color, linestyle=ls)

    _plot_cdf(xor_H / 8.0,          "Theoretical: XOR (H/8)",            COLORS["xor"],    ":")
    _plot_cdf(sub_H / 8.0,          "Theoretical: Sub (H/8)",            COLORS["sub"],    ":")
    _plot_cdf(zz_H / 8.0,           "Theoretical: Sub+ZZ (H/8)",         COLORS["sub_zz"], ":")
    _plot_cdf((lo_H + hi_H) / 16.0, "Theoretical: byte-split (lo+hi)/16",COLORS["theo_combined"], "-")
    _plot_cdf(ratio,                 "Actual ratio",                       COLORS["ratio"],  "-")

    zr_valid = zratio[~np.isnan(zratio)]
    if len(zr_valid) > 100:
        _plot_cdf(zratio, f"Actual zratio (n={len(zr_valid):,})", COLORS["zratio"], "-")

    ax.set_xlabel("Compression Ratio")
    ax.set_ylabel("CDF")
    ax.set_title("CDF: Theoretical Ratio vs. Actual Ratio (Ablation)")
    ax.legend(loc="center right", fontsize=rc["legend_size"], framealpha=0.9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    return fig


def chart_ratio_cdf(rc):
    """CDF of compression ratio — ratio vs zratio."""
    total = query_one(
        "SELECT COUNT(*) FROM compression_results WHERE target_byte_H IS NOT NULL AND ratio IS NOT NULL"
    )[0]
    step = max(1, total // 500_000)

    fig, ax = plt.subplots()

    # ratio
    rows = query(f"""
        SELECT ratio FROM compression_results
        WHERE target_byte_H IS NOT NULL AND ratio IS NOT NULL AND (rowid % {step}) = 0
        ORDER BY ratio
    """)
    vals = np.array([r[0] for r in rows if r[0] is not None])
    n = len(vals)
    ds = max(1, n // 3000)
    ax.plot(vals[::ds], np.linspace(0, 1, n)[::ds],
            label=f"ratio (Sub+ZZ, n={n:,})", color=COLORS["ratio"])

    # zratio
    zrows = query("""
        SELECT zratio FROM compression_results
        WHERE zratio IS NOT NULL AND target_byte_H IS NOT NULL ORDER BY zratio
    """)
    zvals = np.array([r[0] for r in zrows])
    nz = len(zvals)
    if nz > 0:
        dsz = max(1, nz // 3000)
        ax.plot(zvals[::dsz], np.linspace(0, 1, nz)[::dsz],
                label=f"zratio (Zeltax+Zstd, n={nz:,})", color=COLORS["zratio"])

    ax.set_xlabel("Compression Ratio (bytes_out / bytes_in)")
    ax.set_ylabel("CDF")
    ax.set_title("CDF of Compression Ratio — ratio vs. zratio")
    ax.legend(framealpha=0.9)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    return fig


# ── Chart Registry ────────────────────────────────────────────────────

CHARTS = {
    "entropy_cdf": {
        "name": "Entropy CDF (Pipeline Stages)",
        "category": "CDF Analysis",
        "desc": "CDF of byte entropy at each pipeline stage",
        "fn": chart_entropy_cdf,
    },
    "byteplane_cdf": {
        "name": "Byte-Plane Entropy CDF",
        "category": "CDF Analysis",
        "desc": "CDF of low vs high byte-plane entropy after Sub+Zigzag",
        "fn": chart_byteplane_cdf,
    },
    "theo_vs_actual_cdf": {
        "name": "Theoretical vs Actual Ratio CDF",
        "category": "CDF Analysis",
        "desc": "Theoretical compression ratio (from entropy) vs actual",
        "fn": chart_theo_vs_actual_cdf,
    },
    "ratio_cdf": {
        "name": "Compression Ratio CDF",
        "category": "CDF Analysis",
        "desc": "CDF of raw ratio vs zratio (Zeltax+Zstd)",
        "fn": chart_ratio_cdf,
    },
}
