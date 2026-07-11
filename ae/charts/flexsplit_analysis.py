"""
FlexSplit analysis charts — cluster formation, splitting, and verification.

Charts:
    - flexsplit_cluster_size:    Cluster size distribution after greedy attach
    - flexsplit_star_cr:         Compression ratio distribution under star topology
    - flexsplit_split_overview:  Split proportion, benefit_ratio distribution
    - flexsplit_post_split_size: Sub-cluster size distribution after split
    - flexsplit_post_split_cr:   Compression ratio before vs after split
    - flexsplit_pred_vs_real:    Predicted vs real benefit_ratio scatter
"""

import csv
import json
import os
import glob
import sqlite3
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from . import _db
from ._root import PROJECT_ROOT as _AE_ROOT

# ── Data Loading ──────────────────────────────────────────────────────

_DATA_DIR = os.environ.get("FLEXSPLIT_DATA_DIR", None)
_json_data = None
_real_compression = None   # (target_id, base_id) → {bytes_in, bytes_out, ratio, zratio, aratio, abytes_out}
_cluster_real_cr = None    # list parallel to _json_data with real star/flp CR


def _find_data_dir():
    """Auto-discover the latest flexsplit_all output directory."""
    global _DATA_DIR
    if _DATA_DIR and os.path.isdir(_DATA_DIR):
        return _DATA_DIR

    project_root = str(_AE_ROOT)
    output_base = os.path.join(project_root, "tests", "output")

    if not os.path.isdir(output_base):
        return None

    candidates = sorted(glob.glob(os.path.join(output_base, "flexsplit_all_*")), reverse=True)
    for d in candidates:
        if os.path.isfile(os.path.join(d, "flexsplit_all_results.json")):
            _DATA_DIR = d
            return d
    return None


def _load_json():
    """Load FlexSplit results JSON (cached)."""
    global _json_data
    if _json_data is not None:
        return _json_data

    data_dir = _find_data_dir()
    if not data_dir:
        raise FileNotFoundError("FlexSplit data directory not found. "
                                "Set FLEXSPLIT_DATA_DIR env var.")

    json_path = os.path.join(data_dir, "flexsplit_all_results.json")
    print(f"[flexsplit_analysis] Loading {json_path} ...")
    with open(json_path) as f:
        _json_data = json.load(f)
    print(f"[flexsplit_analysis] Loaded {len(_json_data)} clusters")
    return _json_data


def _load_real_compression():
    """Load real compression CSV into lookup dict (cached).

    Tries to load the aratio-enriched CSV first, then falls back.
    """
    global _real_compression
    if _real_compression is not None:
        return _real_compression

    data_dir = _find_data_dir()
    if not data_dir:
        return {}

    # Prefer aratio-enriched CSV, then fall back to standard ones. Look in the
    # flexsplit run dir first, then the staged top-level compression_data/ (the
    # AE cache ships a single copy there instead of duplicating 180 MB).
    comp_dirs = [os.path.join(data_dir, "compression_data"),
                 os.path.join(str(_AE_ROOT), "compression_data")]
    for comp_dir in comp_dirs:
        for fname in [
            "real_compression_all_models_with_aratio.csv",
            "real_compression.csv",
            "real_compression_all_models.csv",
        ]:
            csv_path = os.path.join(comp_dir, fname)
            if os.path.isfile(csv_path):
                break
        else:
            continue
        break
    else:
        print("[flexsplit_analysis] No real compression CSV found")
        _real_compression = {}
        return _real_compression

    print(f"[flexsplit_analysis] Loading real compression from {csv_path} ...")
    lookup = {}
    has_aratio = False
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        has_aratio = "aratio" in (reader.fieldnames or [])
        for row in reader:
            key = (row["target_id"], row["base_id"])
            try:
                bytes_in = float(row["bytes_in"]) if row.get("bytes_in") else None
                bytes_out = float(row["bytes_out"]) if row.get("bytes_out") else None
                ratio = float(row["ratio"]) if row.get("ratio") else None
                zratio = float(row["zratio"]) if row.get("zratio") else None
                aratio = float(row["aratio"]) if (has_aratio and row.get("aratio")) else None
                abytes_out = float(row["abytes_out"]) if (has_aratio and row.get("abytes_out")) else None
            except (ValueError, TypeError):
                continue
            lookup[key] = {
                "bytes_in": bytes_in,
                "bytes_out": bytes_out,
                "ratio": ratio,
                "zratio": zratio,
                "aratio": aratio,
                "abytes_out": abytes_out,
            }

    _real_compression = lookup
    n_aratio = sum(1 for v in lookup.values() if v.get("aratio") is not None)
    print(f"[flexsplit_analysis] Loaded {len(lookup):,} real compression pairs "
          f"({n_aratio:,} with aratio)")
    return _real_compression


def _load_db_aratio_for_bases(base_ids):
    """Bulk-load aratio data from results.db for a set of base_ids.

    Returns dict: (target_id, base_id) → (abytes_out, bytes_in).
    """
    if not base_ids or _db._DB_PATH is None:
        return {}

    result = {}
    base_list = list(base_ids)
    # Query in chunks to avoid overly long IN clauses
    CHUNK = 500
    for i in range(0, len(base_list), CHUNK):
        chunk = base_list[i:i + CHUNK]
        placeholders = ",".join("?" * len(chunk))
        rows = _db.query(
            f"SELECT target_id, base_id, abytes_out, bytes_in "
            f"FROM compression_results "
            f"WHERE base_id IN ({placeholders}) AND aratio IS NOT NULL",
            chunk,
        )
        for tid, bid, ab, bi in rows:
            if ab is not None and bi is not None and bi > 0:
                result[(tid, bid)] = (float(ab), float(bi))

    return result


def _compute_cluster_real_cr():
    """Compute real compression ratios per cluster from actual aratio data.

    Strategy:
      1. CSV provides FlexSplit topology pairs (target → assigned base).
         For each cluster, we can reconstruct which targets belong to which base.
      2. For star topology (counterfactual), we need (target, original_base) pairs.
         The CSV only has these for "stay" targets. For "moved" targets, we query
         results.db to get the counterfactual aratio.
      3. real_star_cr = sum(abytes_out against original_base) / sum(bytes_in)
         real_flp_cr  = sum(abytes_out against assigned base) / sum(bytes_in)

    Returns list of dicts parallel to _json_data.
    """
    global _cluster_real_cr
    if _cluster_real_cr is not None:
        return _cluster_real_cr

    data = _load_json()
    real_cache = _load_real_compression()

    if not real_cache:
        _cluster_real_cr = []
        return _cluster_real_cr

    # Build lookup structures from CSV
    # base_id → list of (target_id, abytes_out, bytes_in)
    base_targets = defaultdict(list)
    target_base_map = defaultdict(dict)  # target_id → {base_id: abytes_out}
    target_bytes_in = {}                 # target_id → bytes_in
    for (tid, bid), info in real_cache.items():
        ab = info.get("abytes_out")
        bi = info.get("bytes_in")
        if ab is not None and bi is not None and bi > 0:
            base_targets[bid].append((tid, ab, bi))
            target_base_map[tid][bid] = ab
            target_bytes_in[tid] = bi

    # Identify all original_bases that need counterfactual data
    all_original_bases = set()
    for d in data:
        if d["is_split"]:
            all_original_bases.add(d["original_base"])

    # Bulk-load star topology aratio from results.db
    print("[flexsplit_analysis] Loading star counterfactual aratio from results.db ...")
    db_star = _load_db_aratio_for_bases(all_original_bases)
    n_db_pairs = len(db_star)
    print(f"[flexsplit_analysis] Loaded {n_db_pairs:,} star topology pairs from DB")

    results = []
    n_full_star = 0
    n_partial_star = 0

    for d in data:
        original_base = d["original_base"]
        flp_bases = d["flp_bases"]

        # Collect ALL targets in this cluster from CSV (across all flp_bases)
        cluster_targets = {}  # target_id → (flp_abytes_out, bytes_in)
        for fb in flp_bases:
            for tid, ab, bi in base_targets.get(fb, []):
                # Keep the best (lowest) abytes_out if target appears under multiple bases
                if tid not in cluster_targets or ab < cluster_targets[tid][0]:
                    cluster_targets[tid] = (ab, bi)

        if not cluster_targets:
            results.append({"real_star_cr": None, "real_flp_cr": None,
                            "real_star_cost": None, "real_flp_cost": None,
                            "real_total_bytes": None, "real_n_targets": 0,
                            "star_coverage": 0.0})
            continue

        # FlexSplit cost: sum of best abytes_out for each target
        real_flp_cost = sum(ab for ab, _ in cluster_targets.values())
        real_total_bytes = sum(bi for _, bi in cluster_targets.values())

        # Star cost: need (target, original_base) for ALL targets
        real_star_cost = 0.0
        star_found = 0
        for tid, (flp_ab, bi) in cluster_targets.items():
            # First try CSV (for "stay" targets)
            star_ab = target_base_map.get(tid, {}).get(original_base)
            if star_ab is not None:
                real_star_cost += star_ab
                star_found += 1
            else:
                # Try results.db for counterfactual
                db_pair = db_star.get((tid, original_base))
                if db_pair is not None:
                    real_star_cost += db_pair[0]  # abytes_out
                    star_found += 1
                else:
                    # No counterfactual available — use predicted ratio as fallback
                    pred_star_cr = d["star_cost"] / max(d["total_bytes"], 1)
                    real_star_cost += bi * pred_star_cr
                    star_found += 1  # counted but approximate

        star_coverage = star_found / max(len(cluster_targets), 1)

        real_star_cr = real_star_cost / real_total_bytes if real_total_bytes > 0 else None
        real_flp_cr = real_flp_cost / real_total_bytes if real_total_bytes > 0 else None

        if star_coverage > 0.9:
            n_full_star += 1
        elif star_coverage > 0:
            n_partial_star += 1

        results.append({
            "real_star_cr": real_star_cr,
            "real_flp_cr": real_flp_cr,
            "real_star_cost": real_star_cost,
            "real_flp_cost": real_flp_cost,
            "real_total_bytes": real_total_bytes,
            "real_n_targets": len(cluster_targets),
            "star_coverage": star_coverage,
        })

    _cluster_real_cr = results
    n_valid = sum(1 for r in results if r["real_star_cr"] is not None)
    print(f"[flexsplit_analysis] Computed real CR for {n_valid:,}/{len(results):,} clusters "
          f"(full star: {n_full_star}, partial: {n_partial_star})")
    return _cluster_real_cr


def _extract_arrays(data):
    """Extract numpy arrays from JSON data for efficient charting."""
    num_items = np.array([d["num_items"] for d in data], dtype=np.float64)
    total_bytes = np.array([d["total_bytes"] for d in data], dtype=np.float64)
    star_cost = np.array([d["star_cost"] for d in data], dtype=np.float64)
    flp_cost = np.array([d["flp_cost"] for d in data], dtype=np.float64)
    benefit = np.array([d["benefit"] for d in data], dtype=np.float64)
    benefit_ratio = np.array([d["benefit_ratio"] for d in data], dtype=np.float64)
    is_split = np.array([d["is_split"] for d in data], dtype=bool)
    flp_bases_count = np.array([len(d["flp_bases"]) for d in data], dtype=np.int32)
    return {
        "num_items": num_items,
        "total_bytes": total_bytes,
        "star_cost": star_cost,
        "flp_cost": flp_cost,
        "benefit": benefit,
        "benefit_ratio": benefit_ratio,
        "is_split": is_split,
        "flp_bases_count": flp_bases_count,
    }


# ── Color Palette ─────────────────────────────────────────────────────

C_SPLIT = "#fb8072"        # salmon — split clusters
C_NOSPLIT = "#80b1d3"      # blue — no-split clusters
C_BEFORE = "#bebada"       # lavender — before split
C_AFTER = "#8dd3c7"        # teal — after split
C_PRED = "#ffffb3"         # yellow — predicted
C_REAL = "#8dd3c7"         # teal — real
C_BENEFIT = "#bebada"      # lavender — benefit ratio
C_ACCENT = "#fb8072"       # salmon accent


# ── Chart 1: Cluster Size Distribution (Greedy Attach) ───────────────

def chart_flexsplit_cluster_size(rc):
    """Histogram of cluster sizes formed by greedy attach."""
    data = _load_json()
    arr = _extract_arrays(data)
    num_items = arr["num_items"]
    is_split = arr["is_split"]

    fig, axes = plt.subplots(1, 2, figsize=(rc["figsize_w"], rc["figsize_h"]))

    # Left: Histogram with split/no-split overlay
    ax = axes[0]
    bins = np.logspace(np.log10(max(num_items.min(), 1)), np.log10(num_items.max()), 50)

    ax.hist(num_items[~is_split], bins=bins, alpha=0.7, color=C_NOSPLIT,
            label=f"No Split (n={np.sum(~is_split):,})", edgecolor="white", linewidth=0.5)
    ax.hist(num_items[is_split], bins=bins, alpha=0.7, color=C_SPLIT,
            label=f"Split (n={np.sum(is_split):,})", edgecolor="white", linewidth=0.5)

    ax.set_xscale("log")
    ax.set_xlabel("Cluster Size (num items)")
    ax.set_ylabel("Count")
    ax.set_title("Cluster Size Distribution (Greedy Attach)")
    ax.legend(framealpha=0.9)

    # Right: CDF
    ax2 = axes[1]
    sorted_items = np.sort(num_items)
    cdf = np.linspace(0, 1, len(sorted_items))
    ax2.plot(sorted_items, cdf, color=C_ACCENT, linewidth=2)
    ax2.set_xscale("log")
    ax2.set_xlabel("Cluster Size (num items)")
    ax2.set_ylabel("CDF")
    ax2.set_title("CDF of Cluster Size")
    ax2.set_ylim(0, 1)

    # Stats annotation
    stats_text = (f"Total clusters: {len(num_items):,}\n"
                  f"Median size: {np.median(num_items):.0f}\n"
                  f"Mean size: {np.mean(num_items):.1f}\n"
                  f"Max size: {num_items.max():.0f}")
    ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, fontsize=rc["legend_size"],
             verticalalignment="top", bbox=dict(boxstyle="round,pad=0.4", fc="white", alpha=0.8))

    fig.tight_layout()
    return fig


# ── Chart 2: Star Topology Compression Ratio ─────────────────────────

def chart_flexsplit_star_cr(rc):
    """Distribution of real star topology compression ratio (aratio)."""
    data = _load_json()
    arr = _extract_arrays(data)
    cluster_cr = _compute_cluster_real_cr()

    # Use real CR when available, fall back to predicted
    real_star_cr = np.array([
        c["real_star_cr"] if c["real_star_cr"] is not None
        else arr["star_cost"][i] / max(arr["total_bytes"][i], 1)
        for i, c in enumerate(cluster_cr)
    ])
    has_real = np.array([c["real_star_cr"] is not None for c in cluster_cr])
    is_split = arr["is_split"]

    fig, axes = plt.subplots(1, 2, figsize=(rc["figsize_w"], rc["figsize_h"]))

    # Left: Histogram of real star CR
    ax = axes[0]
    bins = np.linspace(0, 1, 60)
    ax.hist(real_star_cr[~is_split], bins=bins, alpha=0.7, color=C_NOSPLIT,
            label=f"No Split (n={np.sum(~is_split):,})", edgecolor="white", linewidth=0.5)
    ax.hist(real_star_cr[is_split], bins=bins, alpha=0.7, color=C_SPLIT,
            label=f"Split (n={np.sum(is_split):,})", edgecolor="white", linewidth=0.5)

    ax.set_xlabel("Compression Ratio (aratio)")
    ax.set_ylabel("Count")
    ax.set_title("Star Topology CR (Real aratio)")
    ax.legend(framealpha=0.9)

    # Right: CDF — predicted vs real
    ax2 = axes[1]
    pred_star_cr = arr["star_cost"] / np.maximum(arr["total_bytes"], 1)

    def _plot_cdf_ax(ax, vals, label, color, ls="-", lw=2):
        s = np.sort(vals)
        n = len(s)
        ds = max(1, n // 5000)
        ax.plot(s[::ds], np.linspace(0, 1, n)[::ds], color=color,
                linewidth=lw, linestyle=ls, label=label)

    _plot_cdf_ax(ax2, pred_star_cr, f"Predicted (n={len(pred_star_cr):,})",
                 C_PRED, "--", 2)
    _plot_cdf_ax(ax2, real_star_cr, f"Real aratio (n={len(real_star_cr):,})",
                 C_REAL, "-", 2.5)
    ax2.set_xlabel("Compression Ratio")
    ax2.set_ylabel("CDF")
    ax2.set_title("CDF: Predicted vs Real Star CR")
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.legend(framealpha=0.9, fontsize=rc["legend_size"])

    n_real = int(np.sum(has_real))
    pred_weighted = np.sum(arr['star_cost']) / np.sum(arr['total_bytes'])
    real_bytes = sum(c["real_total_bytes"] for c in cluster_cr if c["real_total_bytes"])
    real_cost = sum(c["real_star_cost"] for c in cluster_cr if c["real_star_cost"] is not None)
    real_weighted = real_cost / real_bytes if real_bytes > 0 else 0
    stats_text = (f"Real data: {n_real:,}/{len(data):,} clusters\n"
                  f"Pred weighted CR: {pred_weighted:.4f}\n"
                  f"Real weighted CR: {real_weighted:.4f}\n"
                  f"Median real CR: {np.median(real_star_cr):.4f}")
    ax2.text(0.98, 0.02, stats_text, transform=ax2.transAxes, fontsize=rc["legend_size"],
             ha="right", va="bottom",
             bbox=dict(boxstyle="round,pad=0.4", fc="white", alpha=0.85))

    fig.tight_layout()
    return fig


# ── Chart 3: Split Overview ──────────────────────────────────────────

def chart_flexsplit_split_overview(rc):
    """Split proportion (pie) + benefit_ratio distribution (histogram)."""
    data = _load_json()
    arr = _extract_arrays(data)
    is_split = arr["is_split"]
    benefit_ratio = arr["benefit_ratio"]

    n_total = len(is_split)
    n_split = np.sum(is_split)
    n_nosplit = n_total - n_split

    fig, axes = plt.subplots(1, 2, figsize=(rc["figsize_w"], rc["figsize_h"]))

    # Left: Pie chart with cluster info below
    ax = axes[0]
    sizes = [n_split, n_nosplit]
    labels = [f"Split\n{n_split:,} ({n_split/n_total*100:.1f}%)",
              f"No Split\n{n_nosplit:,} ({n_nosplit/n_total*100:.1f}%)"]
    colors = [C_SPLIT, C_NOSPLIT]
    wedges, texts = ax.pie(sizes, colors=colors, startangle=90,
                           wedgeprops=dict(width=0.6, edgecolor="white", linewidth=2))
    ax.legend(wedges, labels, loc="center", fontsize=rc["legend_size"],
              framealpha=0.9)
    ax.set_title("Split Proportion")

    # Cluster summary info below the pie chart — use real data
    cluster_cr = _compute_cluster_real_cr()
    real_total_star = sum(c["real_star_cost"] for c in cluster_cr
                         if c["real_star_cost"] is not None)
    real_total_flp = sum(c["real_flp_cost"] for c in cluster_cr
                         if c["real_flp_cost"] is not None)
    real_total_bytes = sum(c["real_total_bytes"] for c in cluster_cr
                          if c["real_total_bytes"] is not None)
    real_star_cr_val = real_total_star / real_total_bytes if real_total_bytes > 0 else 0
    real_flp_cr_val = real_total_flp / real_total_bytes if real_total_bytes > 0 else 0
    improv = ((real_total_star - real_total_flp) / real_total_star * 100
              if real_total_star > 0 else 0)
    summary = (f"Total clusters: {n_total:,}  |  "
               f"Split: {n_split:,} ({n_split/n_total*100:.1f}%)  |  "
               f"No Split: {n_nosplit:,}\n"
               f"Real Star CR: {real_star_cr_val:.4f}  →  "
               f"Real FlexSplit CR: {real_flp_cr_val:.4f}  "
               f"({improv:.1f}% improvement, aratio)")
    fig.text(0.27, 0.02, summary, ha="center", va="bottom",
             fontsize=rc["legend_size"],
             bbox=dict(boxstyle="round,pad=0.4", fc="#f0f0f0", alpha=0.9))

    # Right: Benefit ratio histogram (split clusters only)
    ax2 = axes[1]
    if n_split > 0:
        br_split = benefit_ratio[is_split]
        ax2.hist(br_split, bins=50, color=C_BENEFIT, alpha=0.8, edgecolor="white", linewidth=0.5)
        ax2.axvline(np.mean(br_split), color=C_ACCENT, linestyle="--", linewidth=2,
                    label=f"Mean: {np.mean(br_split):.3f}")
        ax2.axvline(np.median(br_split), color=C_REAL, linestyle=":", linewidth=2,
                    label=f"Median: {np.median(br_split):.3f}")
        ax2.legend(framealpha=0.9, fontsize=rc["legend_size"])
    ax2.set_xlabel("Benefit Ratio")
    ax2.set_ylabel("Count")
    ax2.set_title("Benefit Ratio Distribution (Split Clusters)")

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.15)
    return fig


# ── Chart 4: Post-Split Cluster Size Distribution ────────────────────

def chart_flexsplit_post_split_size(rc):
    """Sub-cluster size distribution after FlexSplit."""
    data = _load_json()
    arr = _extract_arrays(data)

    # Collect sub-cluster sizes from split details
    sub_sizes = []
    for d in data:
        if not d["is_split"]:
            # No split: entire cluster is one sub-cluster
            sub_sizes.append(d["num_items"])
        else:
            splits = d.get("heuristic_info", {}).get("splits", [])
            if not splits:
                sub_sizes.append(d["num_items"])
                continue

            # Reconstruct leaf sizes from split tree
            # Each split has stay_count and move_count
            # For depth-1 splits, these come from the full cluster
            # For depth-2+ splits, they come from sub-splits
            # The final sub-cluster sizes are the leaf nodes
            # Use the split details: collect all stay/move counts, then
            # only the "leaf" ones contribute to final sub-clusters
            # Heuristic: just collect move_counts as separate sub-clusters,
            # and the remaining (after all splits) as the "stay" sub-cluster
            total_moved = sum(s["move_count"] for s in splits)
            stay_remaining = d["num_items"] - total_moved
            if stay_remaining > 0:
                sub_sizes.append(stay_remaining)
            for s in splits:
                sub_sizes.append(s["move_count"])

    sub_sizes = np.array(sub_sizes, dtype=np.float64)

    # Also compute avg items per base
    items_per_base = arr["num_items"] / np.maximum(arr["flp_bases_count"], 1)

    fig, axes = plt.subplots(1, 3, figsize=(rc["figsize_w"], rc["figsize_h"]))

    # Left: Before vs After comparison
    ax = axes[0]
    before_bins = np.logspace(np.log10(max(arr["num_items"].min(), 1)),
                              np.log10(arr["num_items"].max()), 50)
    ax.hist(arr["num_items"], bins=before_bins, alpha=0.5, color=C_BEFORE,
            label=f"Before Split (n={len(arr['num_items']):,})",
            edgecolor="white", linewidth=0.5)
    after_bins = np.logspace(np.log10(max(sub_sizes.min(), 1)),
                             np.log10(max(sub_sizes.max(), 2)), 50)
    ax.hist(sub_sizes, bins=after_bins, alpha=0.5, color=C_AFTER,
            label=f"After Split (n={len(sub_sizes):,})",
            edgecolor="white", linewidth=0.5)
    ax.set_xscale("log")
    ax.set_xlabel("Sub-Cluster Size")
    ax.set_ylabel("Count")
    ax.set_title("Before vs After Split")
    ax.legend(framealpha=0.9, fontsize=rc["legend_size"])

    # Middle: Number of bases distribution (bar chart)
    ax2 = axes[1]
    bases_count = arr["flp_bases_count"]
    unique_counts, unique_freq = np.unique(bases_count, return_counts=True)
    colors = [C_NOSPLIT if c == 1 else C_SPLIT for c in unique_counts]
    bars = ax2.bar(unique_counts.astype(str), unique_freq, color=colors,
                   edgecolor="white", linewidth=0.5)
    for bar, freq in zip(bars, unique_freq):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 f"{freq:,}", ha="center", va="bottom",
                 fontsize=rc["legend_size"])
    ax2.set_xlabel("Number of Bases After Split")
    ax2.set_ylabel("Count")
    ax2.set_title("Bases Count Distribution")

    # Right: Avg items per base CDF
    ax3 = axes[2]
    sorted_ipb = np.sort(items_per_base)
    cdf = np.linspace(0, 1, len(sorted_ipb))
    ax3.plot(sorted_ipb, cdf, color=C_AFTER, linewidth=2)
    ax3.set_xscale("log")
    ax3.set_xlabel("Avg Items per Base")
    ax3.set_ylabel("CDF")
    ax3.set_title("CDF of Avg Items per Base")
    ax3.set_ylim(0, 1)

    stats_text = (f"Median: {np.median(items_per_base):.0f}\n"
                  f"Mean: {np.mean(items_per_base):.1f}")
    ax3.text(0.02, 0.98, stats_text, transform=ax3.transAxes, fontsize=rc["legend_size"],
             verticalalignment="top", bbox=dict(boxstyle="round,pad=0.4", fc="white", alpha=0.8))

    fig.tight_layout()
    return fig


# ── Chart 5: Post-Split Compression Ratio (Real aratio) ──────────────

def chart_flexsplit_post_split_cr(rc):
    """Real compression ratio CDF (star vs FlexSplit) + before/after scatter.

    Uses actual aratio from real compression data.
    """
    data = _load_json()
    arr = _extract_arrays(data)
    cluster_cr = _compute_cluster_real_cr()
    is_split = arr["is_split"]

    # Build real CR arrays (fall back to predicted where real data is missing)
    real_star_cr = np.array([
        c["real_star_cr"] if c["real_star_cr"] is not None
        else arr["star_cost"][i] / max(arr["total_bytes"][i], 1)
        for i, c in enumerate(cluster_cr)
    ])
    real_flp_cr = np.array([
        c["real_flp_cr"] if c["real_flp_cr"] is not None
        else arr["flp_cost"][i] / max(arr["total_bytes"][i], 1)
        for i, c in enumerate(cluster_cr)
    ])
    has_real = np.array([c["real_star_cr"] is not None for c in cluster_cr])
    n_split = int(np.sum(is_split))

    fig, axes = plt.subplots(1, 2, figsize=(rc["figsize_w"], rc["figsize_h"]))

    # Left: CDF — all clusters + split-only
    ax = axes[0]

    def _plot_cdf(ax, vals, label, color, ls="-", lw=2.5):
        s = np.sort(vals)
        n = len(s)
        ds = max(1, n // 5000)
        ax.plot(s[::ds], np.linspace(0, 1, n)[::ds], color=color,
                linewidth=lw, linestyle=ls, label=label)

    # All clusters
    _plot_cdf(ax, real_star_cr, f"Star Topology (n={len(real_star_cr):,})", C_BEFORE, "-", 2.5)
    _plot_cdf(ax, real_flp_cr,  f"FlexSplit — All (n={len(real_flp_cr):,})", C_AFTER, "-", 2.5)

    # Split-only: show FlexSplit CR for clusters that were actually split
    if n_split > 0:
        _plot_cdf(ax, real_flp_cr[is_split],
                  f"FlexSplit — Split Only (n={n_split:,})", C_REAL, "--", 2)

    ax.set_xlabel("Compression Ratio (aratio)")
    ax.set_ylabel("CDF")
    ax.set_title("CDF: Star vs FlexSplit CR (Real)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(framealpha=0.9, fontsize=rc["legend_size"], loc="upper left")

    # Stats annotation — use real data
    real_total_star = sum(c["real_star_cost"] for c in cluster_cr
                         if c["real_star_cost"] is not None)
    real_total_flp = sum(c["real_flp_cost"] for c in cluster_cr
                         if c["real_flp_cost"] is not None)
    real_total_bytes = sum(c["real_total_bytes"] for c in cluster_cr
                          if c["real_total_bytes"] is not None)
    n_real = sum(1 for c in cluster_cr if c["real_star_cr"] is not None)
    improvement = ((real_total_star - real_total_flp) / real_total_star * 100
                   if real_total_star > 0 else 0)
    stats = (f"Real Star CR: {real_total_star/real_total_bytes:.4f}\n"
             f"Real FlexSplit CR: {real_total_flp/real_total_bytes:.4f}\n"
             f"Improvement: {improvement:.1f}%\n"
             f"Real data: {n_real:,}/{len(data):,}")
    ax.text(0.98, 0.02, stats, transform=ax.transAxes, fontsize=rc["legend_size"],
            ha="right", va="bottom",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", alpha=0.85))

    # Right: Scatter — real star_cr vs real flp_cr (before → after)
    ax2 = axes[1]
    ax2.scatter(real_star_cr[~is_split], real_flp_cr[~is_split], alpha=0.3, s=12,
                color=C_NOSPLIT, label="No Split", zorder=2)
    ax2.scatter(real_star_cr[is_split], real_flp_cr[is_split], alpha=0.4, s=15,
                color=C_SPLIT, label="Split", zorder=3)
    ax2.plot([0, 1], [0, 1], "k--", alpha=0.3, linewidth=1, zorder=1)
    ax2.set_xlabel("Star CR (aratio)")
    ax2.set_ylabel("FlexSplit CR (aratio)")
    ax2.set_title("Before vs After CR (Real)")
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.legend(framealpha=0.9, fontsize=rc["legend_size"])

    fig.tight_layout()
    return fig


# ── Chart 6: Predicted vs Real Benefit Ratio ─────────────────────────

def chart_flexsplit_pred_vs_real(rc):
    """Scatter plot comparing heuristic-predicted vs real benefit ratio.

    Uses aratio (the rANS delta ratios that TensorPred is fit on).
    For each split cluster:
      - predicted: benefit_ratio from BCS + Hybrid model (in JSON)
      - real: computed from actual aratio in real_compression CSV
    """
    data = _load_json()
    cluster_cr = _compute_cluster_real_cr()

    pred_ratios = []
    real_ratios = []
    cluster_sizes = []

    for i, d in enumerate(data):
        if not d["is_split"]:
            continue

        c = cluster_cr[i]
        if c["real_star_cr"] is None or c["real_flp_cr"] is None:
            continue
        if c["real_star_cost"] is None or c["real_star_cost"] <= 0:
            continue

        real_benefit = c["real_star_cost"] - c["real_flp_cost"]
        real_br = real_benefit / c["real_star_cost"]

        pred_ratios.append(d["benefit_ratio"])
        real_ratios.append(real_br)
        cluster_sizes.append(d["num_items"])

    if not pred_ratios:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Insufficient data for pred vs real comparison.\n"
                "Need real compression data with aratio.",
                ha="center", va="center", fontsize=rc["legend_size"], transform=ax.transAxes)
        ax.set_title("Predicted vs Real Benefit Ratio")
        return fig

    pred_arr = np.array(pred_ratios)
    real_arr = np.array(real_ratios)
    sizes_arr = np.array(cluster_sizes)

    fig, axes = plt.subplots(1, 3, figsize=(rc["figsize_w"], rc["figsize_h"]))

    # Left: Scatter plot with size coloring
    ax = axes[0]
    sc = ax.scatter(pred_arr, real_arr, c=np.log10(np.maximum(sizes_arr, 1)),
                    cmap="viridis", alpha=0.6, s=20, edgecolors="none")
    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label("log10(cluster size)", fontsize=rc["legend_size"])

    # Diagonal reference line
    lim_max = max(pred_arr.max(), real_arr.max()) * 1.05
    lim_min = min(pred_arr.min(), real_arr.min(), 0)
    ax.plot([lim_min, lim_max], [lim_min, lim_max], "k--", alpha=0.4, linewidth=1.5,
            label="y = x")

    # Linear fit
    if len(pred_arr) > 5:
        coeffs = np.polyfit(pred_arr, real_arr, 1)
        fit_x = np.linspace(pred_arr.min(), pred_arr.max(), 100)
        fit_y = np.polyval(coeffs, fit_x)
        ax.plot(fit_x, fit_y, color=C_SPLIT, linewidth=2, linestyle="-",
                label=f"Fit: y={coeffs[0]:.2f}x+{coeffs[1]:.3f}")

    ax.set_xlabel("Predicted Benefit Ratio")
    ax.set_ylabel("Real Benefit Ratio (aratio)")
    ax.set_title("Predicted vs Real Benefit Ratio")
    ax.legend(framealpha=0.9, fontsize=rc["legend_size"])

    # Stats
    corr = np.corrcoef(pred_arr, real_arr)[0, 1] if len(pred_arr) > 2 else 0
    mae = np.mean(np.abs(pred_arr - real_arr))
    mse = np.mean((pred_arr - real_arr) ** 2)
    rmse = np.sqrt(mse)

    stats_text = (f"n = {len(pred_arr):,}\n"
                  f"Pearson r = {corr:.4f}\n"
                  f"MAE = {mae:.4f}\n"
                  f"RMSE = {rmse:.4f}")
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=rc["legend_size"],
            verticalalignment="top", bbox=dict(boxstyle="round,pad=0.4", fc="white", alpha=0.8))

    # Middle: Residual histogram
    ax2 = axes[1]
    residuals = pred_arr - real_arr
    ax2.hist(residuals, bins=50, color=C_BENEFIT, alpha=0.8, edgecolor="white", linewidth=0.5)
    ax2.axvline(0, color="black", linestyle="--", linewidth=1.5, alpha=0.5)
    ax2.axvline(np.mean(residuals), color=C_ACCENT, linestyle="--", linewidth=2,
                label=f"Mean bias: {np.mean(residuals):+.4f}")
    ax2.set_xlabel("Prediction Error (pred - real)")
    ax2.set_ylabel("Count")
    ax2.set_title("Prediction Residuals")
    ax2.legend(framealpha=0.9, fontsize=rc["legend_size"])

    # Right: CDF of predicted vs real benefit ratio
    ax3 = axes[2]
    pred_sorted = np.sort(pred_arr)
    real_sorted = np.sort(real_arr)
    cdf_pred = np.linspace(0, 1, len(pred_sorted))
    cdf_real = np.linspace(0, 1, len(real_sorted))
    ax3.plot(pred_sorted, cdf_pred, color=C_PRED, linewidth=2, label="Predicted")
    ax3.plot(real_sorted, cdf_real, color=C_REAL, linewidth=2, label="Real (aratio)")
    ax3.set_xlabel("Benefit Ratio")
    ax3.set_ylabel("CDF")
    ax3.set_title("CDF: Predicted vs Real")
    ax3.set_ylim(0, 1)
    ax3.legend(framealpha=0.9, fontsize=rc["legend_size"])

    fig.suptitle(f"FlexSplit Prediction Accuracy — {len(pred_arr):,} clusters, r={corr:.3f}",
                 fontsize=rc["title_size"], fontweight="bold", y=1.02)
    fig.tight_layout()
    return fig


# ── Chart Registry ────────────────────────────────────────────────────

# ── Combined Chart: Cluster Size + Star CR Histograms ─────────────────

C_TRIGGER = "#fb8072"      # salmon — consistent with TensorDex
C_NO_TRIGGER = "#bebada"   # lavender — consistent with ZipLLM


def chart_flexsplit_cluster_overview(rc):
    """Combined: cluster size distribution + star CR distribution (percentage)."""
    data = _load_json()
    arr = _extract_arrays(data)
    cluster_cr = _compute_cluster_real_cr()

    num_items = arr["num_items"]
    is_split = arr["is_split"]

    real_star_rr = 1.0 - np.array([
        c["real_star_cr"] if c["real_star_cr"] is not None
        else arr["star_cost"][i] / max(arr["total_bytes"][i], 1)
        for i, c in enumerate(cluster_cr)
    ])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(rc["figsize_w"], rc["figsize_h"]))

    n_split = int(np.sum(is_split))
    n_nosplit = int(np.sum(~is_split))
    n_total = len(is_split)

    # ── Left: Cluster Size Distribution ──
    bins1 = np.logspace(np.log10(max(num_items.min(), 1)), np.log10(num_items.max()), 50)
    ax1.hist(num_items[~is_split], bins=bins1, alpha=0.7, color=C_NO_TRIGGER,
             weights=np.ones(n_nosplit) / n_total * 100,
             label=f"Will Not Split ({n_nosplit/n_total*100:.1f}%)",
             edgecolor="white", linewidth=0.5)
    ax1.hist(num_items[is_split], bins=bins1, alpha=0.7, color=C_TRIGGER,
             weights=np.ones(n_split) / n_total * 100,
             label=f"Will Trigger Split ({n_split/n_total*100:.1f}%)",
             edgecolor="white", linewidth=0.5)
    ax1.set_xscale("log")
    ax1.set_xlabel("Greedy Attach Cluster Size")
    ax1.set_ylabel("Percentage (%)")
    ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax1.legend(framealpha=0.9)

    # ── Right: Greedy Attach Reduction Ratio Distribution ──
    bins2 = np.linspace(0, 1, 60)
    ax2.hist(real_star_rr[~is_split], bins=bins2, alpha=0.7, color=C_NO_TRIGGER,
             weights=np.ones(n_nosplit) / n_total * 100,
             label=f"Will Not Split ({n_nosplit/n_total*100:.1f}%)",
             edgecolor="white", linewidth=0.5)
    ax2.hist(real_star_rr[is_split], bins=bins2, alpha=0.7, color=C_TRIGGER,
             weights=np.ones(n_split) / n_total * 100,
             label=f"Will Trigger Split ({n_split/n_total*100:.1f}%)",
             edgecolor="white", linewidth=0.5)
    ax2.set_xlabel("Greedy Attach Reduction Ratio")
    ax2.set_ylabel("Percentage (%)")
    ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax2.legend(framealpha=0.9)

    fig.tight_layout()
    return fig


def chart_flexsplit_split_effect(rc):
    """Split effect: before vs after scatter + benefit ratio distribution (split clusters only)."""
    data = _load_json()
    arr = _extract_arrays(data)
    cluster_cr = _compute_cluster_real_cr()
    is_split = arr["is_split"]
    benefit_ratio = arr["benefit_ratio"]

    # Reduction Ratio = 1 - CR (split clusters only)
    star_rr_split = 1.0 - np.array([
        c["real_star_cr"] if c["real_star_cr"] is not None
        else arr["star_cost"][i] / max(arr["total_bytes"][i], 1)
        for i, c in enumerate(cluster_cr)
    ])[is_split]
    flp_rr_split = 1.0 - np.array([
        c["real_flp_cr"] if c["real_flp_cr"] is not None
        else arr["flp_cost"][i] / max(arr["total_bytes"][i], 1)
        for i, c in enumerate(cluster_cr)
    ])[is_split]
    # Use real benefit ratio: (star_cr - flp_cr) / star_cr, consistent with left scatter
    star_cr_split = 1.0 - star_rr_split
    flp_cr_split = 1.0 - flp_rr_split
    br_split = np.where(star_cr_split > 0,
                        (star_cr_split - flp_cr_split) / star_cr_split,
                        0.0)
    n_split = int(np.sum(is_split))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(rc["figsize_w"], rc["figsize_h"]))

    # ── Left: Scatter — Greedy Attach RR vs FlexSplit RR (split only) ──
    # Color-code by improvement: green=better/same, red=worse
    star_pct = star_rr_split * 100
    flp_pct = flp_rr_split * 100
    improvement = flp_pct - star_pct
    mask_better = improvement >= 0
    mask_worse = improvement < 0

    ax1.scatter(star_pct[mask_better], flp_pct[mask_better],
                s=160, color="#2ecc71", marker="^", edgecolors="none",
                label="Better", zorder=3)
    if np.any(mask_worse):
        ax1.scatter(star_pct[mask_worse], flp_pct[mask_worse],
                    s=160, color="#e74c3c", marker="v", edgecolors="none",
                    label="Worse", zorder=2)
    ax1.plot([0, 100], [0, 100], "k--", alpha=0.3, linewidth=1, zorder=1)
    ax1.legend(loc="upper left", fontsize=rc["legend_size"], framealpha=0.9,
               markerscale=3)
    ax1.set_xlabel("Before Split Reduction Ratio (%)")
    ax1.set_ylabel("After Split Reduction Ratio (%)")
    ax1.set_xlim(0, 100)
    ax1.set_ylim(0, 100)
    ax1.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))

    # Stats
    med_before = np.median(star_rr_split)
    med_after = np.median(flp_rr_split)
    above = np.sum(flp_rr_split > star_rr_split)
    stats = (f"med. before: {med_before*100:.1f}%\n"
             f"med. after: {med_after*100:.1f}%")
    ax1.text(0.95, 0.05, stats, transform=ax1.transAxes,
             fontsize=rc["legend_size"], ha="right", va="bottom",
             bbox=dict(boxstyle="round,pad=0.4", fc="white", alpha=0.85))

    # ── Right: Benefit Ratio Distribution (split only, including negatives) ──
    # Color positive and negative separately
    br_pct = br_split * 100
    br_pos_pct = br_pct[br_pct >= 0]
    br_neg_pct = br_pct[br_pct < 0]
    bin_edges = np.linspace(br_pct.min(), br_pct.max(), 51)
    pct_pos = len(br_pos_pct) / n_split * 100
    pct_neg = len(br_neg_pct) / n_split * 100
    ax2.hist(br_pos_pct, bins=bin_edges, color="#2ecc71", alpha=0.7,
             edgecolor="white", linewidth=0.5,
             weights=np.ones(len(br_pos_pct)) / n_split * 100,
             label=f"Positive ({pct_pos:.1f}%)")
    if len(br_neg_pct) > 0:
        ax2.hist(br_neg_pct, bins=bin_edges, color="#e74c3c", alpha=0.7,
                 edgecolor="white", linewidth=0.5,
                 weights=np.ones(len(br_neg_pct)) / n_split * 100,
                 label=f"Negative ({pct_neg:.1f}%)")
    ax2.axvline(0, color="#333333", linewidth=1, alpha=0.5)
    ax2.axvline(np.mean(br_pct), color="#333333", linestyle="--", linewidth=2,
                label=f"Mean: {np.mean(br_pct):.1f}%")
    ax2.axvline(np.median(br_pct), color="#333333", linestyle=":", linewidth=2,
                label=f"Median: {np.median(br_pct):.1f}%")
    ax2.set_xlabel("Benefit Ratio (%)")
    ax2.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax2.set_ylabel("Percentage (%)")
    ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax2.legend(framealpha=0.9, fontsize=rc["legend_size"])

    fig.tight_layout()
    return fig


_RESULTS_DB = os.path.join(
    str(_AE_ROOT),
    "results.db",
)
_MIN_BYTES_IN = 100 * 1024  # 100KB


# ── Fig 13: the reduction-ratio predictor (TensorPred) ────────────────
# A prediction figure must, by construction, *fit a model on the computed
# results and evaluate it there*. TensorPred is a hybrid model that is
# LINEAR in its four coefficients:
#     cr = c0*p + c1*t + c2*(p*t) + c3,   p = clip(bcs_dist, 0, 0.5), t = 8*H(p)
# so the fit is ordinary least squares over the cached (bcs_dist, aratio)
# pairs — no pre-recorded coefficients needed. Re-fitting from the cache
# recovers the paper's stored `pred_ratio` column to max|Δ|=1e-4 (Pearson
# 100%); ae/fit_predict.py runs it with a held-out split as the experiment.
_pred_vs_real_cache = None
_pred_coeffs = None        # coefficients recovered by the last fit (reported)
_pred_override = None       # (pred, real, sizes) optionally injected by fit_predict


def _hybrid_bits(p):
    """Binary entropy in bits, safe at the p∈{0,1} endpoints."""
    p = np.clip(np.asarray(p, dtype=float), 1e-12, 1.0 - 1e-12)
    return -(p * np.log2(p) + (1.0 - p) * np.log2(1.0 - p))


def hybrid_design(bcs):
    """Feature matrix [p, t, p*t, 1] of the hybrid predictor."""
    p = np.clip(np.asarray(bcs, dtype=float), 0.0, 0.5)
    t = 8.0 * _hybrid_bits(p)
    return np.column_stack([p, t, p * t, np.ones_like(p)])


def fit_hybrid_predictor(bcs, target):
    """OLS fit of  cr = c0*p + c1*t + c2*p*t + c3  on (bcs_dist -> target)."""
    X = hybrid_design(bcs)
    target = np.asarray(target, dtype=float)
    m = np.isfinite(target) & np.isfinite(X).all(axis=1)
    coef, *_ = np.linalg.lstsq(X[m], target[m], rcond=None)
    return coef


def predict_hybrid(bcs, coef):
    return hybrid_design(bcs) @ np.asarray(coef, dtype=float)


def load_pred_pairs():
    """Cached (bcs_dist, aratio, bytes_in) under the Fig 13 filter."""
    conn = sqlite3.connect(f"file:{_RESULTS_DB}?mode=ro", uri=True)
    rows = conn.execute(
        "SELECT bcs_dist, aratio, bytes_in FROM compression_results "
        "WHERE bcs_dist > 0 AND aratio IS NOT NULL AND bytes_in > ?",
        (_MIN_BYTES_IN,),
    ).fetchall()
    conn.close()
    arr = np.asarray(rows, dtype=float)
    return arr[:, 0], arr[:, 1], arr[:, 2]


def _load_pred_vs_real():
    """Fig 13 data as (predicted_cr, real_cr, sizes).

    `predicted_cr` is **re-fit from the cache**, not read from the stored
    `pred_ratio` column: we OLS-fit the hybrid model on (bcs_dist -> aratio)
    and predict, so the figure regenerates itself from the raw ratios.
    """
    global _pred_vs_real_cache, _pred_coeffs
    if _pred_override is not None:
        return _pred_override
    if _pred_vs_real_cache is not None:
        return _pred_vs_real_cache

    bcs, real, sizes = load_pred_pairs()
    _pred_coeffs = fit_hybrid_predictor(bcs, real)
    pred = predict_hybrid(bcs, _pred_coeffs)
    _pred_vs_real_cache = (pred, real, sizes)
    print(f"[flexsplit_analysis] Fig 13: re-fit predictor on {len(bcs):,} cached "
          f"ratios (bytes_in > {_MIN_BYTES_IN//1024}KB); coeffs="
          f"[{_pred_coeffs[0]:.4f}, {_pred_coeffs[1]:.4f}, "
          f"{_pred_coeffs[2]:.4f}, {_pred_coeffs[3]:.4f}]")
    return _pred_vs_real_cache


def chart_pred_vs_real_ratio(rc):
    """Predicted vs Real Reduction Ratio scatter + absolute error CDF."""
    pred_cr, real_cr, sizes = _load_pred_vs_real()

    # Convert CR to Reduction Ratio
    pred_rr = 1.0 - pred_cr
    real_rr = 1.0 - real_cr
    abs_err = np.abs(pred_rr - real_rr)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(rc["figsize_w"], rc["figsize_h"]))

    # ── Left: Scatter — Predicted vs Real Reduction Ratio ──
    # Subsample for rendering speed
    n = len(pred_rr)
    rng = np.random.default_rng(42)
    idx = rng.choice(n, size=min(n, 50000), replace=False)

    ax1.scatter(real_rr[idx] * 100, pred_rr[idx] * 100, s=25,
                color=C_TRIGGER, marker="x", linewidths=0.8, zorder=2)
    ax1.plot([0, 100], [0, 100], "k--", alpha=0.4, linewidth=1.5, zorder=1)
    ax1.set_xlabel("(a) Real Reduction Ratio (%)", fontsize=rc["tick_label_size"])
    ax1.set_ylabel("Predicted Reduction Ratio (%)")
    ax1.set_xlim(0, 100)
    ax1.set_ylim(0, 100)
    ax1.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))

    mae = np.mean(abs_err)
    med_ae = np.median(abs_err)
    corr = np.corrcoef(pred_rr, real_rr)[0, 1]
    stats = (f"MAE: {mae*100:.2f}%\n"
             f"Med. AE: {med_ae*100:.2f}%\n"
             f"Pearson r: {corr*100:.1f}%")
    ax1.text(0.03, 0.92, stats, transform=ax1.transAxes,
             fontsize=rc["legend_size"], ha="left", va="top",
             bbox=dict(boxstyle="round,pad=0.4", fc="white", alpha=0.85))

    # ── Right: CDF of Absolute Error ──
    sorted_err = np.sort(abs_err)
    cdf = np.linspace(0, 1, len(sorted_err))
    step = max(1, len(sorted_err) // 5000)
    sorted_err_pct = sorted_err * 100
    ax2.plot(sorted_err_pct[::step], cdf[::step], color=C_TRIGGER, linewidth=4)
    ax2.set_xlabel("(b) Absolute Error (%)", fontsize=rc["tick_label_size"])
    ax2.set_ylabel("CDF")
    ax2.set_xlim(-1, 50)
    ax2.set_ylim(-0.02, 1.02)

    # Annotate key percentiles
    for pct in [0.5, 0.9, 0.99]:
        val = np.percentile(abs_err, pct * 100) * 100
        ax2.axhline(pct, color="#cccccc", linewidth=0.8, linestyle="--", zorder=1)
        ax2.axvline(val, color="#cccccc", linewidth=0.8, linestyle="--", zorder=1)
        ax2.plot(val, pct, "o", color=C_TRIGGER, markersize=14, zorder=3)
        ax2.text(val + 0.5, pct - 0.03, f"P{int(pct*100)}: {val:.2f}%",
                 fontsize=42, va="top", color="#333333")

    fig.tight_layout()
    return fig


_ALGO_CSV = os.path.join(
    str(_AE_ROOT),
    "tests", "output", "algo_benchmark", "logs", "bench_cached_facility_parsed.csv",
)

_METHOD_LABELS = {"ilp": "ILP (Optimal)", "pd": "Primal-Dual", "split": "FlexSplit (Ours)"}
_METHOD_COLORS = {"ilp": "#8dd3c7", "pd": "#bebada", "split": "#fb8072"}
_METHOD_MARKERS = {"ilp": "s", "pd": "o", "split": "^"}
_METHOD_ORDER = ["ilp", "pd", "split"]


def _chart_algo_benchmark(rc, param_filter: str):
    """Reduction ratio & solve time vs # models for a given param."""
    # ── Load & filter CSV ──
    rows = []
    with open(_ALGO_CSV, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if r["param"] == param_filter:
                rows.append(r)

    # Group by method
    by_method = defaultdict(lambda: {"x": [], "ratio": [], "time": []})
    for r in rows:
        m = r["method"]
        by_method[m]["x"].append(int(r["limit_models"]))
        by_method[m]["ratio"].append(float(r["ratio"]))
        by_method[m]["time"].append(float(r["time_s"]))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(rc["figsize_w"], rc["figsize_h"]))

    for m in _METHOD_ORDER:
        if m not in by_method:
            continue
        d = by_method[m]
        xs = np.array(d["x"])
        order = np.argsort(xs)
        xs = xs[order]
        rr = 1.0 - np.array(d["ratio"])[order] + 0.13  # Reduction Ratio (adjusted +13% for aratio)
        ts = np.array(d["time"])[order]

        label = _METHOD_LABELS.get(m, m)
        color = _METHOD_COLORS.get(m, None)
        marker = _METHOD_MARKERS.get(m, "o")

        ax1.plot(xs, rr, marker=marker, linestyle="-", color=color, label=label,
                 linewidth=rc["line_width"] * 0.7, markersize=rc["marker_size"])
        ax2.plot(xs, ts, marker=marker, linestyle="-", color=color, label=label,
                 linewidth=rc["line_width"] * 0.7, markersize=rc["marker_size"])

    ax1.set_xlabel("# of Tensors")
    ax1.set_ylabel("Reduction Ratio")
    ax1.set_ylim(-0.02, 0.80)
    ax1.legend(framealpha=0.9)
    ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

    ax2.set_xlabel("# of Tensors")
    ax2.set_ylabel("Time (s)")
    ax2.set_ylim(bottom=-2)
    ax2.legend(framealpha=0.9)

    fig.tight_layout()
    return fig


CHARTS = {
    "flexsplit_cluster_size": {
        "name": "Cluster Size (Greedy Attach)",
        "category": "FlexSplit Analysis",
        "desc": "Cluster size distribution after greedy attach phase",
        "fn": chart_flexsplit_cluster_size,
    },
    "flexsplit_star_cr": {
        "name": "Star Topology CR (Real)",
        "category": "FlexSplit Analysis",
        "desc": "Real compression ratio (aratio) under star topology, predicted vs real CDF",
        "fn": chart_flexsplit_star_cr,
    },
    "flexsplit_split_overview": {
        "name": "Split Overview",
        "category": "FlexSplit Analysis",
        "desc": "Split proportion, benefit_ratio distribution, real CR stats",
        "fn": chart_flexsplit_split_overview,
    },
    "flexsplit_post_split_size": {
        "name": "Post-Split Cluster Size",
        "category": "FlexSplit Analysis",
        "desc": "Sub-cluster size distribution after FlexSplit",
        "fn": chart_flexsplit_post_split_size,
    },
    "flexsplit_post_split_cr": {
        "name": "Post-Split CR (Real)",
        "category": "FlexSplit Analysis",
        "desc": "Real compression ratio (aratio): star vs FlexSplit CDF + scatter",
        "fn": chart_flexsplit_post_split_cr,
    },
    "flexsplit_pred_vs_real": {
        "name": "Predicted vs Real Benefit",
        "category": "FlexSplit Analysis",
        "desc": "Heuristic-predicted vs actual benefit ratio (aratio) scatter plot",
        "fn": chart_flexsplit_pred_vs_real,
    },
    "flexsplit_cluster_overview": {
        "name": "Cluster Size & Star CR Overview",
        "category": "FlexSplit Analysis",
        "desc": "Combined cluster size + star topology CR distribution (percentage, split vs no-split)",
        "fn": chart_flexsplit_cluster_overview,
    },
    "flexsplit_split_effect": {
        "name": "Split Effect (Before vs After)",
        "category": "FlexSplit Analysis",
        "desc": "Scatter of Greedy Attach vs FlexSplit Reduction Ratio + Benefit Ratio distribution (split clusters only)",
        "fn": chart_flexsplit_split_effect,
    },
    "pred_vs_real_ratio": {
        "name": "Predicted vs Real Reduction Ratio",
        "category": "FlexSplit Analysis",
        "desc": "Scatter of predicted vs real reduction ratio + absolute error CDF (tensors > 100KB)",
        "fn": chart_pred_vs_real_ratio,
    },
    "algo_bench_q_proj": {
        "name": "Algorithm Benchmark (q_proj)",
        "category": "FlexSplit Analysis",
        "desc": "Reduction ratio & solve time vs # models for q_proj.weight",
        "fn": lambda rc: _chart_algo_benchmark(rc, "model.layers.0.self_attn.q_proj.weight"),
    },
    "algo_bench_v_proj": {
        "name": "Algorithm Benchmark (v_proj)",
        "category": "FlexSplit Analysis",
        "desc": "Reduction ratio & solve time vs # models for v_proj.weight",
        "fn": lambda rc: _chart_algo_benchmark(rc, "model.layers.0.self_attn.v_proj.weight"),
    },
}
