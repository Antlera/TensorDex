#!/usr/bin/env python3
"""Fig 14 — FlexSplit vs ILP vs Primal-Dual scalability (real solver runs).

Runs the facility-location benchmark (`bench_facility.py`) across a sweep of
model counts for two representative tensors (q_proj, v_proj), collects each
solver's reduction ratio and wall-clock time, and writes the long-format CSV the
`algo_bench_{q,v}_proj` charts read. The solvers run for real against the cached
pairwise ratios in `results.db`:

  - **ILP (Gurobi)** — optimal, but super-linear time (needs gurobipy + a license;
    skipped automatically if absent — the ILP curve just won't appear).
  - **Primal-Dual** — classical 3-approximation.
  - **FlexSplit (split)** — this paper's heuristic; near-optimal at ~constant time.

    python ae/bench/run_scaling.py [--sizes 100,200,...,800] [--out <csv>]
"""
from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
_AE = os.path.dirname(_HERE)
_CACHE = os.path.join(_AE, "cache")

PARAMS = {
    "model.layers.0.self_attn.q_proj.weight": "q_proj",
    "model.layers.0.self_attn.v_proj.weight": "v_proj",
}
LONG_COLS = ["log_file", "limit_models", "param", "shape", "tensors",
             "cached_pairs", "override_cov", "method", "ratio", "bases", "time_s"]
# wide method prefix -> chart method name
METHODS = [("split", "split"), ("pd", "pd"), ("ilp", "ilp")]


def run_one(size, param, args):
    """Run the bench for one (size, param); return the wide result row or None."""
    with tempfile.NamedTemporaryFile("r", suffix=".csv", delete=False) as tf:
        wide = tf.name
    cmd = [sys.executable, os.path.join(_HERE, "bench_facility.py"),
           "--results-db", args.results_db, "--metadata-db", args.metadata_db,
           "--models-json", args.models_json, "--family", args.family,
           "--limit-models", str(size), "--target-param", param,
           "--methods", args.methods, "--min-items", "2", "--output-csv", wide]
    subprocess.run(cmd, check=False,
                   stdout=(None if args.verbose else subprocess.DEVNULL),
                   stderr=subprocess.STDOUT if args.verbose else subprocess.DEVNULL)
    row = None
    if os.path.exists(wide):
        with open(wide, newline="") as f:
            rows = list(csv.DictReader(f))
        row = rows[0] if rows else None
        os.unlink(wide)
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description="Fig 14 solver-scalability sweep")
    ap.add_argument("--sizes", default="100,200,300,400,500,600,700,800")
    ap.add_argument("--results-db", default=os.path.join(_CACHE, "results.db"))
    ap.add_argument("--metadata-db", default=os.path.join(_CACHE, "data/tensordb_s3/metadata.db"))
    ap.add_argument("--models-json", default=os.path.join(_CACHE, "data/models/models.json"))
    ap.add_argument("--family", default="Qwen/Qwen2.5-7B")
    ap.add_argument("--methods", default="split,primal_dual,ilp")
    ap.add_argument("--out", default=os.path.join(
        _CACHE, "tests/output/algo_benchmark/logs/bench_cached_facility_parsed.csv"))
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    sizes = [int(s) for s in args.sizes.split(",") if s.strip()]
    long_rows, n_ilp = [], 0
    print(f"Sweeping {len(sizes)} sizes × {len(PARAMS)} params "
          f"(methods: {args.methods}) — real solver runs\n")
    for param, short in PARAMS.items():
        for size in sizes:
            row = run_one(size, param, args)
            if not row:
                print(f"  {short} limit={size}: no result (param absent?)")
                continue
            line = f"  {short} limit={size:>4} tensors={row['n']}"
            for wide_m, chart_m in METHODS:
                ratio = row.get(f"{wide_m}_ratio")
                if ratio in (None, "", "None"):
                    continue  # method skipped (e.g. ILP without Gurobi)
                if wide_m == "ilp":
                    n_ilp += 1
                long_rows.append({
                    "log_file": "ae", "limit_models": size, "param": param,
                    "shape": "", "tensors": row["n"], "cached_pairs": row["cached_pairs"],
                    "override_cov": row.get("override_cov", ""), "method": chart_m,
                    "ratio": ratio, "bases": row.get(f"{wide_m}_bases", ""),
                    "time_s": row.get(f"{wide_m}_time", ""),
                })
                line += f"  {chart_m}={float(ratio):.3f}/{float(row.get(f'{wide_m}_time') or 0):.2f}s"
            print(line)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LONG_COLS)
        w.writeheader()
        w.writerows(long_rows)
    print(f"\nWrote {len(long_rows)} rows -> {args.out}")
    if n_ilp == 0:
        print("NOTE: no ILP rows — gurobipy/license absent, so Fig 14's ILP curve "
              "is omitted. Install gurobipy (free academic license) to include it.")
    print("Render Fig 14:  python ae/render.py --only algo_bench_q_proj,algo_bench_v_proj")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
