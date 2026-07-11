#!/usr/bin/env python3
"""Populate `ae/cache/` with the auxiliary data the figures need (authoring tool).

Reviewers get this pre-staged tree via `download_cache.py` (Hugging Face
dataset). Authors run this once against the research monorepo to assemble what
gets uploaded. `results.db` and `sample_blobs/` are handled separately (they are
large); this covers the small-to-medium CSV/JSON inputs the chart modules read.
"""
from __future__ import annotations

import os
import shutil
import sqlite3

R = "/mnt/sam_nvme/tingfeng/codespace/TensorDex"
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")


def build_slim_metadata():
    """Slim metadata.db for the reduction charts — just tensor sizes and the
    model→tensor map (drops the ~6 GB of fingerprint BLOBs in the full db)."""
    src = os.path.join(R, "data", "tensordb_s3", "metadata.db")
    dst = os.path.join(CACHE, "data", "tensordb_s3", "metadata.db")
    if not os.path.exists(src):
        print(f"  MISS {src} (slim metadata.db not built)")
        return
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        print(f"  ok   data/tensordb_s3/metadata.db (slim, exists)")
        return
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    conn = sqlite3.connect(dst)
    conn.execute(f"ATTACH '{src}' AS full")
    # Keep shape/dtype (reduction charts + facility bench) and param_name (bench),
    # drop only the ~6 GB of fingerprint BLOBs.
    conn.executescript(
        "CREATE TABLE tensors (id TEXT PRIMARY KEY, shape TEXT, dtype TEXT, size_bytes INTEGER);"
        "INSERT INTO tensors SELECT id, shape, dtype, size_bytes FROM full.tensors;"
        "CREATE TABLE model_mappings (model_name TEXT, param_name TEXT, tensor_id TEXT);"
        "INSERT INTO model_mappings SELECT model_name, param_name, tensor_id FROM full.model_mappings;"
        "CREATE INDEX idx_mm_tid ON model_mappings(tensor_id);")
    conn.commit(); conn.close()
    print(f"  ok   data/tensordb_s3/metadata.db (slim, {os.path.getsize(dst)/1e6:.0f} MB)")

# newest end-to-end FlexSplit run (flexsplit charts auto-discover flexsplit_all_*)
FLEXSPLIT_RUN = "tests/output/flexsplit_all_2026-04-06_06-57-11"

# (src relative to R, dst relative to CACHE)
FILES = [
    # Fig 2 / Fig 4 — model-hub growth & metadata quality
    ("model_hub_crawl/base_monthly_stats.json", "model_hub_crawl/base_monthly_stats.json"),
    ("model_hub_crawl/model_type_monthly_stats.json", "model_hub_crawl/model_type_monthly_stats.json"),
    ("model_hub_crawl/model_snapshot_merged_last_month.csv", "model_hub_crawl/model_snapshot_merged_last_month.csv"),
    # Fig 13 / 15 / 16 — FlexSplit analysis
    (f"{FLEXSPLIT_RUN}/flexsplit_all_results.json", f"{FLEXSPLIT_RUN}/flexsplit_all_results.json"),
    # NOTE: real_compression_all_models.csv is staged ONCE at top-level
    # compression_data/ (below); flexsplit_analysis falls back to it, so the
    # 180 MB copy under the run dir is not duplicated in the cache.
    # Fig 14 — FlexSplit vs ILP/Primal-Dual scalability
    ("tests/output/algo_benchmark/logs/bench_cached_facility_parsed.csv",
     "tests/output/algo_benchmark/logs/bench_cached_facility_parsed.csv"),
    # Fig 11a — cumulative reduction trace
    ("plots/model_level_reduction/trace_global_flexsplit.csv", "model_level_reduction/trace_global_flexsplit.csv"),
    ("plots/model_level_reduction/trace_global_zipllm.csv", "model_level_reduction/trace_global_zipllm.csv"),
    # model-level reduction baselines
    ("tests/output/compare_methods/zipllm_all_models.csv", "tests/output/compare_methods/zipllm_all_models.csv"),
    ("tests/output/openzl_benchmark.csv", "tests/output/openzl_benchmark.csv"),
    (f"{FLEXSPLIT_RUN}/compression_data/real_compression_all_models.csv", "compression_data/real_compression_all_models.csv"),
    ("data/models/hf_created_at_cache.json", "data/models/hf_created_at_cache.json"),
    # Fig 14 facility benchmark — model filter
    ("data/models/models.json", "data/models/models.json"),
]


def main() -> int:
    staged = missing = 0
    total = 0
    for src_rel, dst_rel in FILES:
        src = os.path.join(R, src_rel)
        dst = os.path.join(CACHE, dst_rel)
        if not os.path.exists(src):
            print(f"  MISS {src_rel}")
            missing += 1
            continue
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if not os.path.exists(dst) or os.path.getsize(dst) != os.path.getsize(src):
            shutil.copy2(src, dst)
        sz = os.path.getsize(dst)
        total += sz
        staged += 1
        print(f"  ok   {dst_rel}  ({sz/1e6:.1f} MB)")
    build_slim_metadata()
    print(f"\nstaged {staged} files ({total/1e6:.0f} MB) into {CACHE}"
          + (f"; {missing} missing" if missing else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
