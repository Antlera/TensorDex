#!/usr/bin/env python3
"""Build the sample-blob bundle shipped in the HF dataset for Tier 1.

Selects cached (target, base) pairs — restricted to ones whose raw tensor
blobs are available locally — under a total size budget, and copies the blobs
into a content-addressed tree `<out>/<xx>/<yy>/<id>.safetensors`. `verify_sample.py`
then re-derives ids + TensorX ratios from exactly these blobs and checks them
against `results.db`.

This is an *authoring* tool (run by the authors, needs the full local blob
store); reviewers just download the resulting bundle.

Usage:
    python ae/build_sample_bundle.py --budget-gb 4 --seed 0 \
        --blobs /path/to/full/blob/store --out ae/cache/sample_blobs
"""
from __future__ import annotations

import argparse
import os
import random
import shutil
import sqlite3

R = "/mnt/sam_nvme/tingfeng/codespace/TensorDex"
DEFAULT_DB = os.path.join(R, "results.db")
DEFAULT_META = os.path.join(R, "data", "tensordb_s3", "metadata.db")
DEFAULT_BLOBS = "/mnt/sam_nvme/tingfeng/data/tensordb/blobs"


def blob_rel(tid: str) -> str:
    return os.path.join(tid[:2], tid[2:4], f"{tid}.safetensors")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=DEFAULT_DB)
    ap.add_argument("--meta", default=DEFAULT_META, help="metadata.db w/ tensors sizes")
    ap.add_argument("--blobs", default=DEFAULT_BLOBS, help="full local blob store")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "cache", "sample_blobs"))
    ap.add_argument("--budget-gb", type=float, default=4.0)
    ap.add_argument("--max-pairs", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    budget = int(args.budget_gb * (1 << 30))

    print("Querying locally-available cached pairs with sizes …")
    conn = sqlite3.connect(args.db)
    conn.execute("PRAGMA query_only=ON")
    conn.execute(f"ATTACH '{args.meta}' AS m")
    rows = conn.execute(
        """SELECT c.target_id, c.base_id, c.tratio, tt.size_bytes, bb.size_bytes
           FROM compression_results c
           JOIN m.tensors tt ON tt.id = c.target_id AND tt.storage_state='local'
           JOIN m.tensors bb ON bb.id = c.base_id  AND bb.storage_state='local'
           WHERE c.tratio IS NOT NULL""").fetchall()
    conn.close()
    print(f"  {len(rows):,} candidate pairs")

    random.seed(args.seed)
    random.shuffle(rows)

    # Greedy under budget: accumulate pairs, tracking unique blob bytes.
    picked = []                      # (target_id, base_id)
    blob_bytes = {}                  # id -> size
    total = 0
    for tid, bid, tratio, tsz, bsz in rows:
        add = 0
        if tid not in blob_bytes:
            add += tsz
        if bid not in blob_bytes:
            add += bsz
        if total + add > budget:
            continue
        picked.append((tid, bid))
        blob_bytes[tid] = tsz
        blob_bytes[bid] = bsz
        total += add
        if len(picked) >= args.max_pairs:
            break

    print(f"  selected {len(picked)} pairs, {len(blob_bytes)} unique blobs, "
          f"{total / (1<<30):.2f} GB")

    os.makedirs(args.out, exist_ok=True)
    copied = 0
    for tid in blob_bytes:
        rel = blob_rel(tid)
        src = os.path.join(args.blobs, rel)
        dst = os.path.join(args.out, rel)
        if os.path.exists(dst):
            copied += 1
            continue
        if not os.path.exists(src):
            print(f"  WARN missing blob {src}")
            continue
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
        copied += 1
    print(f"  copied {copied} blobs into {args.out}")

    # Manifest (informational; verify_sample re-derives everything itself).
    man = os.path.join(os.path.dirname(args.out), "sample_pairs.tsv")
    with open(man, "w") as f:
        f.write("target_id\tbase_id\n")
        for tid, bid in picked:
            f.write(f"{tid}\t{bid}\n")
    print(f"  wrote manifest {man}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
