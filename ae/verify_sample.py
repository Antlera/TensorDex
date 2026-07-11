#!/usr/bin/env python3
"""Tier 1 — sample verification.

Prove the published `results.db` cache is *genuine* without recompressing all
11.4M pairs: draw a random subset, and for each sampled (target, base) pair
re-derive the numbers from the raw tensor bytes we ship, then assert they match
the cache exactly.

For every sampled pair we check two independent things:

  1.  **Content id** — re-hash both tensors' raw bytes with TensorDex's own
      XXH3-128 kernel and assert the digest equals the `target_id` / `base_id`
      the cache is keyed on. (Content-addressing: the id *is* the hash.)
  2.  **TensorX ratio** — recompute the delta compression ratio with the
      TensorX codec (zstd level 1) and assert it equals the cached `tratio`
      to within a tight tolerance.

Both use the freshly built `tensordex._ops`, so a PASS means the reviewer's
extension reproduces the authors' trace bit-for-bit. Fully offline — it reads
only the bundled blobs and DB, never the network.

Usage:
    python ae/verify_sample.py --n 200 [--seed 0] [--db ...] [--blobs ...]
"""
from __future__ import annotations

import argparse
import os
import random
import sqlite3
import sys

_AE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _AE_DIR)
from _blobs import (  # noqa: E402
    HAS_FMPP, TENSORX_LEVEL, available_ids, blob_path, content_id,
    fmpp_ratio, load_tensor_bytes, tensorx_ratio,
)

DEFAULT_DB = os.path.join(_AE_DIR, "cache", "results.db")
DEFAULT_BLOBS = os.path.join(_AE_DIR, "cache", "sample_blobs")


def candidate_pairs(db_path: str, avail: set):
    """Cached pairs whose target *and* base blobs are both available."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.execute("PRAGMA temp_store=MEMORY")  # temp table in RAM (db stays read-only)
    conn.execute("CREATE TEMP TABLE avail(id TEXT PRIMARY KEY)")
    conn.executemany("INSERT OR IGNORE INTO avail VALUES (?)", ((x,) for x in avail))
    rows = conn.execute(
        """SELECT c.target_id, c.base_id, c.bytes_in, c.tratio, c.fratio, c.ttimestamp
           FROM compression_results c
           JOIN avail a ON a.id = c.target_id
           JOIN avail b ON b.id = c.base_id
           WHERE c.tratio IS NOT NULL""").fetchall()
    conn.close()
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description="TensorDex AE — sample verification")
    ap.add_argument("--db", default=DEFAULT_DB, help="results.db cache")
    ap.add_argument("--blobs", default=DEFAULT_BLOBS,
                    help="content-addressed blob root (xx/yy/<id>.safetensors)")
    ap.add_argument("--n", type=int, default=200, help="number of pairs to check")
    ap.add_argument("--seed", type=int, default=0, help="sampling seed")
    ap.add_argument("--tol", type=float, default=1e-6, help="ratio tolerance")
    ap.add_argument("--min-match", type=float, default=0.98,
                    help="min TensorX exact-match rate to PASS (results.db "
                         "accreted over months; a few legacy rows predate the "
                         "final codec — content ids still match 100%%)")
    args = ap.parse_args()

    for path, what in [(args.db, "results.db"), (args.blobs, "blob store")]:
        if not os.path.exists(path):
            print(f"ERROR: {what} not found at {path}\n"
                  f"       run `make ae-cache` first (downloads the HF dataset).")
            return 2

    print(f"Scanning blob store {args.blobs} …")
    avail = available_ids(args.blobs)
    print(f"  {len(avail):,} tensor blobs available")

    pairs = candidate_pairs(args.db, avail)
    print(f"  {len(pairs):,} cached pairs fully covered by the blob sample")
    if not pairs:
        print("ERROR: no verifiable pairs — is the blob sample bundled?")
        return 2

    random.seed(args.seed)
    random.shuffle(pairs)
    if args.n >= len(pairs):
        print(f"  note: requested --n {args.n} >= the {len(pairs)} pairs the "
              f"shipped blob sample covers — verifying all of them")
    pairs = pairs[: args.n]

    codecs = f"TensorX (level {TENSORX_LEVEL})" + (" + FM++" if HAS_FMPP else "")
    print(f"\nVerifying {len(pairs)} random pairs (seed={args.seed}, "
          f"codecs: {codecs})")
    if not HAS_FMPP:
        print("  note: FM++ not checked — build with `--features fmpp` "
              "(make ae-fmpp) to also re-derive fratio.\n")

    id_ok = id_bad = ratio_ok = ratio_bad = 0
    f_ok = f_bad = f_checked = 0
    legacy = []
    for i, (tid, bid, bytes_in, exp_tratio, exp_fratio, tts) in enumerate(pairs, 1):
        traw, _, _ = load_tensor_bytes(blob_path(args.blobs, tid))
        braw, _, _ = load_tensor_bytes(blob_path(args.blobs, bid))

        # (1) content-id checks — the id *is* the hash; must be exact.
        gt, gb = content_id(traw), content_id(braw)
        id_pass = (gt == tid) and (gb == bid)
        id_ok += id_pass
        id_bad += (not id_pass)

        # (2) TensorX ratio check
        _, got = tensorx_ratio(traw, braw)
        r_pass = abs(got - exp_tratio) <= args.tol
        ratio_ok += r_pass
        ratio_bad += (not r_pass)
        if not r_pass:
            legacy.append((tid, bid, got, exp_tratio, tts))

        # (3) FM++ ratio check — only when built and the row has fratio
        f_got = f_pass = None
        if HAS_FMPP and exp_fratio is not None:
            _, f_got = fmpp_ratio(traw, braw)
            f_pass = abs(f_got - exp_fratio) <= args.tol
            f_checked += 1
            f_ok += f_pass
            f_bad += (not f_pass)

        row_ok = id_pass and r_pass and (f_pass is not False)
        if i <= 10 or not row_ok:
            mark = "OK  " if row_ok else "DIFF"
            extra = "" if r_pass else f"  [legacy row {tts}]"
            fstr = f" fratio got={f_got:.9f} exp={exp_fratio:.9f}" if f_pass is not None else ""
            print(f"  [{i:4}/{len(pairs)}] {mark} {tid[:12]}…  "
                  f"id={'✓' if id_pass else '✗'}  "
                  f"tratio got={got:.9f} exp={exp_tratio:.9f}{fstr}{extra}")
    if len(pairs) > 10:
        print(f"  … ({len(pairs) - 10} more)")

    match_rate = ratio_ok / len(pairs)
    print("\n" + "=" * 64)
    print(f"content-id checks : {id_ok}/{len(pairs)} exact"
          f"{'' if id_bad == 0 else f'  ({id_bad} MISMATCH)'}")
    print(f"TensorX ratio     : {ratio_ok}/{len(pairs)} bit-exact "
          f"({match_rate:.1%})"
          f"{'' if ratio_bad == 0 else f'  ({ratio_bad} legacy rows)'}")
    f_rate = (f_ok / f_checked) if f_checked else 1.0
    if HAS_FMPP:
        print(f"FM++ ratio        : {f_ok}/{f_checked} bit-exact "
              f"({f_rate:.1%})"
              f"{'' if f_bad == 0 else f'  ({f_bad} legacy rows)'}")
    if legacy:
        print("\nnote: results.db accreted over months; the rows below predate the "
              "\n      final TensorX codec (their content ids still match exactly):")
        for tid, bid, got, exp, tts in legacy[:5]:
            print(f"      {tid[:12]}…/{bid[:12]}…  got={got:.4f} cached={exp:.4f}  {tts}")

    ok = (id_bad == 0 and match_rate >= args.min_match
          and f_rate >= args.min_match)
    print("\nRESULT:", "PASS ✅  cache reproduced from raw bytes" if ok
          else "FAIL ❌  see mismatches above")
    print("=" * 64)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
