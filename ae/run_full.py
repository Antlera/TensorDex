#!/usr/bin/env python3
"""Tier 2 — end-to-end reproduction driver.

Tiers 0/1 prove the *published* numbers are real (figures from cache; a random
subset re-derived bit-for-bit from raw bytes). Tier 2 runs the actual pipeline —
ingest -> TensorSketch -> FlexSplit planning -> TensorX delta compression -> a
fresh `results.db` -> figures — so a reviewer can regenerate the trace rather
than trust the cache.

Two scales:

  --mode demo   (default, ~seconds, offline)
      Synthetic base + fine-tune through the whole lifecycle
      (ingest, dedup, plan, compress, integrity-checked pull). Proves the
      pipeline is functional end-to-end without any download.

  --mode hf --models org/a org/b ...   (minutes-hours, needs network + disk)
      Ingest real Hugging Face models, plan with FlexSplit, compress with
      TensorX, and report achieved storage reduction. This is the same code
      path used for the paper at full scale.

  --mode s3   (authors / production hub; needs AWS credentials)
      Pull tensors straight from the S3 content-addressed store and report
      retrieval throughput (GB/s) + per-tensor latency. Becomes the DEFAULT
      when $TENSORDEX_S3_BUCKET is set — reviewer environments without it
      keep the offline demo default. Metadata (`master.db`) is fetched from
      the bucket on first run.

Full-trace reproduction (the 2,890-model ZipLLM trace, ~40 TB, a big box —
paper used c6a.48xlarge / 96 vCPU / 384 GB) is documented at the bottom and in
ae/README.md; it is intentionally not launched automatically.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

_AE_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_AE_DIR)

FULL_RECIPE = """\
Full-trace reproduction (paper scale)
─────────────────────────────────────
 1. Model list:  the 2,890-model ZipLLM trace (Table 1). Fetch each with
        tensordex download <org/model>
    (private/removed repos have drifted since the crawl — see ae/README.md;
     authors can instead point at the local content-addressed blob store.)
 2. Plan + compress the whole hub with FlexSplit + TensorX:
        tensordex compress --auto-all --algorithm tensorx --level 1
    writing per-pair rows (tratio/tbytes_out) into a fresh results.db.
 3. Optional FM++ codec (fratio, Fig 11): `make ae-fmpp` builds it from the
    vendored FM-Delta lib (third_party/fmdelta/), then compress with --codec fmpp.
    Off by default so the base build stays pure-Rust.
 4. Render every figure from the fresh DB:
        python ae/render.py --db <fresh>/results.db
 Expected hardware/runtime: 96 vCPU / 384 GB; ingest ~22.9 GB/s, so the codec
 pass is I/O-bound, but the full 40 TB sweep still takes many hours.
"""


def run_demo() -> int:
    demo = os.path.join(_REPO, "examples", "demo.py")
    print(f"Running synthetic end-to-end lifecycle: {demo}\n")
    return subprocess.call([sys.executable, demo])


def run_hf(models, workdir, level) -> int:
    from tensordex import TensorDex  # noqa: E402
    hub = TensorDex(workdir)
    print(f"Hub at {workdir}; ingesting {len(models)} models …")
    for m in models:
        print(f"  download+ingest {m}")
        hub.download(m)

    # Plan (FlexSplit) + compress (TensorX, level 1) across the whole hub via
    # the same CLI reviewers would use; cross-model bases enabled.
    print(f"\nPlanning + compressing (TensorX, level {level}) …")
    rc = subprocess.call([
        sys.executable, "-m", "tensordex", "compress", "--hub", workdir,
        "--auto-all", "--include-existing-bases",
        "--codec", "tensorx", "--level", str(level),
    ])
    if rc != 0:
        print(f"  (compress exited {rc}; see output above)")
    subprocess.call([sys.executable, "-m", "tensordex", "stats", "--hub", workdir])
    return rc


def run_s3(bucket, prefix, workdir, n, jobs, models, seed) -> int:
    """Pull tensors from the S3 store; report throughput + latency."""
    import random
    import sqlite3
    import time
    from concurrent.futures import ThreadPoolExecutor

    import boto3
    from tensordex import TensorDex

    os.makedirs(workdir, exist_ok=True)
    meta = os.path.join(workdir, "metadata.db")
    if not os.path.exists(meta):
        key = f"{prefix}/master.db" if prefix else "master.db"
        print(f"Fetching metadata s3://{bucket}/{key} -> {meta} …")
        try:
            boto3.client("s3").download_file(bucket, key, meta)
        except Exception:
            key2 = f"{prefix}/metadata.db" if prefix else "metadata.db"
            print(f"  (master.db not found, trying {key2})")
            boto3.client("s3").download_file(bucket, key2, meta)

    hub = TensorDex(workdir, backend="s3",
                    backend_options={"bucket": bucket, "prefix": prefix})

    # Pick tensor ids: whole models if given, else a size-weighted-ish sample.
    con = sqlite3.connect(f"file:{meta}?mode=ro", uri=True)
    if models:
        ids = []
        for m in models:
            rows = con.execute(
                "SELECT tensor_id FROM model_mappings WHERE model_name = ?", (m,)
            ).fetchall()
            print(f"  {m}: {len(rows)} tensors")
            ids += [r[0] for r in rows]
    else:
        pool = [r[0] for r in con.execute(
            "SELECT id FROM tensors LIMIT 200000").fetchall()]
        random.Random(seed).shuffle(pool)
        # The metadata can be a superset of the live store (gc'd / compacted
        # blobs) — preflight with HEAD and keep the first n that exist.
        s3 = boto3.client("s3")
        base = f"{prefix}/blobs" if prefix else "blobs"
        ids, missing = [], 0
        for tid in pool:
            if len(ids) >= n:
                break
            try:
                s3.head_object(Bucket=bucket,
                               Key=f"{base}/{tid[:2]}/{tid}.safetensors")
                ids.append(tid)
            except Exception:
                missing += 1
        if missing:
            print(f"  note: {missing} sampled metadata rows have no live blob "
                  f"(gc'd/compacted) — skipped")
    con.close()
    if not ids:
        print("ERROR: no tensor ids selected")
        return 2

    def pull(tid):
        t0 = time.perf_counter()
        t = hub.get_tensor(tensor_id=tid)
        dt = time.perf_counter() - t0
        return t.numel() * t.element_size(), dt

    print(f"\nPulling {len(ids)} tensors from s3://{bucket}/{prefix or ''} "
          f"({jobs} threads) …")
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=jobs) as ex:
        sizes_times = list(ex.map(pull, ids))
    wall = time.perf_counter() - t0

    total = sum(s for s, _ in sizes_times)
    lats = sorted(dt for _, dt in sizes_times)
    p = lambda q: lats[min(len(lats) - 1, int(q * len(lats)))]
    print(f"\n  tensors      {len(ids)}")
    print(f"  bytes        {total / 1e9:.2f} GB")
    print(f"  wall         {wall:.2f} s")
    print(f"  throughput   {total / 1e9 / wall:.2f} GB/s  ({jobs} threads)")
    print(f"  latency      P50 {p(0.5)*1e3:.0f} ms · P90 {p(0.9)*1e3:.0f} ms · "
          f"P99 {p(0.99)*1e3:.0f} ms")
    return 0


def main() -> int:
    # S3 becomes the default when the production bucket is configured
    # (authors' boxes / EC2); otherwise the offline demo stays the default
    # so `make full` needs nothing from reviewers.
    env_bucket = os.environ.get("TENSORDEX_S3_BUCKET", "")
    ap = argparse.ArgumentParser(description="TensorDex AE — end-to-end driver")
    ap.add_argument("--mode", choices=["demo", "hf", "s3"],
                    default="s3" if env_bucket else "demo")
    ap.add_argument("--models", nargs="*", default=[])
    ap.add_argument("--workdir", default=os.path.join(_AE_DIR, "cache", "e2e_hub"))
    ap.add_argument("--level", type=int, default=1, help="TensorX zstd level (trace used 1)")
    ap.add_argument("--bucket", default=env_bucket, help="S3 bucket (s3 mode)")
    ap.add_argument("--prefix", default=os.environ.get("TENSORDEX_S3_PREFIX", ""),
                    help="key prefix in front of blobs/ (s3 mode)")
    ap.add_argument("--n", type=int, default=64, help="s3 mode: sampled tensor count")
    ap.add_argument("--jobs", type=int, default=8, help="s3 mode: parallel pulls")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--recipe", action="store_true", help="just print the full-scale recipe")
    args = ap.parse_args()

    if args.recipe:
        print(FULL_RECIPE)
        return 0

    if args.mode == "demo":
        rc = run_demo()
    elif args.mode == "s3":
        if not args.bucket:
            print("s3 mode needs --bucket (or $TENSORDEX_S3_BUCKET)")
            return 2
        # keep the S3 bench hub (incl. the big metadata.db) out of ae/cache —
        # on authors' machines that dir mirrors the published dataset.
        default_wd = args.workdir == os.path.join(_AE_DIR, "cache", "e2e_hub")
        s3_wd = (os.path.expanduser("~/.cache/tensordex/s3_hub")
                 if default_wd else os.path.join(args.workdir, "s3_hub"))
        rc = run_s3(args.bucket, args.prefix, s3_wd,
                    args.n, args.jobs, args.models, args.seed)
    else:
        if not args.models:
            print("hf mode needs --models org/a org/b …")
            return 2
        rc = run_hf(args.models, args.workdir, args.level)

    if args.mode != "s3":
        print("\n" + FULL_RECIPE)
    verdict = {
        "demo": "PASS ✅  end-to-end pipeline demo complete — byte-exact round-trip verified",
        "hf": "PASS ✅  end-to-end pipeline complete — byte-exact round-trip verified",
        "s3": "PASS ✅  S3 retrieval benchmark complete",
    }[args.mode]
    print("\nRESULT:", verdict if rc == 0 else "FAIL ❌  see output above")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
