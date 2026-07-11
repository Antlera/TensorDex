#!/usr/bin/env python3
"""Verify that S3 blob keys ARE the XXH3-128 content hash of their tensor.

TensorDex stores every tensor content-addressed: the object key is
`[prefix/]blobs/<id[:2]>/<id>.safetensors` (S3 backend; synced local stores may
use a 2-level `blobs/<xx>/<yy>/<id>.safetensors` shard — both are handled) and
`id == XXH3-128(raw little-endian tensor bytes)`. This script samples objects
from a live bucket, re-hashes each tensor with the *installed* package's Rust
kernel (`tensordex._ops.content_hash`), and asserts key == hash — proving the
store on S3 is genuinely content-addressed with the paper's hash (Table 2).

    python ae/verify_s3_ids.py --bucket <bucket> [--prefix hub1] [--n 200]
                               [--seed 0] [--endpoint-url URL] [--all]

Credentials come from the usual AWS chain (env / ~/.aws / instance role).
Exit codes: 0 = PASS, 1 = mismatches found, 2 = setup error.
"""
from __future__ import annotations

import argparse
import os
import random
import sys

_AE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _AE_DIR)

from _blobs import content_id, real_tensor_key  # noqa: E402


def iter_blob_keys(client, bucket: str, prefix: str, cap: int):
    """Yield up to `cap` blob keys under [prefix/]blobs/."""
    base = f"{prefix.strip('/')}/blobs/" if prefix else "blobs/"
    paginator = client.get_paginator("list_objects_v2")
    seen = 0
    for page in paginator.paginate(Bucket=bucket, Prefix=base):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith(".safetensors"):
                yield key
                seen += 1
                if seen >= cap:
                    return


def tensor_raw_bytes(body: bytes) -> bytes:
    """Raw little-endian bytes of the real tensor in a safetensors payload."""
    import torch
    from safetensors.torch import load
    tensors = load(body)
    key = real_tensor_key(list(tensors.keys()))
    if key is None:
        raise ValueError("no tensor key in payload")
    t = tensors[key]
    return t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()


def main() -> int:
    ap = argparse.ArgumentParser(description="Verify S3 blob keys == XXH3-128(content)")
    ap.add_argument("--bucket", required=True)
    ap.add_argument("--prefix", default="", help="key prefix in front of blobs/")
    ap.add_argument("--n", type=int, default=200, help="sample size")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--all", action="store_true", help="verify every listed blob")
    ap.add_argument("--max-list", type=int, default=200_000,
                    help="cap on keys listed before sampling")
    ap.add_argument("--endpoint-url", default=None, help="S3-compatible endpoint")
    ap.add_argument("--region", default=None)
    args = ap.parse_args()

    try:
        import boto3
    except ImportError:
        print("ERROR: pip install boto3  (or `pip install '.[s3]'`)")
        return 2

    client = boto3.client("s3", region_name=args.region,
                          endpoint_url=args.endpoint_url)

    print(f"Listing s3://{args.bucket}/{args.prefix + '/' if args.prefix else ''}blobs/ "
          f"(up to {args.max_list:,} keys) ...")
    keys = list(iter_blob_keys(client, args.bucket, args.prefix, args.max_list))
    if not keys:
        print("ERROR: no .safetensors blobs found under that bucket/prefix")
        return 2
    print(f"  {len(keys):,} blobs listed")

    if not args.all and len(keys) > args.n:
        rng = random.Random(args.seed)
        keys = rng.sample(keys, args.n)
    print(f"Verifying {len(keys)} blobs (seed={args.seed})\n")

    ok = bad = 0
    for i, key in enumerate(keys, 1):
        expect = key.rsplit("/", 1)[-1][: -len(".safetensors")]
        body = client.get_object(Bucket=args.bucket, Key=key)["Body"].read()
        got = content_id(tensor_raw_bytes(body))
        if got == expect:
            ok += 1
            print(f"  [{i:>4}/{len(keys)}] OK   {expect[:12]}…  key == XXH3-128(bytes)")
        else:
            bad += 1
            print(f"  [{i:>4}/{len(keys)}] FAIL {key}\n"
                  f"        key id  = {expect}\n"
                  f"        hash    = {got}")

    print("\n" + "=" * 64)
    print(f"content-addressing check : {ok}/{len(keys)} keys == XXH3-128(content)")
    print("RESULT:", "PASS ✅  S3 store is content-addressed with the paper's hash"
          if bad == 0 else f"FAIL ❌  {bad} mismatched keys")
    print("=" * 64)
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
