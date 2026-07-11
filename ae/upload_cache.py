#!/usr/bin/env python3
"""Publish the local `ae/cache/` to the Hugging Face dataset (authors only).

Run once, after `stage_data.py` + `build_sample_bundle.py` have populated the
cache and `results.db` has been copied/symlink-resolved into it. Needs a write
token: `huggingface-cli login` or `$HF_TOKEN`.

Usage:
    python ae/upload_cache.py --repo <org>/<dataset> [--create]
"""
from __future__ import annotations

import argparse
import os

_AE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.join(_AE_DIR, "cache")
DEFAULT_REPO = os.environ.get("TENSORDEX_AE_DATASET", "tensordex/tensordex-ae-cache")


def main() -> int:
    ap = argparse.ArgumentParser(description="Upload the TensorDex AE cache")
    ap.add_argument("--repo", default=DEFAULT_REPO, help="HF dataset id")
    ap.add_argument("--cache", default=DEFAULT_CACHE, help="local cache dir")
    ap.add_argument("--create", action="store_true", help="create the dataset repo if absent")
    ap.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    args = ap.parse_args()

    from huggingface_hub import HfApi

    # results.db must be a real file, not a dev symlink into the monorepo.
    rdb = os.path.join(args.cache, "results.db")
    if os.path.islink(rdb):
        print(f"ERROR: {rdb} is a symlink — copy the real file in before upload:\n"
              f"       cp --remove-destination $(readlink -f {rdb}) {rdb}")
        return 2

    api = HfApi(token=args.token)
    if args.create:
        api.create_repo(args.repo, repo_type="dataset", exist_ok=True)
        print(f"ensured dataset repo {args.repo}")

    print(f"Uploading {args.cache} -> {args.repo} (dataset) …")
    api.upload_large_folder(
        repo_id=args.repo,
        repo_type="dataset",
        folder_path=args.cache,
    )
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
