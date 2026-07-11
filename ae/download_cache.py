#!/usr/bin/env python3
"""Fetch the published TensorDex AE cache from Hugging Face into `ae/cache/`.

The cache holds everything the offline tiers need:
  - `results.db`         the 11.4M-pair compression cache (Tier 0 figures)
  - `sample_blobs/`      raw tensor blobs for the sampled pairs (Tier 1 verify)
  - aux CSV/JSON/DB      inputs the chart modules read (staged layout)

Set the dataset id with `--repo` or `$TENSORDEX_AE_DATASET` (default below).
No token needed for a public dataset.

Usage:
    python ae/download_cache.py [--repo <org>/<dataset>] [--only results.db]
"""
from __future__ import annotations

import argparse
import os

_AE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.join(_AE_DIR, "cache")
DEFAULT_REPO = os.environ.get("TENSORDEX_AE_DATASET", "tensordex/tensordex-ae-cache")


def main() -> int:
    ap = argparse.ArgumentParser(description="Download the TensorDex AE cache")
    ap.add_argument("--repo", default=DEFAULT_REPO, help="HF dataset id")
    ap.add_argument("--cache", default=DEFAULT_CACHE, help="local cache dir")
    ap.add_argument("--only", nargs="*", default=None,
                    help="restrict to these path globs (e.g. results.db 'sample_blobs/*')")
    args = ap.parse_args()

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("ERROR: pip install huggingface_hub")
        return 2

    from huggingface_hub.utils import (
        GatedRepoError, RepositoryNotFoundError)

    # A dangling `ae/cache` symlink (left over from the authors' dev layout)
    # would make makedirs crash on a fresh machine — replace it with a real dir.
    if os.path.islink(args.cache) and not os.path.exists(args.cache):
        print(f"note: removing dangling symlink {args.cache}")
        os.unlink(args.cache)
    os.makedirs(args.cache, exist_ok=True)
    print(f"Downloading {args.repo} (dataset) -> {args.cache}")
    try:
        snapshot_download(
            repo_id=args.repo,
            repo_type="dataset",
            local_dir=args.cache,
            allow_patterns=args.only,       # None => everything
            max_workers=8,
        )
    except RepositoryNotFoundError:
        print(f"\nERROR: dataset '{args.repo}' not found (or not yet public).\n"
              "  • If the authors published it under another id, pass "
              "--repo <org>/<name> or set $TENSORDEX_AE_DATASET.\n"
              "  • If it is private/gated, run `huggingface-cli login` first.\n"
              "  (Authors: publish it with `python ae/upload_cache.py --create`.)")
        return 3
    except GatedRepoError:
        print(f"\nERROR: dataset '{args.repo}' is gated — request access on the "
              "Hugging Face page, then `huggingface-cli login` and retry.")
        return 3
    print("\nDone. Verify with:")
    print(f"  ls -lh {os.path.join(args.cache, 'results.db')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
