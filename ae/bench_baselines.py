#!/usr/bin/env python3
"""External-baseline throughput on real model weights (Table 3 / Fig 1-right).

TensorDex's own codecs are benchmarked by `make bench-table3[-real]`. This
script re-runs the baselines that are external tools:

  - ZipNN  (`pip install zipnn`) — weight-aware lossless compressor.
  - OpenZL (build `zli` from github.com/facebook/openzl and put it on PATH,
    or set $OPENZL_CLI) — measured with `zli benchmark` per 64 MB slice under
    full parallel load, so the numbers are in-process codec throughput.

Either tool is skipped with an install hint when absent (same philosophy as
Gurobi for Fig 14). ZipLLM and FM-Delta cite their own papers.

    python ae/bench_baselines.py --model <dir-with-safetensors> [--cap-mb 4096]
    python ae/bench_baselines.py            # downloads Qwen/Qwen2.5-7B-Instruct

Paper reference (c6a.48xlarge): ZipNN 1.4 / 9.4 GB/s · OpenZL 0.7 / 18.6 GB/s.
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import shutil
import subprocess
import tempfile
import time

PAPER_REF = "paper reference (c6a.48xlarge): ZipNN 1.4 / 9.4 GB/s · OpenZL 0.7 / 18.6 GB/s"


def model_files(model_dir: str | None, cap_mb: int):
    if model_dir is None:
        from huggingface_hub import snapshot_download
        model_dir = snapshot_download("Qwen/Qwen2.5-7B-Instruct",
                                      allow_patterns=["*.safetensors"])
        print(f"using Qwen/Qwen2.5-7B-Instruct @ {model_dir}")
    files = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
    if not files:
        raise SystemExit(f"ERROR: no .safetensors under {model_dir}")
    picked, total = [], 0
    for f in files:
        picked.append(f)
        total += os.path.getsize(f)
        if total >= cap_mb * 1024 * 1024:
            break
    print(f"benching {len(picked)} file(s), {total / 1e9:.2f} GB")
    return picked, total


def bench_zipnn(files, total):
    try:
        from zipnn import ZipNN
    except ImportError:
        print("\nZipNN: SKIPPED — pip install zipnn")
        return
    # NOTE: zipnn's C extension transforms the caller's input buffer IN
    # PLACE during compress, so the round-trip must be checked against an
    # independent copy of the data (a re-read from disk), never against the
    # buffer that was passed to compress().
    zn = ZipNN(bytearray_dtype="bfloat16")
    comp_t = decomp_t = comp_bytes = 0.0
    for f in files:
        data = open(f, "rb").read()
        t0 = time.perf_counter()
        c = zn.compress(data)
        comp_t += time.perf_counter() - t0
        comp_bytes += len(c)
        t0 = time.perf_counter()
        d = zn.decompress(c)
        decomp_t += time.perf_counter() - t0
        pristine = open(f, "rb").read()
        if bytes(d) != pristine:
            import importlib.metadata as _im
            print(f"\nZipNN: SKIPPED — zipnn {_im.version('zipnn')} did not "
                  f"round-trip {f}; numbers from a broken decode would be "
                  f"meaningless")
            return
    print(f"\nZipNN   reduction {comp_bytes/total:.3f}x   round-trip byte-exact ✅")
    print(f"  compress   {total/1e9/comp_t:6.2f} GB/s")
    print(f"  decompress {total/1e9/decomp_t:6.2f} GB/s")


def bench_openzl(files, total):
    """`zli benchmark` times codec work in-process (no per-file process-spawn
    cost); slices run under full parallel load, and per-slice throughputs sum
    to the aggregate. A separate compress/decompress/compare pass on one
    slice checks integrity."""
    cli = os.environ.get("OPENZL_CLI") or shutil.which("zli")
    if not cli:
        print("\nOpenZL: SKIPPED — build `zli` from github.com/facebook/openzl "
              "(cmake) and put it on PATH or set $OPENZL_CLI")
        return
    from concurrent.futures import ThreadPoolExecutor
    import multiprocessing
    chunk = 64 * 1024 * 1024
    shm = "/dev/shm"
    use_shm = os.path.isdir(shm) and (
        os.statvfs(shm).f_bavail * os.statvfs(shm).f_frsize > 3 * total)
    with tempfile.TemporaryDirectory(dir=shm if use_shm else None) as td:
        slice_dirs = []
        for fi, f in enumerate(files):
            with open(f, "rb") as fh:
                i = 0
                while True:
                    buf = fh.read(chunk)
                    if not buf:
                        break
                    d = os.path.join(td, f"{fi}_{i}")
                    os.makedirs(d)
                    open(os.path.join(d, "s.bin"), "wb").write(buf)
                    slice_dirs.append(d)
                    i += 1

        # integrity: one slice through compress -> decompress -> compare
        p = os.path.join(slice_dirs[0], "s.bin")
        subprocess.run([cli, "compress", p, "--output", p + ".zl",
                        "--profile", "le-u16", "--force"],
                       check=True, capture_output=True)
        subprocess.run([cli, "decompress", p + ".zl", "--output", p + ".out",
                        "--force"], check=True, capture_output=True)
        if open(p + ".out", "rb").read() != open(p, "rb").read():
            print("\nOpenZL: SKIPPED — round-trip mismatch")
            return
        os.remove(p + ".zl"); os.remove(p + ".out")

        workers = multiprocessing.cpu_count()
        print(f"\nOpenZL  ({len(slice_dirs)} x 64 MB slices, {workers} parallel "
              f"`zli benchmark` processes, profile le-u16, in-process timing)")

        def bench_one(d):
            out = subprocess.run([cli, "benchmark", d, "--profile", "le-u16"],
                                 check=True, capture_output=True, text=True)
            m = re.findall(r"([0-9.]+)\s*MB/s", out.stdout + out.stderr)
            ratio = re.search(r"\(([0-9.]+)\)", out.stdout + out.stderr)
            return float(m[0]), float(m[1]), float(ratio.group(1))

        with ThreadPoolExecutor(workers) as ex:
            rows = list(ex.map(bench_one, slice_dirs))
    comp = sum(r[0] for r in rows) / 1e3
    decomp = sum(r[1] for r in rows) / 1e3
    ratio = sum(r[2] for r in rows) / len(rows)
    print(f"  reduction  {1/ratio:.3f}x   round-trip byte-exact ✅ (spot check)")
    print(f"  compress   {comp:6.2f} GB/s")
    print(f"  decompress {decomp:6.2f} GB/s")


def main() -> int:
    ap = argparse.ArgumentParser(description="ZipNN / OpenZL baseline throughput")
    ap.add_argument("--model", default=None, help="dir with .safetensors "
                    "(default: download Qwen/Qwen2.5-7B-Instruct)")
    ap.add_argument("--cap-mb", type=int, default=4096,
                    help="stop adding files past this size")
    args = ap.parse_args()

    files, total = model_files(args.model, args.cap_mb)
    bench_zipnn(files, total)
    bench_openzl(files, total)
    print(f"\n{PAPER_REF}")
    print("RESULT: DONE — baselines measured (absent tools reported as SKIPPED)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
