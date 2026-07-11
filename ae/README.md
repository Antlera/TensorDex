# TensorDex — SOSP Artifact Evaluation

This directory is the reproducibility package for *TensorDex: A Compact,
Tensor-Centric Storage System for Modern AI Models* (SOSP '26). A reviewer can
confirm the paper's results at three levels of effort, from a few minutes to a
full end-to-end run:

| Tier | What it shows | Effort | Needs |
|------|---------------|--------|-------|
| **0 — Figures from cache** | every paper figure re-plotted from the authors' `results.db` | ~6 GB download + ~10 min render | CPU |
| **1 — Sample verification** | a random subset of the cache re-derived bit-for-bit from raw tensor bytes — proving the cache is genuine, not hand-drawn | +~2 GB download + ~10 min | CPU |
| **2 — Full end-to-end** | the pipeline itself (ingest → sketch → FlexSplit → TensorX) run from scratch | seconds (demo) → hours (full trace) | build toolchain; big box for the full trace |

The idea: Tier 0 shows the figures come straight from the data; Tier 1
proves that data is real by recomputing it from the shipped raw bytes on a
random sample you choose the seed for; Tier 2 lets you run the whole thing.

### What's where

| File | Purpose |
|---|---|
| [`RESULTS.md`](RESULTS.md) | **all reproduced figures in one place**, laid out to mirror the paper's evaluation section — `make figures` regenerates them in place |
| [`FIGURE_MAP.md`](FIGURE_MAP.md) | index only: paper figure number ↔ chart id ↔ data source |
| [`appendix/appendix.pdf`](appendix/appendix.pdf) | the Artifact Appendix: claims table + per-experiment walkthrough with expected outputs |
| `results.html` | `RESULTS.md` as one self-contained page for the browser (`make report`) |
| `figures/` | where rendered charts land |

### Requirements at a glance

> **Test machine:** please comment on HotCRP with your SSH public key and a
> time window, and we will provision the paper's eval box (AWS
> **c6a.48xlarge**) for you.

| | |
|---|---|
| Hardware | any x86_64 Linux box · 8 GB RAM · ~15 GB disk · no GPU |
| Software | Rust ≥ 1.78 + Python ≥ 3.8 — or just Docker (see below) |
| Time | kick-the-tires ~5 min · full offline reproduction ~45 min |

### Kick the tires (~5 minutes, no download)

```bash
# in a virtualenv — see "0. Install" below if `python3 -m venv` is missing
pip install . && pip install -r ae/requirements-ae.txt
make check       # builds nothing else, downloads nothing:
                 #   "OK  build + XXH3-128 hash"  +  the unit tests pass
make full        # Tier 2 demo: ingest → dedup → delta → byte-exact pull, offline
```

Or with Docker (no local toolchain at all):

```bash
docker build -t tensordex-ae .
docker run --rm tensordex-ae                # = make check inside the container
docker run -it --rm -v tensordex-cache:/tensordex/ae/cache tensordex-ae bash
# inside the container:  make ae-cache && make reproduce-all
```

### One command for everything offline

After `make ae-cache`, a single target runs Tier 0 + Tier 1 + the Tier 2 demo
and builds the HTML report (~30–45 min, CPU only):

```bash
make reproduce-all
```

### Paper claim → how to reproduce

| Claim (paper) | Command | Expected |
|---|---|---|
| **70.5 %** storage reduction — TensorDex-FM++ (Fig 1, 11) | `make ae-fmpp && make verify` | `fratio` bit-exact vs cache; `codec_storage_reduction.png` → 0.29× |
| **65.1 %** reduction — TensorDex-TX | `make verify` | `tratio` bit-exact (100 %) |
| Tensor dedup by content hash (§5.1) | `make verify` | content ids 100 % exact |
| **Recall@1 = 1.00** — TensorSketch (Fig 12a) | `make verify-recall` | Recall@1 = 1.000 |
| Reduction-ratio prediction, MAE ~1 % (Fig 13) | `make verify-predict` | re-fits TensorPred from cache; **held-out** MAE 1.11 %, Pearson 99.3 % |
| **FlexSplit near-optimal & fast vs ILP** (Fig 14) | `make bench-fig14` | runs ILP/PD/FlexSplit solvers; FlexSplit ≈ ILP ratio at ~constant time |
| Codec throughput (Table 3, Fig 1-right) | `make bench-table3-real` | the paper's exact setup, real Qwen2.5-7B pair: 22.9/28.4 GB/s at 59.4 % reduction (synthetic variant: `bench-table3`; external baselines: `bench-baselines`) |

Start with `make check` (no download) to confirm the build, then pick a tier.
The Artifact Appendix — the claims-to-experiment map as a single PDF — is at
[`appendix/appendix.pdf`](appendix/appendix.pdf); `make appendix` rebuilds it.

---

## 0. Install

Needs **Rust ≥ 1.78** and **Python ≥ 3.8**. From the repo root:

```bash
# Rust toolchain, if you don't have one:
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y && source "$HOME/.cargo/env"

# A virtualenv. Stock Ubuntu/Debian ships neither venv nor a usable system
# pip (PEP 668) — install python3-venv once, or use uv instead:
sudo apt-get install -y python3-venv     # skip if `python3 -m venv` already works
python3 -m venv .venv && source .venv/bin/activate

# Optional but saves ~5 GB: the artifact never touches a GPU, so CPU-only
# torch wheels suffice. Install them first and pip keeps them:
pip install torch --index-url https://download.pytorch.org/whl/cpu

pip install .            # builds the Rust extension
pip install -r ae/requirements-ae.txt
```

Confirm the build before downloading anything — this checks the XXH3-128
content hash (the hash every published id is keyed on) and runs the offline
unit tests:

```bash
make check      # ~30 s, no download: "OK build + XXH3-128 hash" + unit tests pass
```

## 1. Get the cache

The paper's evaluation runs over a ~40 TB, 2,890-model trace — re-running it
end-to-end takes days on a large machine, so the artifact ships the trace's
pre-computed per-pair results instead, together with enough raw tensor bytes
to verify them by random sampling (Tier 1). `results.db` (5.6 GB), the
sampled raw-tensor blobs, and the chart inputs live in a public Hugging Face
**dataset** (no token needed):

```bash
make ae-cache-figures    # Tier 0 only — results.db + aux (~6 GB, skips the blobs)
make ae-cache            # everything, incl. the Tier-1 blobs (~8.3 GB)
# or a specific dataset id:
python ae/download_cache.py --repo <org>/tensordex-ae-cache
```

If the dataset isn't published yet, the download prints a clear message with the
id to override (`--repo` / `$TENSORDEX_AE_DATASET`).

This populates `ae/cache/`:

```
ae/cache/
  results.db                 11.4M-pair compression cache (every figure's numbers)
  sample_blobs/<xx>/<yy>/<id>.safetensors   raw tensors for the Tier-1 sample
  data/tensordb_s3/metadata.db              slim: tensor sizes + model→tensor map
  model_hub_crawl/  tests/output/  model_level_reduction/  compression_data/   chart inputs
```

---

## Tier 0 — regenerate every figure from the cache

```bash
make figures                          # → python ae/render.py  → ae/figures/*.png
```

Renders **38 charts** covering Fig 1, 2, 4, 11, 12, 13, 14, 15, 16 and Table 3
(PNG by default; `make figures FMT=pdf` for paper-quality). Open
[`RESULTS.md`](RESULTS.md) afterwards — it lays out the results with each figure
embedded inline, so the plots appear as soon as they're generated. Each
`ae/figures/<id>.png` maps to a paper figure in [`FIGURE_MAP.md`](FIGURE_MAP.md).
Charts read only the cache — the ratios you see are exactly the numbers plotted in
the paper. (Fig 6 needs the full research repo and is out of scope; a handful of
throughput/QPS panels replot values measured on the eval box — Tier 2 re-measures
them.)

Spot-check against the paper: `codec_storage_reduction.png` → 0.29× (70.5 % saved);
`reduction_violin_by_family.png` → Fig 11c; `bcs_recall.png` → Fig 12a.

## Tier 1 — prove the cache is real (sample verification)

The cache has 11.4 M rows; instead of recompressing all of them, draw a random
sample and re-derive each row from the raw bytes we ship. The shipped blob
bundle covers ~342 cache rows (~2 GB of raw tensors) — any `--n` at or above
that verifies the whole covered set:

```bash
make verify                           # → python ae/verify_sample.py --n 200
python ae/verify_sample.py --n 300 --seed 7      # your own seed / size
```

For every sampled `(target, base)` pair it runs three independent checks, all
with the freshly built `tensordex._ops`:

1. Content id — re-hash both tensors' raw bytes (XXH3-128) and assert the
   digest equals the `target_id` / `base_id` the cache is keyed on. The id *is*
   the hash, so this ties each row to real tensor content.
2. TensorX ratio — recompute the delta compression ratio (zstd level 1) and
   assert it equals the cached `tratio`.
3. FM++ ratio — if the extension was built with FM++ (`make ae-fmpp`),
   also recompute the FM++ delta and assert it equals the cached `fratio`, the
   codec behind the paper's 70.5 % result.

Expected: content ids 100 % exact, and TensorX and FM++ ratios ~100 %
bit-exact (`Δ = 0.0`). `results.db` accreted over months, so a small fraction
of legacy rows predate the final codec — those are printed with their timestamp
and the run still PASSes (content ids match regardless). Pick any seed: a passing
sample you chose is the evidence the whole cache is genuine.

To include the FM++ check, build the optional codec first (links the vendored
FM-Delta lib in `third_party/fmdelta/`; x86_64-linux):

```bash
make ae-fmpp          # maturin develop --release --features fmpp
make verify           # now re-derives fratio too
```

### Recall@1 experiment (Fig 12a)

```bash
make verify-recall    # → python ae/verify_recall.py   (needs hnswlib)
```

Reproduces the similarity-search claim: TensorSketch + an HNSW index selects the
same delta base as exact brute-force search over the sketches, so the planner
never reads full tensors. A faithful port of the authors' internal ANN-vs-BCS benchmark (its constants
live in `tests/test_flexsplit.py`): it recomputes BCS fingerprints from the shipped
blobs with `tensordex._ops`, runs greedy base-selection twice (brute-force =
ground truth, HNSW = approximate), and reports the fraction of tensors assigned
to the same base. Expected: Recall@1 = 1.00.

### Reduction-ratio prediction (Fig 13)

```bash
make verify-predict   # → python ae/fit_predict.py
```

Reproduces the prediction-accuracy claim as an experiment, not a replot. A
prediction figure is by nature *fit a model on the computed results and measure
it there*, so this re-derives TensorPred from the cache: its hybrid model is
linear in four coefficients (`cr = c0·p + c1·t + c2·p·t + c3`,
`p = clip(bcs_dist, 0, 0.5)`, `t = 8·H(p)`), so the fit is ordinary least
squares over the cached `(bcs_dist, aratio)` pairs; no coefficients are
pre-recorded. (`aratio` is the dense per-pair delta-ratio column computed
during development as TensorPred's fitting and evaluation set — 5.77 M pairs,
about 6× the FM++ coverage; the dataset card has a column guide.) It fits on
a random train half and evaluates on the held-out half. Expected: held-out
MAE 1.11 %, median 0.79 %, Pearson 99.3 % (the paper's numbers), and the
recovered predictions match the stored `pred_ratio` column to about 1e-4 —
that column is this model, not a hand-tuned constant. The figure `pred_vs_real_ratio.png` is re-rendered from the fresh fit.

### FlexSplit vs ILP / Primal-Dual (Fig 14)

```bash
make bench-fig14      # ~15–25 min; ILP needs gurobipy + a license (free academic)
```

Runs the three facility-location solvers for real — ILP (Gurobi, optimal),
Primal-Dual, and FlexSplit — over the cached pairwise ratios in `results.db`,
sweeping model counts, and re-plots `algo_bench_{q,v}_proj`. It reproduces the
paper's finding: FlexSplit stays within a few points of the ILP optimum at
near-constant time while ILP grows super-linearly. `gurobipy` is optional — if
it's absent the ILP curve is omitted (Primal-Dual + FlexSplit still run).
Code: `ae/bench/` (a faithful copy of the paper's `algorithms/` solvers).

## Tier 2 — run the pipeline end-to-end

```bash
make full                             # → python ae/run_full.py   (synthetic, ~seconds, offline)
python ae/run_full.py --mode hf --models Qwen/Qwen2.5-0.5B Qwen/Qwen2.5-0.5B-Instruct
python ae/run_full.py --recipe        # print the full 2,890-model trace recipe
```

`--mode demo` runs a synthetic base+fine-tune through the whole lifecycle
(ingest, dedup, FlexSplit plan, TensorX compress, integrity-checked pull) — a
fast proof the pipeline is functional with no download. `--mode hf` does the same
on real Hugging Face models and reports achieved storage reduction. The full
40 TB / 2,890-model trace (paper hardware: c6a.48xlarge, 96 vCPU, 384 GB) is
documented by `--recipe`.

---

## Badge → evidence map

| Badge | Where it's supported |
|---|---|
| **Available** | public GitHub repo + published HF dataset (cache); archived with DOI on Zenodo for the final version; Apache-2.0 license; this README references the paper |
| **Functional** | `make check` (build + unit tests, no download); `make full` (end-to-end demo); every dependency pinned in `pyproject.toml` / `ae/requirements-ae.txt`; Dockerfile for a zero-setup environment; the repo contains all code + data used by the paper's figures and nothing else |
| **Reproduced** | one command per experiment (claims table above), or `make reproduce-all` for everything offline; each script prints an explicit PASS/FAIL verdict plus the paper's expected numbers; `make report` renders the human-readable results page |

## Caveats

- **Both codecs are re-derivable.** TensorX (`tratio`, 65.1 %) is pure Rust
  and re-derived by `make verify` out of the box. FM++ (`fratio`, the 70.5 %
  number) extends the external FM-Delta C++ coder — vendored prebuilt in
  `third_party/fmdelta/` (x86_64-linux) and enabled with `make ae-fmpp`; once
  built, `make verify` re-derives `fratio` bit-exact too. On other platforms, or
  to avoid the prebuilt binary, rebuild the lib from source per
  `third_party/fmdelta/README.md`. (Note: confirm FM-Delta's license before
  redistributing the `.a`.)
- **HuggingFace drift.** Many fine-tune repos in the trace have since been made
  private or re-uploaded, so re-downloading "the same model" can yield different
  bytes. That is exactly why Tier 1 verifies against **local shipped blobs**, not
  live HF — the check is offline and deterministic.
- **Hash provenance.** Content ids are **XXH3-128** digests of the raw tensor
  bytes throughout, so a fresh ingest lands on the same ids as `results.db`
  and the metadata store (paper Table 2) — `make check` cross-verifies the
  hash against an independent library.
