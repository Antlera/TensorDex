# TensorDex Makefile
# Build and development commands for TensorDex with Rust operations

.PHONY: help build install dev-install test bench clean format lint check-rust setup-dev

# Default target
help:
	@echo "TensorDex Build Commands:"
	@echo "  setup-dev     - Set up development environment"
	@echo "  build         - Build Rust extensions"
	@echo "  install       - Install package"
	@echo "  dev-install   - Install in development mode"
	@echo "  test          - Run tests"
	@echo "  bench         - Run benchmarks"
	@echo "  format        - Format code"
	@echo "  lint          - Run linting"
	@echo "  check-rust    - Check Rust code"
	@echo "  clean         - Clean build artifacts"

# Development environment setup
setup-dev:
	@echo "Setting up development environment..."
	pip install -e .[dev]
	pip install maturin>=1.0
	@echo "Development environment ready!"

# Build Rust extensions
build:
	@echo "Building Rust extensions..."
	maturin build --release
	@echo "Build complete!"

# Install package
install: build
	@echo "Installing TensorDex..."
	pip install target/wheels/*.whl --force-reinstall
	@echo "Installation complete!"

# Development installation
dev-install:
	@echo "Installing in development mode..."
	maturin develop --release
	@echo "Development installation complete!"

# Run tests
test:
	@echo "Running tests..."
	python -m pytest tests/ -v
	@echo "Tests complete!"

# Run benchmarks
bench:
	@echo "Running benchmarks..."
	python -m pytest tests/test_benchmarks.py -v --benchmark-only
	@echo "Benchmarks complete!"

# Format code
format:
	@echo "Formatting Python code..."
	black src/ tests/
	isort src/ tests/
	@echo "Formatting Rust code..."
	cargo fmt
	@echo "Formatting complete!"

# Lint code
lint:
	@echo "Linting Python code..."
	ruff check src/ tests/
	mypy src/
	@echo "Linting Rust code..."
	cargo clippy -- -D warnings
	@echo "Linting complete!"

# Check Rust code
check-rust:
	@echo "Checking Rust code..."
	cargo check
	cargo clippy -- -D warnings
	cargo fmt --check
	@echo "Rust check complete!"

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	rm -rf target/
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} +
	@echo "Clean complete!"

# Quick development cycle
quick-dev: format check-rust dev-install
	@echo "Quick development cycle complete!"

# Full development cycle
full-dev: clean format lint build test
	@echo "Full development cycle complete!"

# Check if Rust is installed
check-rust-install:
	@which rustc > /dev/null || (echo "Rust not found. Please install Rust from https://rustup.rs/" && exit 1)
	@which cargo > /dev/null || (echo "Cargo not found. Please install Rust from https://rustup.rs/" && exit 1)
	@echo "Rust installation OK"

# Setup CI environment
setup-ci: check-rust-install
	@echo "Setting up CI environment..."
	pip install maturin>=1.0
	pip install -e .[dev]
	@echo "CI environment ready!"

# Release build
release: clean format lint check-rust
	@echo "Building release..."
	maturin build --release --strip
	@echo "Release build complete!"

# Install release
install-release: release
	@echo "Installing release build..."
	pip install target/wheels/*.whl --force-reinstall
	@echo "Release installation complete!"

# ── Artifact Evaluation (SOSP) ────────────────────────────────────────
# Three tiers: figures-from-cache, sample verification, full end-to-end.
# See ae/README.md for the full guide.
.PHONY: ae-help check ae-install ae-fmpp ae-cache ae-cache-figures figures report serve verify verify-recall verify-predict bench-fig14 bench-table3 bench-table3-real bench-baselines full reproduce-all appendix ae-clean

# PY = python interpreter; N = sample size for `make verify`
# FMT = figure format; png so ae/RESULTS.md displays them inline (pdf for paper)
# Prefer `python`, but many distros ship only `python3` — fall back automatically.
PY ?= $(shell command -v python >/dev/null 2>&1 && echo python || echo python3)
N ?= 200
FMT ?= png

ae-help:
	@echo "TensorDex AE targets (start with `make check`):"
	@echo "  check       - build works? hash + tests, NO download (~30s)"
	@echo "  ae-install  - build+install the package (Rust extension)"
	@echo "  ae-fmpp     - rebuild with the FM++ codec so verify also checks fratio"
	@echo "  ae-cache-figures - Tier 0 download only (results.db + aux, skips 2GB blobs)"
	@echo "  ae-cache    - full download: results.db + blobs + aux data (HF dataset)"
	@echo "  figures     - Tier 0: render every paper figure from the cache"
	@echo "  report      - build ae/results.html (single-column, figures embedded)"
	@echo "  serve       - build + serve results.html on :8000 (tunnel for public URL)"
	@echo "  verify      - Tier 1: re-derive a random sample from raw bytes (N=$(N))"
	@echo "  verify-recall - Tier 1: TensorSketch Recall@1 experiment (Fig 12a)"
	@echo "  verify-predict - Tier 1: re-fit TensorPred from cache, held-out eval (Fig 13)"
	@echo "  bench-fig14 - run ILP/Primal-Dual/FlexSplit solvers, re-render Fig 14"
	@echo "  bench-table3 - codec throughput, synthetic pair (Table 3 methodology)"
	@echo "  bench-table3-real - Table 3's exact setup: real Qwen2.5-7B pair (~30GB dl)"
	@echo "  bench-baselines - ZipNN/OpenZL baseline throughput on real weights"
	@echo "  full        - Tier 2: run the pipeline end-to-end (demo mode)"
	@echo "  reproduce-all - everything offline in one go (Tier 0+1+2 + report)"

ae-install:
	$(PY) -m pip install . || uv pip install .

# Smoke test — confirms the extension built and the hash is correct, with no
# cache download. Run this first; if it passes, the rest is just data + compute.
check:
	@$(PY) -c "import tensordex" 2>/dev/null || \
	    { echo "ERROR: tensordex not importable with $(PY) — run \`make ae-install\` (or activate the venv you installed into)"; exit 1; }
	@$(PY) -c "import xxhash, pytest" 2>/dev/null || \
	    { echo "ERROR: AE deps missing — run \`$(PY) -m pip install -r ae/requirements-ae.txt\`"; exit 1; }
	@$(PY) -c "from tensordex import _ops; import xxhash; b=b'tensordex'; \
	assert _ops.content_hash(b)==xxhash.xxh128_hexdigest(b); print('OK  build + XXH3-128 hash')"
	@$(PY) -m pytest tests/ -q --no-header
	@echo "OK  build verified — no download needed for this check"

# Rebuild WITH the FM++ codec (links the vendored FM-Delta lib) so `verify`
# also re-derives fratio, the paper's 70.5% number. Needs a C++ toolchain (to link
# libstdc++). Ensures maturin is present, then rebuilds into the active venv.
# Alternative (pip): pip install . --force-reinstall \
#     --config-settings=build-args="--features fmpp"
ae-fmpp:
	@command -v maturin >/dev/null 2>&1 || $(PY) -m pip install maturin || uv pip install maturin
	@command -v patchelf >/dev/null 2>&1 || $(PY) -m pip install patchelf || uv pip install patchelf || true
	maturin develop --release --features fmpp

# Full cache (results.db + sample_blobs + aux); Tier 0/1/2.
ae-cache:
	$(PY) ae/download_cache.py

# Tier-0-only: skip the 2 GB sample_blobs (needed only by `make verify`).
ae-cache-figures:
	$(PY) ae/download_cache.py --only results.db 'data/*' 'model_hub_crawl/*' \
	    'tests/*' 'compression_data/*' 'model_level_reduction/*'

figures:
	$(PY) ae/render.py --format $(FMT)

verify:
	$(PY) ae/verify_sample.py --n $(N)

verify-recall:
	$(PY) ae/verify_recall.py

# Live-store check: sample blobs from an S3 bucket and assert every object key
# equals XXH3-128 of the tensor bytes it stores (content addressing, Table 2).
# Usage: make verify-s3 BUCKET=my-bucket [S3_PREFIX=hub1] [N=200]
verify-s3:
	$(PY) ae/verify_s3_ids.py --bucket $(BUCKET) $(if $(S3_PREFIX),--prefix $(S3_PREFIX)) --n $(N)

# Fig 13 — re-fit the TensorPred reduction-ratio predictor from the cache
# (OLS on bcs_dist -> aratio), evaluate on a held-out split, cross-check against
# the stored column, and re-render the figure. No pre-recorded coefficients.
verify-predict:
	$(PY) ae/fit_predict.py

# Fig 14 — actually run the facility-location solvers (ILP / Primal-Dual /
# FlexSplit) across model counts, write the parsed CSV, and re-render the charts.
# ILP needs gurobipy + a license (free academic); skipped automatically if absent.
bench-fig14:
	$(PY) ae/bench/run_scaling.py
	$(PY) ae/render.py --only algo_bench_q_proj,algo_bench_v_proj --format $(FMT)

# Table 3 — codec throughput, measured the way the paper measured it: pure
# codec compute over 2 MB chunks, all cores, shipped kernels (zstd L1).
# `bench-table3` = synthetic pair, no download. `bench-table3-real` = the
# paper's exact setup (real Qwen2.5-7B vs -Instruct, ~30 GB download, ~32 GB
# RAM) — reproduces 22.9 GB/s compress @ 59.4% reduction on a c6a.48xlarge.
bench-table3:
	cargo run --release --example table3_bench

# External baselines on the same real model: ZipNN + OpenZL (optional tools,
# skipped with an install hint when absent) and ZipLLM's official BitX
# (vendored from github.com/ds2-lab/ZipLLM, Apache-2.0 — see
# third_party/zipllm_bitx/).
bench-baselines:
	$(PY) ae/bench_baselines.py
	@BASE=$$($(PY) -c "from huggingface_hub import snapshot_download; print(snapshot_download('Qwen/Qwen2.5-7B', allow_patterns=['*.safetensors']))") && \
	TGT=$$($(PY) -c "from huggingface_hub import snapshot_download; print(snapshot_download('Qwen/Qwen2.5-7B-Instruct', allow_patterns=['*.safetensors']))") && \
	cargo run --release --example zipllm_bitx_bench -- --base-model $$BASE --target-model $$TGT

bench-table3-real:
	@BASE=$$($(PY) -c "from huggingface_hub import snapshot_download; print(snapshot_download('Qwen/Qwen2.5-7B', allow_patterns=['*.safetensors']))") && 	TGT=$$($(PY) -c "from huggingface_hub import snapshot_download; print(snapshot_download('Qwen/Qwen2.5-7B-Instruct', allow_patterns=['*.safetensors']))") && 	cargo run --release --example table3_bench -- --base-model $$BASE --target-model $$TGT

# Build a self-contained single-column HTML results page (ae/results.html) from
# the already-rendered figures (run `make figures` first). Fast — no re-render.
report:
	$(PY) ae/build_report.py

serve: report
	@echo "Serving http://localhost:8000/results.html  (Ctrl+C to stop)"
	@echo "Public URL: cloudflared tunnel --url http://localhost:8000"
	cd ae && $(PY) -m http.server 8000

full:
	$(PY) ae/run_full.py

# One command → every offline result: all figures (Tier 0), sample verification,
# the Recall@1 and TensorPred experiments (Tier 1), the end-to-end demo (Tier 2),
# and the HTML report. Needs `make ae-cache` first. ~30–45 min, CPU only.
reproduce-all:
	@echo "==== [1/6] Tier 0 — render all figures ===================="
	$(PY) ae/render.py --format $(FMT)
	@echo "==== [2/6] Tier 1 — sample verification (N=$(N)) =========="
	$(PY) ae/verify_sample.py --n $(N)
	@echo "==== [3/6] Tier 1 — TensorSketch Recall@1 (Fig 12a) ======="
	$(PY) ae/verify_recall.py
	@echo "==== [4/6] Tier 1 — TensorPred re-fit (Fig 13) ============"
	$(PY) ae/fit_predict.py
	@echo "==== [5/6] Tier 2 — end-to-end pipeline demo =============="
	$(PY) ae/run_full.py
	@echo "==== [6/6] report — ae/results.html ======================="
	$(PY) ae/build_report.py
	@echo
	@echo "ALL DONE ✅  figures in ae/figures/, report at ae/results.html"
	@echo "Optional extras: make bench-fig14 (solver benchmark), make ae-fmpp && make verify (FM++ bit-exact)"

# Build the Artifact Appendix PDF (authors; needs pdflatex).
appendix:
	cd ae/appendix && pdflatex -interaction=nonstopmode appendix.tex >/dev/null && \
	pdflatex -interaction=nonstopmode appendix.tex >/dev/null && \
	echo "OK  ae/appendix/appendix.pdf"

ae-clean:
	rm -rf ae/figures
