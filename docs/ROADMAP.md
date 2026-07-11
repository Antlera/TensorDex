# TensorDex — Positioning & Roadmap

## Positioning

TensorDex is **tensor-centric, content-addressable storage for AI model hubs**. It sits between *tensor producers* (HuggingFace, training runs, local checkpoints) and *tensor consumers* (transformers, vLLM, analysis scripts), deduplicating and delta-compressing at the **tensor level** and reconstructing exact `.safetensors` on demand.

Two ideas set it apart:

- **TensorSketch** — a compact per-tensor fingerprint that estimates pairwise compressibility *without reading the weights*, so similarity search over the whole hub is cheap.
- **FlexSplit** — an incremental planner that organizes tensors into multi-center clusters and picks high-quality `(target, base)` pairs, opening a new base only where it pays off.

## Where it fits

| Neighbor | Unit | Relationship |
|---|---|---|
| **transformers / vLLM** | `nn.Module`, runtime | Downstream consumers — TensorDex is a data source, not a replacement. |
| **huggingface_hub** | whole files | Complementary — file-level dedup can't see shared tensors; TensorDex dedups one level down. |
| **Xet / chunk stores** | byte chunks | Closest in spirit, but chunk-level dedup is tensor-blind. Redundancy in fine-tunes emerges at the *tensor* level — which TensorDex targets directly. |

**Result (TensorDex paper):** on a 40 TB trace of 2,890 real HuggingFace models, TensorDex reduces storage by **70.5%** (3.39× smaller, 37% below the prior state of the art), losslessly, at **22.9 / 28.4 GB/s** compress / decompress.

## Architecture

Python orchestrates I/O and exposes the API/CLI; the Rust extension `tensordex._ops` owns all persistent state and data processing.

```
Rust (src/rust/ — tensordex._ops)
  ingest/        atomic pipeline: mmap safetensors → hash → dedup → sketch → blob → SQL
  metadata/      SQLite MetadataStore (source of truth) + TensorSketch FingerprintStore
  compression/   FlexSplit planner → plan → execute
  kernels/       sketch · bitx · tensorx · xor
  codec/         generic zstd / lz4 fallbacks
  resolvers/     blob backends (local, S3)
  transfer.rs    Rust HTTP transfer backend

Python (src/tensordex/)
  core/engine    TensorDex: ingest / compress / pull / get_tensor / gc / ...
  integrations/  HuggingFace download → ingest
  client + server  FastAPI read-only repo + remote pull (blob-cache reuse)
  cli            thin Typer layer, 1:1 to engine methods
```

**Layering rule:** Python holds no durable state — the SQLite `MetadataStore` is the single source of truth for tensor metadata, model mappings, and the delta-base graph (`tensor_deltas`).

## Status — shipped

- **Ingest & dedup** — atomic Rust ingest over memory-mapped safetensors; XXH3-128 content addressing; TensorSketch fingerprints computed in the same pass.
- **Compression** — FlexSplit planner; the `tensorx` delta codec; `compress --auto-all`, pairwise `compress`, and `--bundle` star compression for a checkpoint group.
- **Reconstruction** — `pull` rebuilds exact `.safetensors` (transparent delta decode); `--verify` re-hashes every tensor; `--max-shard-size` writes HF-style shards.
- **CLI** — `init, download (--revision), ls, info, stats, get, pull, compress, rm, gc, serve, env, whoami` (+ `demo-transfer`).
- **Remote** — read-only HTTP `serve` + remote `pull` with blob caching; optional Rust transfer backend; S3 blob backend.

## Roadmap — next

Directional, not commitments; priority follows real usage.

- **Similarity & lineage as first-class CLI** — expose the fingerprint index via `similar <model>:<param>` and a cross-model shared-tensor `lineage <model>` graph. *(The Rust arena already supports it; only the CLI/engine surface is missing.)*
- **Online clustering at hub scale** — incremental FlexSplit that re-plans as new models arrive, instead of batch `--auto-all`.
- **Broader coverage** — more dtypes / quantized formats; continued codec and throughput tuning.
- **Producer side** — `upload` / push to a remote hub; a richer remote protocol with auth and multi-tenant access.
- **Consumer integration** — stream tensors directly into vLLM / transformers loaders without a full `pull`.
- **Observability** — hub-wide dedup / compression metrics surfaced through `stats`.
