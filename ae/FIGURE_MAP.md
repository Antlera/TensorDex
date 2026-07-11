# Paper figure → chart → output

`python ae/render.py` writes each chart below to `ae/figures/<chart_id>.pdf`
(use `--only <id>` for one). Every ratio-bearing figure is driven by the cached
`results.db` (+ the staged CSV/JSON inputs in `ae/cache/`). A few panels report
values measured on the eval machine (throughput/QPS) — those are recorded
constants in the chart, not re-timed here; Tier 2 re-measures them.

| Paper | Panel | chart_id | Source |
|-------|-------|----------|--------|
| **Fig 1** | Left — storage reduction bars | `codec_storage_reduction` | results.db + CSV |
| **Fig 1** | Right — comp/decomp throughput | `codec_throughput` | measured constants |
| **Fig 2** | Model-hub storage & count growth | `modelhub_growth` | model_hub_crawl JSON |
| **Fig 4** | Metadata quality by download rank | `modelhub_metadata` | model_hub_crawl CSV |
| **Fig 6** | Per-tensor reduction heatmap | `compression_matrix_heatmap` | ⚠ research monorepo only |
| **Fig 11a** | Cumulative reduction over trace | `reduction_global` (`reduction_trace`) | results.db + metadata |
| **Fig 11b** | Per-tensor reduction CDF | `zipllm_real_cdf` (`ratio_cdf`) | results.db |
| **Fig 11c** | Reduction violin by family | `reduction_violin_by_family` | results.db + metadata |
| **Fig 11** | All three combined | `reduction_combined` | results.db + metadata |
| **Fig 12a** | TensorSketch Recall@1 | `bcs_recall` | measured constants |
| **Fig 12b** | End-to-end QPS | `bcs_qps` | measured constants |
| **Fig 12c** | Per-query IO | `bcs_memory` | measured constants |
| **Fig 12** | All panels | `bcs_overview` | measured constants |
| **Fig 13** | Predicted vs real reduction ratio | `pred_vs_real_ratio` | **re-runnable** — `make verify-predict` re-fits TensorPred (OLS) from cache; held-out MAE 1.11% |
| **Fig 14** | FlexSplit vs ILP/Primal-Dual scaling | `algo_bench_q_proj`, `algo_bench_v_proj` | **re-runnable** — `make bench-fig14` runs the real solvers (ILP=Gurobi) |
| **Fig 15** | Cluster size / reduction, will-split | `flexsplit_cluster_size`, `flexsplit_split_overview` | flexsplit_all_results.json |
| **Fig 16** | Phase-II split effect / net gain | `flexsplit_split_effect`, `flexsplit_pred_vs_real` | flexsplit json + real_compression CSV |
| **Table 3** | Ingest/retrieval throughput | `throughput_comparison`, `throughput_and_bar` | charts replot recorded values; **re-runnable** — `make bench-table3` re-measures codec throughput with the shipped kernels |

**Supplementary** (analysis, not numbered figures): `entropy_cdf`, `byteplane_cdf`,
`theo_vs_actual_cdf`, `ratio_cdf`, `layer_entropy`, `layer_ratio`, `pipeline_bar`,
`lowhi_scatter`, `bcs_reduction`, `reduction_by_family`, `zipllm_real_bar`,
`flexsplit_*` (cluster/star/post-split), `rr_qps_comparison`.

## Notes

- **38 of 41 charts render from the shipped cache.** The 3 that don't
  (`compression_matrix_heatmap`, `compression_matrix_bar`, `source_diversity`,
  i.e. Fig 6) pull raw fingerprints through the research monorepo
  (`tensordb.core.engine`, `tests/`, `algorithms/`) and are out of scope for the
  slim AE cache — reproduce them from the full repo if needed.
- **Reproducible experiments** (not just plotted constants): **Fig 12a Recall@1**
  (`make verify-recall`) re-runs greedy base-selection (brute-force vs HNSW) →
  Recall@1 = 1.00; **Fig 13** (`make verify-predict`) re-fits the TensorPred
  reduction-ratio predictor from the cache by OLS and evaluates it on a held-out
  split → MAE 1.11 %, Pearson 99.3 % (the recovered model matches the stored
  `pred_ratio` column to 1e-4); **Fig 14** (`make bench-fig14`) actually runs the
  ILP (Gurobi), Primal-Dual, and FlexSplit solvers across model counts and
  re-plots the scaling curves. The remaining Fig 12 panels (QPS, IO) and
  throughput (Fig 1-right, Table 3) are hardware measurements — plotted from
  recorded values; `run_full.py` re-times codec throughput on the reviewer's box.
- FM++ (`fratio`, the 70.5 % result) and TensorX (`tratio`, 65.1 %) are
  both re-derived bit-exact by Tier 1. TensorX is on by default; FM++ needs the
  optional codec (`make ae-fmpp`, vendored FM-Delta lib in `third_party/fmdelta/`).
  Without it, the FM++ *values* still render from the cache in Tier 0.
