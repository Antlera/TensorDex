//! Compression Execution Pipeline
//!
//! This module handles the execution of compression plans.
//! It is generic over the compression algorithm (Bitx or TensorX).

use pyo3::prelude::*;
use rayon::prelude::*;

use crate::compression::plan::{
    per_pair_records_to_json, CompressionPlan, LegacyPairReport, PairReport,
};
use crate::kernels::bitx::compress_bitx;
use crate::kernels::tensorx::compress_tensorx_parallel;
use crate::resolvers::{S3TensorResolver, SimpleTensorResolver, TensorFileResolver};

// =============================================================================
// Execution / Pipeline Logic
// =============================================================================

const DEFAULT_PAIRWISE_BATCH_SIZE: usize = 512;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Algorithm {
    /// Bitx: bit-plane delta + Zstd
    Bitx,
    /// TensorX: XOR delta + byte-plane split + Zstd
    TensorX,
}

impl Algorithm {
    fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "bitx" => Some(Algorithm::Bitx),
            "tensorx" => Some(Algorithm::TensorX),
            _ => None,
        }
    }
}

#[derive(Clone)]
struct PairBatchResult {
    bytes_in: u64,
    bytes_out: u64,
}

struct BatchMetrics {
    load_ms: u128,
    compress_ms: u128,
    artifacts_ms: u128,
    pair_count: usize,
    estimated_bytes: usize,
}

struct BatchExecutionOutcome {
    results: Vec<(usize, Result<PairBatchResult, String>)>,
    metrics: BatchMetrics,
}

struct BatchPlan<'a> {
    index: usize,
    plan: &'a CompressionPlan,
}

#[derive(Default)]
struct PairStageMetrics {
    load_ms: u128,
    compress_ms: u128,
    artifacts_ms: u128,
}

#[derive(Clone)]
enum PairOutcome {
    Success { bytes_in: u64, bytes_out: u64 },
    Failure { reason: String },
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct CompressReport {
    pub total_plans: usize,
    pub executed_plans: usize,
    pub failed_plans: usize,
    pub total_original_bytes: u64,
    pub total_compressed_bytes: u64,
    pub compression_ratio: f64,
    pub execution_time_ms: u128,
    pub pairs: Vec<LegacyPairReport>,
    pub per_pair: Vec<PairReport>,
    pub per_pair_json: Option<String>,
}

/// All tunables for a single `compress_batch` call.
///
/// `batch_size` and `max_batch_bytes` are advanced knobs — leave them at the
/// defaults unless profiling shows you need different values.
#[derive(Debug, Clone)]
pub struct CompressConfig {
    pub algorithm: Algorithm,
    pub level: i32,
    pub output_dir: Option<String>,
    pub verbose: bool,
    pub batch_size: usize,
    pub max_batch_bytes: Option<usize>,
}

impl CompressConfig {
    pub fn new(algorithm: Algorithm) -> Self {
        Self {
            algorithm,
            level: 3,
            output_dir: None,
            verbose: false,
            batch_size: DEFAULT_PAIRWISE_BATCH_SIZE,
            max_batch_bytes: None,
        }
    }

    pub fn write_artifacts(&self) -> bool {
        self.output_dir.is_some()
    }
}

/// Compress a batch of target→base pairs under `config`.
pub fn compress_batch(
    plans: Vec<CompressionPlan>,
    tensor_dir: &str,
    config: &CompressConfig,
) -> Result<CompressReport, String> {
    let start_time = std::time::Instant::now();
    let CompressConfig {
        algorithm,
        level,
        output_dir,
        verbose,
        batch_size,
        max_batch_bytes,
    } = config.clone();
    let write_artifacts = output_dir.is_some();

    if verbose {
        println!(
            "compress_batch: {} plans algo={:?} batch_size={} max_batch_bytes={:?} write_artifacts={}",
            plans.len(), algorithm, batch_size, max_batch_bytes, write_artifacts,
        );
    }

    // Create tensor resolver
    let resolver = build_tensor_resolver(tensor_dir)?;

    // Create output directory if write_artifacts is enabled
    if write_artifacts {
        let out_dir = output_dir
            .as_ref()
            .ok_or_else(|| "output_dir is required when write_artifacts=true".to_string())?;
        std::fs::create_dir_all(out_dir)
            .map_err(|e| format!("Failed to create output directory: {}", e))?;
    }

    let output_dir_ref = output_dir.as_deref();

    let plan_estimates: Vec<usize> = plans
        .iter()
        .map(estimate_plan_num_bytes)
        .collect::<Result<_, _>>()?;

    let mut outcomes: Vec<Option<PairOutcome>> = vec![None; plans.len()];
    let mut plan_cursor = 0usize;
    let mut batch_id = 0usize;

    while plan_cursor < plans.len() {
        let mut batch_indices = Vec::new();
        let mut batch_bytes = 0usize;

        while plan_cursor < plans.len() && batch_indices.len() < batch_size {
            let idx = plan_cursor;
            let estimate = plan_estimates[idx];
            let exceeds_max = max_batch_bytes
                .map(|max_bytes| estimate > max_bytes)
                .unwrap_or(false);
            let would_overflow = max_batch_bytes
                .map(|max_bytes| !exceeds_max && batch_bytes + estimate > max_bytes)
                .unwrap_or(false);

            if !batch_indices.is_empty() && (exceeds_max || would_overflow) {
                break;
            }

            batch_indices.push(idx);
            batch_bytes = batch_bytes.saturating_add(estimate);
            plan_cursor += 1;

            if exceeds_max {
                break;
            }
            if let Some(max_bytes) = max_batch_bytes {
                if batch_bytes >= max_bytes {
                    break;
                }
            }
        }

        if batch_indices.is_empty() {
            let idx = plan_cursor;
            batch_indices.push(idx);
            batch_bytes = plan_estimates[idx];
            plan_cursor += 1;
        }

        let mut batch_plans = Vec::with_capacity(batch_indices.len());
        for &idx in &batch_indices {
            batch_plans.push(BatchPlan {
                index: idx,
                plan: &plans[idx],
            });
        }

        let outcome = run_pairwise_batch(
            batch_id,
            batch_bytes,
            &batch_plans,
            resolver.as_ref(),
            algorithm,
            level,
            write_artifacts,
            output_dir_ref,
            verbose,
        );

        if verbose {
            println!(
                "Batch {:04}: pairs={} est_bytes={} load={}ms compress={}ms artifacts={}ms",
                batch_id,
                outcome.metrics.pair_count,
                outcome.metrics.estimated_bytes,
                outcome.metrics.load_ms,
                outcome.metrics.compress_ms,
                outcome.metrics.artifacts_ms
            );
        }

        for (plan_idx, result) in outcome.results {
            let slot = &mut outcomes[plan_idx];
            *slot = Some(match result {
                Ok(pair) => PairOutcome::Success {
                    bytes_in: pair.bytes_in,
                    bytes_out: pair.bytes_out,
                },
                Err(reason) => PairOutcome::Failure { reason },
            });
        }

        batch_id += 1;
    }

    let mut total_compressed_bytes = 0u64;
    let mut total_original_bytes = 0u64;
    let mut pair_reports = Vec::with_capacity(plans.len());
    let mut per_pair_reports = Vec::with_capacity(plans.len());
    let mut executed_plans = 0usize;
    let mut failed_plans = 0usize;

    for (idx, plan) in plans.iter().enumerate() {
        match outcomes.get_mut(idx).and_then(|slot| slot.take()) {
            Some(PairOutcome::Success {
                bytes_in,
                bytes_out,
            }) => {
                executed_plans += 1;
                total_original_bytes += bytes_in;
                total_compressed_bytes += bytes_out;
                pair_reports.push(LegacyPairReport {
                    target_id: plan.target_tensor_id.clone(),
                    base_id: plan.base_tensor_id.clone(),
                    bytes_in,
                    bytes_out,
                });
                per_pair_reports.push(PairReport {
                    target_tensor_id: plan.target_tensor_id.clone(),
                    base_tensor_id: plan.base_tensor_id.clone(),
                    bytes_out,
                });
            }
            Some(PairOutcome::Failure { reason }) => {
                failed_plans += 1;
                if verbose {
                    eprintln!("Failed pair {}: {}", plan.param_name, reason);
                }
            }
            None => {
                failed_plans += 1;
            }
        }
    }

    let compression_ratio = if total_original_bytes > 0 {
        total_compressed_bytes as f64 / total_original_bytes as f64
    } else {
        0.0
    };

    let execution_time_ms = start_time.elapsed().as_millis();
    let per_pair_json = per_pair_records_to_json(&per_pair_reports);

    Ok(CompressReport {
        total_plans: plans.len(),
        executed_plans,
        failed_plans,
        total_original_bytes,
        total_compressed_bytes,
        compression_ratio,
        execution_time_ms,
        pairs: pair_reports,
        per_pair: per_pair_reports,
        per_pair_json,
    })
}

fn run_pairwise_batch(
    batch_id: usize,
    estimated_bytes: usize,
    batch_plans: &[BatchPlan],
    resolver: &dyn TensorFileResolver,
    algorithm: Algorithm,
    compression_level: i32,
    write_artifacts: bool,
    output_dir: Option<&str>,
    verbose: bool,
) -> BatchExecutionOutcome {
    let _ = verbose;
    let pipeline_results: Vec<_> = batch_plans
        .par_iter()
        .map(|batch_plan| {
            execute_plan_pipeline(
                batch_plan,
                resolver,
                algorithm,
                compression_level,
                write_artifacts,
                output_dir,
                batch_id,
            )
        })
        .collect();

    let mut results = Vec::with_capacity(batch_plans.len());
    let mut metrics = BatchMetrics {
        load_ms: 0,
        compress_ms: 0,
        artifacts_ms: 0,
        pair_count: batch_plans.len(),
        estimated_bytes,
    };

    for (plan_idx, outcome, p_metrics) in pipeline_results {
        results.push((plan_idx, outcome));
        metrics.load_ms += p_metrics.load_ms;
        metrics.compress_ms += p_metrics.compress_ms;
        metrics.artifacts_ms += p_metrics.artifacts_ms;
    }

    BatchExecutionOutcome { results, metrics }
}

fn execute_plan_pipeline(
    batch_plan: &BatchPlan,
    resolver: &dyn TensorFileResolver,
    algorithm: Algorithm,
    level: i32,
    write_artifacts: bool,
    output_dir: Option<&str>,
    _batch_id: usize,
) -> (usize, Result<PairBatchResult, String>, PairStageMetrics) {
    let plan = batch_plan.plan;
    let mut metrics = PairStageMetrics::default();

    let plan_result = (|| -> Result<PairBatchResult, String> {
        let load_timer = std::time::Instant::now();
        // 1. Load Data — both target and base use zero-copy mmap
        //    This avoids the ~170 MB/s per-thread memcpy bottleneck that
        //    dominated load time when copying target into a buffer.
        let element_size = plan.element_size()?;

        let target_batch = resolver
            .bulk_load_tensors_mmap(&vec![(
                plan.target_tensor_id.clone(),
                plan.target_shape.clone(),
            )])
            .map_err(|e| format!("Load target failed: {}", e))?;

        let target_view = target_batch
            .into_iter()
            .next()
            .ok_or_else(|| "Target tensor missing".to_string())?;
        let target_slice = target_view.data();

        let base_batch = resolver
            .bulk_load_tensors_mmap(&vec![(
                plan.base_tensor_id.clone(),
                plan.target_shape.clone(),
            )])
            .map_err(|e| format!("Load base failed: {}", e))?;

        let base_view = base_batch
            .into_iter()
            .next()
            .ok_or_else(|| "Base tensor missing".to_string())?;
        let base_slice = base_view.data();

        if target_slice.len() != base_slice.len() {
            return Err(format!(
                "Size mismatch: target {} bytes vs base {} bytes",
                target_slice.len(),
                base_slice.len()
            ));
        }

        metrics.load_ms = load_timer.elapsed().as_millis() as u128;

        // 2. Compress (Dispatch to Algorithm)
        //    All compress functions read target & base as &[u8] (read-only)
        //    and allocate their own output buffer internally.
        let compress_timer = std::time::Instant::now();

        let compressed = match algorithm {
            Algorithm::Bitx => compress_bitx(target_slice, Some(base_slice), element_size, level)?,
            Algorithm::TensorX => {
                compress_tensorx_parallel(target_slice, base_slice, element_size, 1)?
            }
        };

        metrics.compress_ms = compress_timer.elapsed().as_millis() as u128;

        // 3. Artifacts
        if write_artifacts {
            let dir = output_dir.ok_or("output_dir required")?;
            let path = format!("{}/{}.bin", dir, sanitize_filename(&plan.param_name));
            let art_timer = std::time::Instant::now();
            std::fs::write(&path, &compressed)
                .map_err(|e| format!("Write artifact failed: {}", e))?;
            metrics.artifacts_ms = art_timer.elapsed().as_millis() as u128;
        }

        Ok(PairBatchResult {
            bytes_in: target_slice.len() as u64,
            bytes_out: compressed.len() as u64,
        })
    })();

    (batch_plan.index, plan_result, metrics)
}

fn estimate_plan_num_bytes(plan: &CompressionPlan) -> Result<usize, String> {
    let elem_size = plan.element_size()?;
    let element_count = plan.target_shape.iter().try_fold(1usize, |acc, &dim| {
        acc.checked_mul(dim).ok_or_else(|| "Overflow".to_string())
    })?;
    element_count
        .checked_mul(elem_size)
        .ok_or_else(|| "Overflow".to_string())
}

fn build_tensor_resolver(tensor_dir: &str) -> Result<Box<dyn TensorFileResolver>, String> {
    let trimmed = tensor_dir.trim();
    if let Some(stripped) = trimmed.strip_prefix("s3://") {
        let (path_part, query_part) = match stripped.split_once('?') {
            Some((p, q)) => (p, Some(q)),
            None => (stripped, None),
        };
        let mut parts = path_part.splitn(2, '/');
        let bucket = parts
            .next()
            .filter(|s| !s.is_empty())
            .ok_or("Bucket missing")?;
        let prefix = parts.next().unwrap_or("").trim_matches('/');
        let region = query_part.and_then(|q| {
            q.split('&').find_map(|pair| {
                let mut kv = pair.splitn(2, '=');
                if kv.next()?.eq_ignore_ascii_case("region") {
                    kv.next().map(String::from)
                } else {
                    None
                }
            })
        });
        Ok(Box::new(S3TensorResolver::new(
            bucket.into(),
            prefix.into(),
            region,
        )?))
    } else {
        Ok(Box::new(SimpleTensorResolver::new(tensor_dir.into())))
    }
}

fn sanitize_filename(name: &str) -> String {
    name.chars()
        .map(|c| {
            if c.is_alphanumeric() || "._-".contains(c) {
                c
            } else {
                '_'
            }
        })
        .collect()
}

// =============================================================================
// Python Bindings
// =============================================================================

/// Python entry: run `compress_batch` with keyword-only tuning args.
///
/// Args:
///     plans (str): JSON-encoded list of CompressionPlan records.
///     tensor_dir (str): Local path or `s3://bucket/prefix?region=...` URI.
///     output_dir (str, optional): Where to write compressed artifacts;
///         ``None`` disables artifact output (metrics-only run).
///     algorithm (str, optional): Codec name (``"bitx"`` or ``"tensorx"``).
///         Defaults to ``"bitx"``.
///     level (int): zstd compression level 1-22. Default 3.
///     verbose (bool): Emit Rust-side progress logs. Default false.
#[pyfunction]
#[pyo3(
    name = "compress",
    signature = (
        plans,
        tensor_dir,
        *,
        output_dir = None,
        algorithm = None,
        level = 3,
        verbose = false,
    )
)]
pub fn compress_py(
    py: Python,
    plans: &str,
    tensor_dir: &str,
    output_dir: Option<String>,
    algorithm: Option<&str>,
    level: i32,
    verbose: bool,
) -> PyResult<PyObject> {
    let plans: Vec<CompressionPlan> = serde_json::from_str(plans)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;

    let algo_str = algorithm.unwrap_or("bitx");
    let algo = Algorithm::from_str(algo_str).ok_or_else(|| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Unknown algorithm: {}", algo_str))
    })?;

    let config = CompressConfig {
        algorithm: algo,
        level,
        output_dir,
        verbose,
        batch_size: DEFAULT_PAIRWISE_BATCH_SIZE,
        max_batch_bytes: None,
    };
    let tensor_dir_owned = tensor_dir.to_string();

    let report = py
        .allow_threads(|| compress_batch(plans, &tensor_dir_owned, &config))
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))?;

    let result_dict = pyo3::types::PyDict::new(py);
    result_dict.set_item("total_plans", report.total_plans)?;
    result_dict.set_item("executed_plans", report.executed_plans)?;
    result_dict.set_item("failed_plans", report.failed_plans)?;
    result_dict.set_item("total_original_bytes", report.total_original_bytes)?;
    result_dict.set_item("total_compressed_bytes", report.total_compressed_bytes)?;
    result_dict.set_item("compression_ratio", report.compression_ratio)?;
    result_dict.set_item("execution_time_ms", report.execution_time_ms)?;

    use pyo3::types::PyList;
    let pairs_list = PyList::empty(py);
    for pair in &report.pairs {
        let pair_dict = pyo3::types::PyDict::new(py);
        pair_dict.set_item("target_id", &pair.target_id)?;
        pair_dict.set_item("base_id", &pair.base_id)?;
        pair_dict.set_item("bytes_in", pair.bytes_in)?;
        pair_dict.set_item("bytes_out", pair.bytes_out)?;
        pairs_list.append(pair_dict)?;
    }
    result_dict.set_item("pairs", pairs_list)?;

    let per_pair_list = PyList::empty(py);
    for pair in &report.per_pair {
        let pair_dict = pyo3::types::PyDict::new(py);
        pair_dict.set_item("target_tensor_id", &pair.target_tensor_id)?;
        pair_dict.set_item("base_tensor_id", &pair.base_tensor_id)?;
        pair_dict.set_item("bytes_out", pair.bytes_out)?;
        per_pair_list.append(pair_dict)?;
    }
    result_dict.set_item("per_pair", per_pair_list)?;

    match &report.per_pair_json {
        Some(json) => result_dict.set_item("per_pair_json", json)?,
        None => result_dict.set_item("per_pair_json", py.None())?,
    }

    Ok(result_dict.to_object(py))
}
