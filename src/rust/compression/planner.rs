//! FlexSplit attach-stage planner.
//!
//! Given a `FingerprintStore` and an ordered list of tensors (each with
//! its shape key + n_bits), group by shape and process in arrival order:
//! the first fingerprint per shape becomes a base; every subsequent
//! tensor computes BCS Hamming distance to all existing bases in that
//! shape, predicts compression ratio via the Hybrid model, and either
//! attaches to the nearest base (``pred_cr <= cr_threshold``) or opens
//! a new base.
//!
//! This mirrors `build_clusters_incremental` in `FlexSplit.py` — the
//! algorithm stays the same, we just move the hot loop to Rust so the
//! fingerprint data never has to cross FFI.

use std::collections::HashMap;

use pyo3::prelude::*;

use crate::metadata::fingerprint::FingerprintStore;

/// Default Hybrid CR coefficients — v3, fitted on 710K tratio (TensorX) pairs.
/// See `FlexSplit.py::HYBRID_COEFFS`.
pub const DEFAULT_HYBRID_COEFFS: (f64, f64, f64, f64) = (-23.727944, 0.522466, 1.966862, -0.043132);

/// Default cr threshold for attach (matches `STANDALONE_ZSTD_CR`).
pub const DEFAULT_CR_THRESHOLD: f64 = 0.70;

/// BCS row count — d in the (d, w) factorisation. Distance averages
/// squared differences across rows, so this divisor matters for the
/// normalized Hamming estimate.
pub const DEFAULT_BCS_D: usize = 2;

// ---------------------------------------------------------------------------
// Prediction primitives
// ---------------------------------------------------------------------------

/// Binary entropy H(p) = -p·log2(p) - (1-p)·log2(1-p), clipped near the ends.
#[inline]
fn binary_entropy(p: f64) -> f64 {
    if p <= 1e-15 || p >= 1.0 - 1e-15 {
        0.0
    } else {
        -p * p.log2() - (1.0 - p) * (1.0 - p).log2()
    }
}

/// Hybrid model: `CR = a·p + b·t + c·p·t + d`, with `t = 8·H(p)` and
/// `p` clamped to `[0, 0.5]`. Output clipped to `[0, 1]`.
#[inline]
fn predict_cr(p: f64, coeffs: (f64, f64, f64, f64)) -> f64 {
    let p = p.clamp(0.0, 0.5);
    let t = 8.0 * binary_entropy(p);
    let (a, b, c, d) = coeffs;
    let cr = a * p + b * t + c * p * t + d;
    cr.clamp(0.0, 1.0)
}

/// BCS normalized Hamming distance between two fingerprint rows.
///
/// Matches `_bcs_dist_one_vs_many_jit`: sum of squared per-slot diffs
/// divided by `d` (the BCS row count) and by `n_bits`.
#[inline]
fn bcs_distance(a: &[i32], b: &[i32], n_bits: f64, d_rows: f64) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    let mut sum: i64 = 0;
    for (&x, &y) in a.iter().zip(b.iter()) {
        let diff = i64::from(x) - i64::from(y);
        sum += diff * diff;
    }
    (sum as f64) / d_rows / n_bits.max(1.0)
}

// ---------------------------------------------------------------------------
// Result types exposed to Python
// ---------------------------------------------------------------------------

/// One (target → base) pair decided by the planner.
#[derive(Clone)]
#[pyclass(module = "tensordex._ops")]
pub struct AttachPair {
    #[pyo3(get)]
    pub target_id: String,
    #[pyo3(get)]
    pub base_id: String,
    #[pyo3(get)]
    pub distance: f64,
    #[pyo3(get)]
    pub predicted_cr: f64,
}

#[pymethods]
impl AttachPair {
    fn __repr__(&self) -> String {
        format!(
            "AttachPair(target={:?}, base={:?}, dist={:.4}, pred_cr={:.4})",
            self.target_id, self.base_id, self.distance, self.predicted_cr
        )
    }
}

/// Full plan output.
#[derive(Clone)]
#[pyclass(module = "tensordex._ops")]
pub struct AttachPlan {
    #[pyo3(get)]
    pub bases: Vec<String>,
    #[pyo3(get)]
    pub pairs: Vec<AttachPair>,
    #[pyo3(get)]
    pub skipped_no_fp: Vec<String>,
    #[pyo3(get)]
    pub n_shapes: usize,
}

#[pymethods]
impl AttachPlan {
    fn __repr__(&self) -> String {
        format!(
            "AttachPlan(bases={}, pairs={}, skipped={}, shapes={})",
            self.bases.len(),
            self.pairs.len(),
            self.skipped_no_fp.len(),
            self.n_shapes
        )
    }
}

// ---------------------------------------------------------------------------
// Python entry
// ---------------------------------------------------------------------------

/// Build an attach plan for a list of tensors.
///
/// ``tensors`` is an **ordered** list of `(tensor_id, shape_key, n_bits)`:
///   - `shape_key` is any string — it just defines shape-group identity
///     (Python typically stringifies the shape tuple).
///   - `n_bits` is total bits in the tensor (numel * dtype_bits), used
///     to normalise the BCS distance.
///
/// Tensors that lack a fingerprint in ``fingerprints`` are skipped (the
/// first one per shape can still become a base, matching FlexSplit's
/// ``n_no_fp`` handling, but won't attract members).
#[pyfunction]
#[pyo3(
    name = "plan_attach",
    signature = (
        fingerprints,
        tensors,
        cr_threshold=DEFAULT_CR_THRESHOLD,
        coeffs=DEFAULT_HYBRID_COEFFS,
        d_rows=DEFAULT_BCS_D,
    )
)]
pub fn plan_attach_py(
    fingerprints: PyRef<FingerprintStore>,
    tensors: Vec<(String, String, i64)>,
    cr_threshold: f64,
    coeffs: (f64, f64, f64, f64),
    d_rows: usize,
) -> PyResult<AttachPlan> {
    let fp = &*fingerprints;
    let _ = fp.dim(); // sanity — also forces a reference read so clippy stops warning

    // Preserve arrival order within each shape group.
    let mut shape_order: Vec<String> = Vec::new();
    let mut shape_groups: HashMap<String, Vec<(String, f64)>> = HashMap::new();
    for (tid, shape_key, n_bits) in tensors {
        if !shape_groups.contains_key(&shape_key) {
            shape_order.push(shape_key.clone());
            shape_groups.insert(shape_key.clone(), Vec::new());
        }
        shape_groups
            .get_mut(&shape_key)
            .unwrap()
            .push((tid, n_bits as f64));
    }

    let d_rows_f = d_rows.max(1) as f64;

    let mut bases: Vec<String> = Vec::new();
    let mut pairs: Vec<AttachPair> = Vec::new();
    let mut skipped_no_fp: Vec<String> = Vec::new();

    for shape_key in &shape_order {
        let tids = match shape_groups.get(shape_key) {
            Some(v) => v,
            None => continue,
        };

        // Per-shape list of bases that actually have fingerprints. Stored
        // as parallel vectors of (id, fp_slice) — using the crate-visible
        // `row_by_id` so we never copy through Python.
        let mut fp_base_ids: Vec<&str> = Vec::new();
        let mut fp_base_rows: Vec<&[i32]> = Vec::new();

        for (tid, n_bits) in tids {
            let row_opt = fp.row_by_id(tid);

            // No bases yet → first tensor anchors the shape, regardless of fp.
            if fp_base_ids.is_empty() {
                bases.push(tid.clone());
                match row_opt {
                    Some(row) => {
                        fp_base_ids.push(tid.as_str());
                        fp_base_rows.push(row);
                    }
                    None => skipped_no_fp.push(tid.clone()),
                }
                continue;
            }

            // Tensor without a fingerprint → can't compare, opens a new base.
            let query = match row_opt {
                Some(r) => r,
                None => {
                    bases.push(tid.clone());
                    skipped_no_fp.push(tid.clone());
                    continue;
                }
            };

            // Existing bases all lack fingerprints → this one becomes a new fp'd base.
            if fp_base_rows.is_empty() {
                bases.push(tid.clone());
                fp_base_ids.push(tid.as_str());
                fp_base_rows.push(query);
                continue;
            }

            // Find nearest base via BCS distance.
            let mut best_idx: usize = 0;
            let mut best_dist: f64 = f64::INFINITY;
            for (idx, base_row) in fp_base_rows.iter().enumerate() {
                let dist = bcs_distance(query, base_row, *n_bits, d_rows_f);
                if dist < best_dist {
                    best_dist = dist;
                    best_idx = idx;
                }
            }

            let pred_cr = predict_cr(best_dist, coeffs);
            if pred_cr <= cr_threshold {
                // Attach to the nearest base.
                pairs.push(AttachPair {
                    target_id: tid.clone(),
                    base_id: fp_base_ids[best_idx].to_string(),
                    distance: best_dist,
                    predicted_cr: pred_cr,
                });
            } else {
                // Too far — open a new base in this shape group.
                bases.push(tid.clone());
                fp_base_ids.push(tid.as_str());
                fp_base_rows.push(query);
            }
        }
    }

    Ok(AttachPlan {
        bases,
        pairs,
        skipped_no_fp,
        n_shapes: shape_order.len(),
    })
}
