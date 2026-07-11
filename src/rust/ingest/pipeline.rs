//! Atomic ingest pipeline.
//!
//! Six phases — enumerate, hash, dedup, fingerprint, save blobs, commit —
//! run under a single Python call. SQLite writes happen in one rusqlite
//! transaction on the `MetadataStore` handle; blobs hit disk before the SQL
//! commit, so a crash mid-ingest can leave orphan blob files but never half
//! a tensor row.

use std::collections::{HashMap, HashSet};
use std::path::Path;

use chrono::Utc;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::kernels::sketch::{BitCountSketch, BCS_DEFAULT_SEEDS};
use crate::metadata::fingerprint::FingerprintStore;
use crate::metadata::store::{MappingInsertRow, MetadataStore, TensorInsertRow};

use super::blob_writer::{blob_relpath, prepare_dirs, write_blob};
use super::hash::content_hash_hex;
use super::safetensors_reader::{dtype_to_torch_str, shape_to_json, OpenSafetensors, TensorSlot};

fn rt_err<E: std::fmt::Display>(e: E) -> PyErr {
    PyRuntimeError::new_err(e.to_string())
}

/// Encode an i32 fingerprint vector to the little-endian byte form stored
/// in SQLite's `tensors.fingerprint` column.
fn fp_to_blob(vec: &[i32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(vec.len() * 4);
    for &v in vec {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

/// Ingest one or more `.safetensors` files into the given stores.
///
/// ``param_filter`` — if present, only tensor names in this set are ingested.
/// ``blobs_root`` — local-fs root where blobs are written, mirroring
/// `LocalStorageBackend.blob_path_for_id` (2-level sharded).
///
/// Returns `{param_name: tensor_id}` for every ingested (or already-present)
/// tensor — same shape as the old `batch_add_tensors` collector.
#[pyfunction]
#[pyo3(signature = (
    metadata_store, fp_store, blobs_root, files, model_name,
    param_filter = None, bcs_d = 2, bcs_w = 1024,
))]
#[allow(clippy::too_many_arguments)]
pub fn ingest_from_safetensors_files(
    _py: Python<'_>,
    metadata_store: &PyCell<MetadataStore>,
    fp_store: &PyCell<FingerprintStore>,
    blobs_root: &str,
    files: Vec<String>,
    model_name: &str,
    param_filter: Option<HashSet<String>>,
    bcs_d: usize,
    bcs_w: usize,
) -> PyResult<HashMap<String, String>> {
    let root = Path::new(blobs_root);

    // ── Phase 0: open + enumerate ───────────────────────────────────
    let mmaps: Vec<OpenSafetensors> = files
        .iter()
        .map(|p| OpenSafetensors::open(Path::new(p)))
        .collect::<anyhow::Result<_>>()
        .map_err(rt_err)?;

    let enumerated: Vec<(usize, usize)> = {
        let mut out = Vec::new();
        for (fi, m) in mmaps.iter().enumerate() {
            for (si, s) in m.slots.iter().enumerate() {
                let keep = param_filter.as_ref().map_or(true, |f| f.contains(&s.name));
                if keep {
                    out.push((fi, si));
                }
            }
        }
        out
    };

    if enumerated.is_empty() {
        return Ok(HashMap::new());
    }

    // ── Phase 1: parallel hash ──────────────────────────────────────
    let hashed: Vec<(usize, usize, String)> = enumerated
        .par_iter()
        .map(|pair| {
            let fi = pair.0;
            let si = pair.1;
            let m = &mmaps[fi];
            let slot: &TensorSlot = &m.slots[si];
            let tid = content_hash_hex(m.slot_bytes(slot));
            (fi, si, tid)
        })
        .collect();

    // ── Phase 2: dedup against MetadataStore ────────────────────────
    let unique_ids: Vec<String> = {
        let mut set: HashSet<String> = HashSet::new();
        for row in &hashed {
            set.insert(row.2.clone());
        }
        set.into_iter().collect()
    };
    let existing_pairs: Vec<(String, String)> =
        metadata_store.borrow().existing_tensor_ids(unique_ids)?;
    let existing_ids: HashSet<String> = existing_pairs.iter().map(|p| p.0.clone()).collect();

    // First occurrence of each new tid (drops within-batch duplicates too).
    let new_items: Vec<(usize, usize, String)> = {
        let mut seen: HashSet<String> = HashSet::new();
        let mut out: Vec<(usize, usize, String)> = Vec::new();
        for row in &hashed {
            let tid = &row.2;
            if !existing_ids.contains(tid) && seen.insert(tid.clone()) {
                out.push((row.0, row.1, tid.clone()));
            }
        }
        out
    };

    // ── Phase 3: parallel BCS ───────────────────────────────────────
    let fingerprints: Vec<(String, Vec<i32>)> = new_items
        .par_iter()
        .map(|row| {
            let fi = row.0;
            let si = row.1;
            let tid = &row.2;
            let m = &mmaps[fi];
            let slot: &TensorSlot = &m.slots[si];
            let sk =
                BitCountSketch::from_bytes(m.slot_bytes(slot), bcs_d, bcs_w, &BCS_DEFAULT_SEEDS);
            let v: Vec<i32> = sk.table.iter().map(|&x| x.round() as i32).collect();
            (tid.clone(), v)
        })
        .collect();

    // ── Phase 4: parallel blob write ────────────────────────────────
    let new_tid_refs: Vec<&str> = new_items.iter().map(|r| r.2.as_str()).collect();
    prepare_dirs(root, &new_tid_refs).map_err(rt_err)?;

    let storage_uris: HashMap<String, String> = new_items
        .par_iter()
        .map(|row| -> PyResult<(String, String)> {
            let fi = row.0;
            let si = row.1;
            let tid = &row.2;
            let m = &mmaps[fi];
            let slot: &TensorSlot = &m.slots[si];
            let dtype_str = dtype_to_torch_str(slot.dtype);
            let uri = write_blob(
                root,
                tid,
                &slot.shape,
                slot.dtype,
                dtype_str,
                m.slot_bytes(slot),
            )
            .map_err(rt_err)?;
            Ok((tid.clone(), uri))
        })
        .collect::<PyResult<HashMap<_, _>>>()?;

    // ── Phase 5: single-transaction SQL commit ──────────────────────
    let now = Utc::now().to_rfc3339();
    let fp_blob_for: HashMap<String, Vec<u8>> = fingerprints
        .iter()
        .map(|pair| (pair.0.clone(), fp_to_blob(&pair.1)))
        .collect();

    let tensor_rows: Vec<TensorInsertRow> = new_items
        .iter()
        .map(|row| {
            let fi = row.0;
            let si = row.1;
            let tid = &row.2;
            let m = &mmaps[fi];
            let slot = &m.slots[si];
            let dtype_str = dtype_to_torch_str(slot.dtype).to_string();
            let size_bytes = slot.data_len as i64;
            let uri = storage_uris
                .get(tid)
                .cloned()
                .unwrap_or_else(|| blob_relpath(tid).to_string_lossy().to_string());
            let fp = fp_blob_for.get(tid).cloned();
            (
                tid.clone(),
                shape_to_json(&slot.shape),
                dtype_str,
                size_bytes,
                uri,
                fp,
                now.clone(),
            )
        })
        .collect();

    let mapping_rows: Vec<MappingInsertRow> = hashed
        .iter()
        .map(|row| {
            let fi = row.0;
            let si = row.1;
            let tid = &row.2;
            let slot = &mmaps[fi].slots[si];
            (
                model_name.to_string(),
                slot.name.clone(),
                tid.clone(),
                now.clone(),
            )
        })
        .collect();

    metadata_store
        .borrow()
        .ingest_batch(tensor_rows, mapping_rows)?;

    // ── Phase 6: absorb new fingerprints into FingerprintStore ──────
    {
        let mut fp_mut = fp_store.borrow_mut();
        for pair in fingerprints {
            fp_mut.absorb_vec(pair.0, &pair.1);
        }
    }

    // Collector: every param_name we saw → its tensor_id.
    let mut collector: HashMap<String, String> = HashMap::with_capacity(hashed.len());
    for row in &hashed {
        let slot = &mmaps[row.0].slots[row.1];
        collector.insert(slot.name.clone(), row.2.clone());
    }
    Ok(collector)
}
