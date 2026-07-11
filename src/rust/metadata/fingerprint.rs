//! FingerprintStore — contiguous i32 arena for BCS fingerprints.
//!
//! Holds N × k int32 values in a single row-major `Vec<i32>`, keyed by
//! tensor_id.  Python receives numpy views and never touches the raw
//! bytes — the store owns decode (from SQLite blob), insert (from
//! freshly-computed vectors), single/batch gather, and full-matrix
//! materialisation.

use std::collections::HashMap;

use ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1};
use pyo3::exceptions::{PyKeyError, PyValueError};
use pyo3::prelude::*;

/// Decode a SQLite-persisted fingerprint blob (little-endian i32[k]).
pub(crate) fn decode_blob(blob: &[u8], k: usize) -> Option<Vec<i32>> {
    if blob.len() != k * 4 {
        return None;
    }
    let mut out = Vec::with_capacity(k);
    for chunk in blob.chunks_exact(4) {
        out.push(i32::from_le_bytes(chunk.try_into().unwrap()));
    }
    Some(out)
}

#[pyclass(module = "tensordex._ops")]
pub struct FingerprintStore {
    k: usize,
    ids: Vec<String>,
    index: HashMap<String, usize>,
    arena: Vec<i32>,
}

impl FingerprintStore {
    #[inline]
    fn row_slice(&self, row: usize) -> &[i32] {
        let start = row * self.k;
        &self.arena[start..start + self.k]
    }

    /// Row lookup by tensor id, crate-visible so the planner can read
    /// fingerprint vectors without crossing the Python boundary.
    #[inline]
    pub(crate) fn row_by_id(&self, id: &str) -> Option<&[i32]> {
        self.index.get(id).map(|&row| self.row_slice(row))
    }

    /// Fingerprint dimension (d * w). Crate-visible for the planner.
    #[inline]
    pub(crate) fn dim(&self) -> usize {
        self.k
    }

    fn upsert_row(&mut self, id: String, vec: &[i32]) -> PyResult<()> {
        if vec.len() != self.k {
            return Err(PyValueError::new_err(format!(
                "fingerprint length {} != k={}",
                vec.len(),
                self.k
            )));
        }
        self.upsert_row_checked(id, vec);
        Ok(())
    }

    /// Length-checked upsert. Caller guarantees `vec.len() == self.k`.
    fn upsert_row_checked(&mut self, id: String, vec: &[i32]) {
        if let Some(&row) = self.index.get(&id) {
            let start = row * self.k;
            self.arena[start..start + self.k].copy_from_slice(vec);
            return;
        }
        let row = self.ids.len();
        self.index.insert(id.clone(), row);
        self.ids.push(id);
        self.arena.extend_from_slice(vec);
    }

    /// Decode a SQLite-persisted blob and upsert it. Used by `MetadataStore`
    /// to ferry fingerprints directly from rusqlite into this arena without
    /// bouncing through Python. Returns `false` if the blob length is wrong.
    pub(crate) fn absorb_blob(&mut self, id: String, blob: &[u8]) -> bool {
        match decode_blob(blob, self.k) {
            Some(vec) => {
                self.upsert_row_checked(id, &vec);
                true
            }
            None => false,
        }
    }

    /// Upsert a freshly-computed fingerprint vector. Used by the Rust
    /// ingest pipeline; returns `false` if the length is wrong.
    pub(crate) fn absorb_vec(&mut self, id: String, vec: &[i32]) -> bool {
        if vec.len() != self.k {
            return false;
        }
        self.upsert_row_checked(id, vec);
        true
    }
}

#[pymethods]
impl FingerprintStore {
    /// Create an empty store for fingerprints of dimension ``k`` (= d * w).
    #[new]
    #[pyo3(signature = (k, capacity_hint=0))]
    fn new(k: usize, capacity_hint: usize) -> PyResult<Self> {
        if k == 0 {
            return Err(PyValueError::new_err("k must be > 0"));
        }
        Ok(Self {
            k,
            ids: Vec::with_capacity(capacity_hint),
            index: HashMap::with_capacity(capacity_hint),
            arena: Vec::with_capacity(capacity_hint.saturating_mul(k)),
        })
    }

    #[getter]
    fn k(&self) -> usize {
        self.k
    }

    fn __len__(&self) -> usize {
        self.ids.len()
    }

    fn __contains__(&self, tensor_id: &str) -> bool {
        self.index.contains_key(tensor_id)
    }

    /// Insert a freshly-computed fingerprint (numpy int32 array, length k).
    fn insert_vec(&mut self, tensor_id: String, vec: PyReadonlyArray1<i32>) -> PyResult<()> {
        let slice = vec.as_slice().map_err(|e| {
            PyValueError::new_err(format!("fingerprint array must be contiguous: {}", e))
        })?;
        self.upsert_row(tensor_id, slice)
    }

    /// Insert a fingerprint decoded from a SQLite blob (little-endian i32[k]).
    fn insert_blob(&mut self, tensor_id: String, blob: &[u8]) -> PyResult<()> {
        let decoded = decode_blob(blob, self.k).ok_or_else(|| {
            PyValueError::new_err(format!(
                "fingerprint blob must be {} bytes (k={} * 4), got {}",
                self.k * 4,
                self.k,
                blob.len()
            ))
        })?;
        self.upsert_row(tensor_id, &decoded)
    }

    /// Insert many (tensor_id, blob) pairs in one call — used by hydration.
    /// Invalid blobs are silently skipped and counted in the returned tuple.
    fn insert_blobs(&mut self, items: Vec<(String, Vec<u8>)>) -> (usize, usize) {
        let mut ok = 0usize;
        let mut skipped = 0usize;
        for (tid, blob) in items {
            match decode_blob(&blob, self.k) {
                Some(vec) => {
                    let _ = self.upsert_row(tid, &vec);
                    ok += 1;
                }
                None => skipped += 1,
            }
        }
        (ok, skipped)
    }

    /// Fetch one fingerprint as an i32 numpy array, or ``None`` if absent.
    fn get<'py>(&self, py: Python<'py>, tensor_id: &str) -> Option<&'py PyArray1<i32>> {
        let row = *self.index.get(tensor_id)?;
        Some(PyArray1::from_slice(py, self.row_slice(row)))
    }

    /// Gather a batch of fingerprints as an (N, k) int32 numpy matrix, ordered
    /// to match ``tensor_ids``. Raises ``KeyError`` on the first missing id.
    fn get_batch<'py>(
        &self,
        py: Python<'py>,
        tensor_ids: Vec<&str>,
    ) -> PyResult<&'py PyArray2<i32>> {
        let n = tensor_ids.len();
        let mut out = Vec::with_capacity(n * self.k);
        for tid in &tensor_ids {
            let row = self.index.get(*tid).ok_or_else(|| {
                PyKeyError::new_err(format!("fingerprint missing for tensor {}", tid))
            })?;
            out.extend_from_slice(self.row_slice(*row));
        }
        let arr = Array2::from_shape_vec((n, self.k), out)
            .map_err(|e| PyValueError::new_err(format!("shape error: {}", e)))?;
        Ok(arr.into_pyarray(py))
    }

    /// Return the full (N, k) int32 matrix. Copies the arena — caller owns it.
    fn matrix<'py>(&self, py: Python<'py>) -> PyResult<&'py PyArray2<i32>> {
        let n = self.ids.len();
        let arr = Array2::from_shape_vec((n, self.k), self.arena.clone())
            .map_err(|e| PyValueError::new_err(format!("shape error: {}", e)))?;
        Ok(arr.into_pyarray(py))
    }

    fn ids(&self) -> Vec<String> {
        self.ids.clone()
    }

    fn clear(&mut self) {
        self.ids.clear();
        self.index.clear();
        self.arena.clear();
    }
}
