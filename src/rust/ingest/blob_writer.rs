//! Write a tensor payload as a self-describing ``.safetensors`` blob in the
//! canonical 2-level sharded CAS layout: ``blobs/{id[:2]}/{id[2:4]}/{id}.safetensors``.
//!
//! Mirrors `LocalStorageBackend.blob_path_for_id` on the Python side —
//! changing the layout here requires updating both.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use safetensors::tensor::TensorView;
use safetensors::Dtype;
use serde::Serialize;

pub const ENGINE_VERSION: &str = "2.0";
pub const SAFETENSOR_KEY: &str = "tensor";

pub fn blob_relpath(tensor_id: &str) -> PathBuf {
    let p1 = if tensor_id.len() >= 2 {
        &tensor_id[..2]
    } else {
        "00"
    };
    let p2 = if tensor_id.len() >= 4 {
        &tensor_id[2..4]
    } else {
        "00"
    };
    PathBuf::from("blobs")
        .join(p1)
        .join(p2)
        .join(format!("{}.safetensors", tensor_id))
}

pub fn blob_abspath(root: &Path, tensor_id: &str) -> PathBuf {
    root.join(blob_relpath(tensor_id))
}

/// Pre-create every directory needed by the given ids (dedup'd).
pub fn prepare_dirs(root: &Path, tensor_ids: &[&str]) -> Result<()> {
    let mut dirs: HashSet<PathBuf> = HashSet::new();
    for tid in tensor_ids {
        if let Some(parent) = blob_abspath(root, tid).parent() {
            dirs.insert(parent.to_path_buf());
        }
    }
    for d in dirs {
        fs::create_dir_all(&d).with_context(|| format!("mkdir {:?}", d))?;
    }
    Ok(())
}

#[derive(Serialize)]
struct BlobHeader<'a> {
    id: &'a str,
    shape: &'a [usize],
    dtype: &'a str,
    engine_version: &'a str,
}

/// Serialize the tensor payload into a single-key safetensors file.
/// Returns the POSIX-style path relative to ``root`` (the storage_uri).
pub fn write_blob(
    root: &Path,
    tensor_id: &str,
    shape: &[usize],
    dtype: Dtype,
    dtype_str: &str,
    data: &[u8],
) -> Result<String> {
    let path = blob_abspath(root, tensor_id);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).with_context(|| format!("mkdir {:?}", parent))?;
    }

    let view = TensorView::new(dtype, shape.to_vec(), data)
        .map_err(|e| anyhow::anyhow!("build TensorView: {}", e))?;
    let header = BlobHeader {
        id: tensor_id,
        shape,
        dtype: dtype_str,
        engine_version: ENGINE_VERSION,
    };
    let header_json = serde_json::to_string(&header).context("serialize blob header json")?;
    let mut metadata: HashMap<String, String> = HashMap::new();
    metadata.insert(SAFETENSOR_KEY.to_string(), header_json);

    // safetensors::serialize wants a sorted map of name -> view.
    let mut tensors: BTreeMap<String, TensorView> = BTreeMap::new();
    tensors.insert(SAFETENSOR_KEY.to_string(), view);
    let bytes = safetensors::serialize(&tensors, &Some(metadata))
        .map_err(|e| anyhow::anyhow!("serialize safetensors: {}", e))?;

    // Write + fsync not strictly needed (ingest is idempotent on crash — the
    // tensor row only lands via the final SQL tx). Plain write_all suffices.
    let mut f = fs::File::create(&path).with_context(|| format!("create blob file {:?}", path))?;
    f.write_all(&bytes)
        .with_context(|| format!("write blob {:?}", path))?;

    Ok(blob_relpath(tensor_id).to_string_lossy().to_string())
}
