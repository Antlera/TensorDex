//! TensorDex Rust extension — PyO3 entry.
//!
//! Layout:
//!   `kernels/`      — byte-level compute (sketch, xor, bitx, tensorx)
//!   `codec/`        — generic `Compressor` trait + zstd/lz4 backends
//!   `compression/`  — plan schema + batch executor (future: planner)
//!   `metadata/`     — persistent state: fingerprint arena (→ SQLite in step 2)
//!   `resolvers/`    — local FS + S3 tensor I/O
//!   `utils/`        — transposition, zigzag, zstd xor dict

use pyo3::prelude::*;

pub mod codec;
pub mod compression;
pub mod ingest;
pub mod kernels;
pub mod metadata;
pub mod resolvers;
pub mod transfer;
pub mod utils;

use compression::execute::compress_py;
use compression::planner::{plan_attach_py, AttachPair, AttachPlan};
use ingest::hash::content_hash;
use ingest::ingest_from_safetensors_files;
use kernels::{sketch, tensorx};
use metadata::fingerprint::FingerprintStore;
use metadata::store::MetadataStore;
use transfer::serve_transfer;

#[pymodule]
fn _ops(_py: Python, m: &PyModule) -> PyResult<()> {
    // Metadata
    m.add_class::<FingerprintStore>()?;
    m.add_class::<MetadataStore>()?;

    // Ingest pipeline
    m.add_function(wrap_pyfunction!(ingest_from_safetensors_files, m)?)?;

    // Content-address hash (XXH3-128) — for read-time integrity checks
    m.add_function(wrap_pyfunction!(content_hash, m)?)?;

    // HTTP blob transfer data plane
    m.add_function(wrap_pyfunction!(serve_transfer, m)?)?;

    // Compression pipeline
    m.add_function(wrap_pyfunction!(compress_py, m)?)?;

    // Compression planner (FlexSplit attach stage)
    m.add_class::<AttachPair>()?;
    m.add_class::<AttachPlan>()?;
    m.add_function(wrap_pyfunction!(plan_attach_py, m)?)?;

    // TensorX (XOR delta + byte-plane split + zstd)
    m.add_function(wrap_pyfunction!(tensorx::compress_tensorx_rust, m)?)?;
    m.add_function(wrap_pyfunction!(tensorx::decompress_tensorx_rust, m)?)?;

    // FM++ delta codec (only when built with `--features fmpp`)
    #[cfg(feature = "fmpp")]
    m.add_function(wrap_pyfunction!(kernels::fmpp::compress_fmpp_rust, m)?)?;

    // BCS (BitCountSketch) fingerprint
    m.add_function(wrap_pyfunction!(sketch::compute_bcs_fingerprint_py, m)?)?;
    m.add_function(wrap_pyfunction!(sketch::compute_bcs_fingerprint_u16_py, m)?)?;
    m.add_function(wrap_pyfunction!(
        sketch::compute_bcs_fingerprints_batch_py,
        m
    )?)?;

    Ok(())
}
