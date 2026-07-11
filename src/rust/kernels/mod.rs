//! Byte-level computational kernels exposed to Python via PyO3.
//!
//! - `sketch`     — BCS (BitCountSketch) fingerprinting
//! - `xor`        — XOR primitives
//! - `bitx`       — Bitx compression (plane split + XOR + zstd)
//! - `tensorx`    — TensorX compression (XOR delta + byte-plane + zstd)
//! - `fmpp`       — FM++ delta codec (FM-Delta FFI; `fmpp` feature only)

pub mod bitx;
#[cfg(feature = "fmpp")]
pub mod fmpp;
pub mod sketch;
pub mod tensorx;
pub mod xor;
