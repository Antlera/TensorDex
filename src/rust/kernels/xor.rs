use numpy::PyReadonlyArray1;
use pyo3::prelude::*;
use std::arch::x86_64::*;

/// XOR src_b into dst in-place.
pub fn xor_inplace(dst: &mut [u8], src_b: &[u8]) -> Result<(), String> {
    if dst.len() != src_b.len() {
        return Err("slice length mismatch".into());
    }

    let (dst_prefix, dst_u64, dst_suffix) = unsafe { dst.align_to_mut::<u64>() };
    let (src_prefix, src_u64, src_suffix) = unsafe { src_b.align_to::<u64>() };

    if dst_prefix.len() != src_prefix.len()
        || dst_suffix.len() != src_suffix.len()
        || dst_u64.len() != src_u64.len()
    {
        return Err("unaligned middle length mismatch".into());
    }

    for (d, s) in dst_prefix.iter_mut().zip(src_prefix.iter()) {
        *d ^= *s;
    }

    for (d, &s) in dst_u64.iter_mut().zip(src_u64.iter()) {
        *d ^= s;
    }

    for (d, s) in dst_suffix.iter_mut().zip(src_suffix.iter()) {
        *d ^= *s;
    }

    Ok(())
}

/// Compute XOR difference between two tensors and return as bytes
/// Highly optimized Rust implementation with SIMD and memory optimizations
#[pyfunction]
pub fn compute_xor_bytes_rust(
    _py: Python,
    target: PyReadonlyArray1<f32>,
    base: PyReadonlyArray1<f32>,
) -> PyResult<Vec<u8>> {
    let target = target.as_array();
    let base = base.as_array();

    if target.len() != base.len() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Tensors must have the same length",
        ));
    }

    let len = target.len();

    // Pre-allocate output buffer with exact size
    let mut bytes = Vec::with_capacity(len * 4);

    // Use SIMD-optimized vectorized XOR when possible
    if len >= 8 && is_x86_feature_detected!("avx2") {
        unsafe {
            compute_xor_bytes_avx2(target.as_ptr(), base.as_ptr(), &mut bytes, len);
        }
    } else {
        // Fallback to optimized scalar version
        compute_xor_bytes_scalar(target.as_ptr(), base.as_ptr(), &mut bytes, len);
    }

    Ok(bytes)
}

/// SIMD-optimized XOR computation using AVX2
#[target_feature(enable = "avx2")]
unsafe fn compute_xor_bytes_avx2(
    target_ptr: *const f32,
    base_ptr: *const f32,
    output: &mut Vec<u8>,
    len: usize,
) {
    let chunks = len / 8;
    let remainder = len % 8;

    // Process 8 f32 values at a time using AVX2
    for i in 0..chunks {
        let offset = i * 8;

        // Load 8 f32 values
        let target_vec = _mm256_loadu_ps(target_ptr.add(offset));
        let base_vec = _mm256_loadu_ps(base_ptr.add(offset));

        // Convert to integer representation
        let target_bits = _mm256_castps_si256(target_vec);
        let base_bits = _mm256_castps_si256(base_vec);

        // XOR operation
        let xor_result = _mm256_xor_si256(target_bits, base_bits);

        // Store result as bytes
        let result_bytes = std::mem::transmute::<__m256i, [u8; 32]>(xor_result);
        output.extend_from_slice(&result_bytes);
    }

    // Handle remainder with scalar operations
    if remainder > 0 {
        compute_xor_bytes_scalar(
            target_ptr.add(chunks * 8),
            base_ptr.add(chunks * 8),
            output,
            remainder,
        );
    }
}

/// Optimized scalar XOR computation
fn compute_xor_bytes_scalar(
    target_ptr: *const f32,
    base_ptr: *const f32,
    output: &mut Vec<u8>,
    len: usize,
) {
    // Process in chunks to improve cache locality
    const CHUNK_SIZE: usize = 1024;
    let chunks = len / CHUNK_SIZE;
    let remainder = len % CHUNK_SIZE;

    for chunk in 0..chunks {
        let start = chunk * CHUNK_SIZE;
        let end = start + CHUNK_SIZE;

        unsafe {
            for i in start..end {
                let target_val = *target_ptr.add(i);
                let base_val = *base_ptr.add(i);
                let xor_result = target_val.to_bits() ^ base_val.to_bits();
                output.extend_from_slice(&xor_result.to_le_bytes());
            }
        }
    }

    // Handle remainder
    if remainder > 0 {
        let start = chunks * CHUNK_SIZE;
        unsafe {
            for i in start..(start + remainder) {
                let target_val = *target_ptr.add(i);
                let base_val = *base_ptr.add(i);
                let xor_result = target_val.to_bits() ^ base_val.to_bits();
                output.extend_from_slice(&xor_result.to_le_bytes());
            }
        }
    }
}

/// Fast XOR zero ratio calculation for BF16 tensors
/// Optimized Rust implementation of xor_zero_ratio from utils.py
#[pyfunction]
pub fn xor_zero_ratio_rust(
    _py: Python,
    t1: PyReadonlyArray1<u16>,
    t2: PyReadonlyArray1<u16>,
) -> PyResult<f64> {
    let t1 = t1.as_array();
    let t2 = t2.as_array();

    if t1.len() != t2.len() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Tensors must have the same length",
        ));
    }

    // Parallel XOR and bit counting
    let ones_count: u64 = t1
        .iter()
        .zip(t2.iter())
        .map(|(&a, &b)| {
            let xor_result = a ^ b;
            // Count set bits using built-in popcount
            xor_result.count_ones() as u64
        })
        .sum();

    let total_bits = t1.len() as u64 * 16; // BF16 is 16 bits
    let zero_ratio = 1.0 - (ones_count as f64 / total_bits as f64);

    Ok(zero_ratio)
}

/// Optimized batch XOR operation for multiple tensor pairs
#[pyfunction]
pub fn batch_xor_rust(
    _py: Python,
    targets: Vec<PyReadonlyArray1<f32>>,
    bases: Vec<PyReadonlyArray1<f32>>,
) -> PyResult<Vec<Vec<u8>>> {
    if targets.len() != bases.len() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Target and base arrays must have the same length",
        ));
    }

    let results: Result<Vec<Vec<u8>>, PyErr> = targets
        .iter()
        .zip(bases.iter())
        .map(|(target, base)| compute_xor_bytes_rust(_py, target.clone(), base.clone()))
        .collect();

    results
}
