//! Bitx Compression Algorithm
//!
//! This module implements the "Bitx" compression algorithm:
//! - Split BF16/bytes into planes (Transpose)
//! - XOR with base
//! - Zstd Compression
//!
//! Optimized Pipeline:
//! - Fused Kernel: (XOR + Transpose/Regroup) in a single pass.
//! - Parallel Execution: Uses Rayon.
//! - Efficient Memory Management: Avoids zero-initialization.

use pyo3::prelude::*;
use rayon::prelude::*;
use std::sync::atomic::{AtomicU64, Ordering};

const PAR_THRESHOLD: usize = 16384;

static BITX_FUSED_NS: AtomicU64 = AtomicU64::new(0);
static BITX_ZSTD_NS: AtomicU64 = AtomicU64::new(0);
static BITX_CALLS: AtomicU64 = AtomicU64::new(0);

pub fn get_bitx_timing() -> (u64, u64, u64) {
    (
        BITX_FUSED_NS.load(Ordering::Relaxed),
        BITX_ZSTD_NS.load(Ordering::Relaxed),
        BITX_CALLS.load(Ordering::Relaxed),
    )
}

pub fn reset_bitx_timing() {
    BITX_FUSED_NS.store(0, Ordering::Relaxed);
    BITX_ZSTD_NS.store(0, Ordering::Relaxed);
    BITX_CALLS.store(0, Ordering::Relaxed);
}

#[inline(always)]
fn compress_fused_regroup_xor(target: &[u8], base: Option<&[u8]>) -> Vec<u8> {
    let len_bytes = target.len();
    let count = len_bytes / 2;

    let mut result = Vec::with_capacity(len_bytes);
    unsafe {
        result.set_len(len_bytes);
    }

    let (exponents, mantissas) = result.split_at_mut(count);

    let process_chunk = |range: std::ops::Range<usize>, exp_ptr: *mut u8, man_ptr: *mut u8| {
        let mut i = range.start;
        let end = range.end;

        while i + 8 <= end {
            unsafe {
                let idx = i * 2;
                let t_ptr = target.as_ptr().add(idx) as *const u64;
                let t_0 = t_ptr.read_unaligned();
                let t_1 = t_ptr.add(1).read_unaligned();

                let (v_0, v_1) = if let Some(b) = base {
                    let b_ptr = b.as_ptr().add(idx) as *const u64;
                    (
                        t_0 ^ b_ptr.read_unaligned(),
                        t_1 ^ b_ptr.add(1).read_unaligned(),
                    )
                } else {
                    (t_0, t_1)
                };

                let mut exp_pack: u64 = 0;
                let mut sm_pack: u64 = 0;

                macro_rules! step {
                    ($j:expr, $chunk:expr) => {{
                        let shift = ($j % 4) * 16;
                        let val = (($chunk >> shift) & 0xFFFF) as u16;
                        let exp = ((val >> 7) & 0xFF) as u64;
                        let sm = (((val & 0x8000) >> 8) | (val & 0x7F)) as u64;
                        exp_pack |= exp << ($j * 8);
                        sm_pack |= sm << ($j * 8);
                    }};
                }

                step!(0, v_0);
                step!(1, v_0);
                step!(2, v_0);
                step!(3, v_0);
                step!(4, v_1);
                step!(5, v_1);
                step!(6, v_1);
                step!(7, v_1);

                (exp_ptr.add(i) as *mut u64).write_unaligned(exp_pack.to_le());
                (man_ptr.add(i) as *mut u64).write_unaligned(sm_pack.to_le());
            }
            i += 8;
        }

        while i < end {
            let idx = i * 2;
            let mut val = unsafe {
                let ptr = target.as_ptr().add(idx);
                u16::from_le_bytes([*ptr, *ptr.add(1)])
            };

            if let Some(b) = base {
                let b_val = unsafe {
                    let ptr = b.as_ptr().add(idx);
                    u16::from_le_bytes([*ptr, *ptr.add(1)])
                };
                val ^= b_val;
            }

            let exp = ((val >> 7) & 0xFF) as u8;
            let sm = (((val & 0x8000) >> 8) | (val & 0x7F)) as u8;

            unsafe {
                *exp_ptr.add(i) = exp;
                *man_ptr.add(i) = sm;
            }
            i += 1;
        }
    };

    if count >= PAR_THRESHOLD {
        let exp_addr = exponents.as_mut_ptr() as usize;
        let man_addr = mantissas.as_mut_ptr() as usize;
        let num_threads = rayon::current_num_threads();
        let chunk_size = ((count + num_threads - 1) / num_threads + 7) & !7;

        (0..num_threads).into_par_iter().for_each(|t| {
            let start = t * chunk_size;
            let end = (start + chunk_size).min(count);
            if start < end {
                unsafe {
                    process_chunk(start..end, exp_addr as *mut u8, man_addr as *mut u8);
                }
            }
        });
    } else {
        process_chunk(0..count, exponents.as_mut_ptr(), mantissas.as_mut_ptr());
    }

    result
}

#[inline(always)]
fn compress_fused_transpose_xor(target: &[u8], base: Option<&[u8]>, item_size: usize) -> Vec<u8> {
    if item_size <= 1 {
        if let Some(b) = base {
            let mut res = target.to_vec();
            for (r, &bv) in res.iter_mut().zip(b.iter()) {
                *r ^= bv;
            }
            return res;
        } else {
            return target.to_vec();
        }
    }

    let len_bytes = target.len();
    let num_elements = len_bytes / item_size;

    let mut result = Vec::with_capacity(len_bytes);
    unsafe {
        result.set_len(len_bytes);
    }

    let result_ptr = result.as_mut_ptr() as usize;

    if num_elements >= PAR_THRESHOLD {
        (0..num_elements).into_par_iter().for_each(|elem| {
            let src_idx = elem * item_size;
            for plane in 0..item_size {
                let byte_val = if let Some(b) = base {
                    unsafe {
                        *target.get_unchecked(src_idx + plane) ^ *b.get_unchecked(src_idx + plane)
                    }
                } else {
                    unsafe { *target.get_unchecked(src_idx + plane) }
                };

                let dst_idx = plane * num_elements + elem;
                unsafe {
                    *(result_ptr as *mut u8).add(dst_idx) = byte_val;
                }
            }
        });
    } else {
        for elem in 0..num_elements {
            let src_idx = elem * item_size;
            for plane in 0..item_size {
                let byte_val = if let Some(b) = base {
                    unsafe {
                        *target.get_unchecked(src_idx + plane) ^ *b.get_unchecked(src_idx + plane)
                    }
                } else {
                    unsafe { *target.get_unchecked(src_idx + plane) }
                };

                let dst_idx = plane * num_elements + elem;
                unsafe {
                    *(result_ptr as *mut u8).add(dst_idx) = byte_val;
                }
            }
        }
    }

    result
}

#[inline(always)]
fn decompress_fused_regroup_xor(transposed: &[u8], base: Option<&[u8]>) -> Vec<u8> {
    let len_bytes = transposed.len();
    let count = len_bytes / 2;

    let mut result = Vec::with_capacity(len_bytes);
    unsafe {
        result.set_len(len_bytes);
    }

    let (exponents, mantissas) = transposed.split_at(count);

    let process_chunk = |range: std::ops::Range<usize>, out_ptr: *mut u8| {
        let mut i = range.start;
        let end = range.end;

        while i + 8 <= end {
            unsafe {
                let exp_pack = (exponents.as_ptr().add(i) as *const u64).read_unaligned();
                let sm_pack = (mantissas.as_ptr().add(i) as *const u64).read_unaligned();

                let b_0: u64;
                let b_1: u64;
                if let Some(b) = base {
                    let b_ptr = b.as_ptr().add(i * 2) as *const u64;
                    b_0 = b_ptr.read_unaligned();
                    b_1 = b_ptr.add(1).read_unaligned();
                } else {
                    b_0 = 0;
                    b_1 = 0;
                }

                let mut out_0: u64 = 0;
                let mut out_1: u64 = 0;

                macro_rules! step_dec {
                    ($j:expr) => {{
                        let exp = ((exp_pack >> ($j * 8)) & 0xFF) as u16;
                        let sm = ((sm_pack >> ($j * 8)) & 0xFF) as u16;

                        let s = (sm & 0x80) << 8;
                        let e = exp << 7;
                        let m = sm & 0x7F;
                        let mut val = s | e | m;

                        let chunk_base = if $j < 4 { b_0 } else { b_1 };
                        let shift = ($j % 4) * 16;
                        let b_val = ((chunk_base >> shift) & 0xFFFF) as u16;
                        val ^= b_val;

                        if $j < 4 {
                            out_0 |= (val as u64) << ($j * 16);
                        } else {
                            out_1 |= (val as u64) << (($j - 4) * 16);
                        }
                    }};
                }

                step_dec!(0);
                step_dec!(1);
                step_dec!(2);
                step_dec!(3);
                step_dec!(4);
                step_dec!(5);
                step_dec!(6);
                step_dec!(7);

                (out_ptr.add(i * 2) as *mut u64).write_unaligned(out_0.to_le());
                (out_ptr.add(i * 2 + 8) as *mut u64).write_unaligned(out_1.to_le());
            }
            i += 8;
        }

        while i < end {
            unsafe {
                let exp = *exponents.get_unchecked(i) as u16;
                let sm = *mantissas.get_unchecked(i) as u16;

                let s = (sm & 0x80) << 8;
                let e = exp << 7;
                let m = sm & 0x7F;
                let mut val = s | e | m;

                if let Some(b) = base {
                    let ptr = b.as_ptr().add(i * 2);
                    let b_val = u16::from_le_bytes([*ptr, *ptr.add(1)]);
                    val ^= b_val;
                }

                let bytes = val.to_le_bytes();
                let dst = out_ptr.add(i * 2);
                *dst = bytes[0];
                *dst.add(1) = bytes[1];
            }
            i += 1;
        }
    };

    if count >= PAR_THRESHOLD {
        let out_addr = result.as_mut_ptr() as usize;
        let num_threads = rayon::current_num_threads();
        let chunk_size = ((count + num_threads - 1) / num_threads + 7) & !7;

        (0..num_threads).into_par_iter().for_each(|t| {
            let start = t * chunk_size;
            let end = (start + chunk_size).min(count);
            if start < end {
                unsafe {
                    process_chunk(start..end, out_addr as *mut u8);
                }
            }
        });
    } else {
        process_chunk(0..count, result.as_mut_ptr());
    }

    result
}

#[inline(always)]
fn decompress_fused_transpose_xor(
    transposed: &[u8],
    base: Option<&[u8]>,
    item_size: usize,
) -> Vec<u8> {
    if item_size <= 1 {
        if let Some(b) = base {
            let mut res = transposed.to_vec();
            for (r, &bv) in res.iter_mut().zip(b.iter()) {
                *r ^= bv;
            }
            return res;
        } else {
            return transposed.to_vec();
        }
    }

    let len_bytes = transposed.len();
    let num_elements = len_bytes / item_size;

    let mut result = Vec::with_capacity(len_bytes);
    unsafe {
        result.set_len(len_bytes);
    }

    let result_ptr = result.as_mut_ptr() as usize;

    if num_elements >= PAR_THRESHOLD {
        (0..num_elements).into_par_iter().for_each(|elem| {
            let dst_idx = elem * item_size;
            for plane in 0..item_size {
                let src_idx = plane * num_elements + elem;
                let mut byte_val = unsafe { *transposed.get_unchecked(src_idx) };

                if let Some(b) = base {
                    byte_val ^= unsafe { *b.get_unchecked(dst_idx + plane) };
                }

                unsafe {
                    *(result_ptr as *mut u8).add(dst_idx + plane) = byte_val;
                }
            }
        });
    } else {
        for elem in 0..num_elements {
            let dst_idx = elem * item_size;
            for plane in 0..item_size {
                let src_idx = plane * num_elements + elem;
                let mut byte_val = unsafe { *transposed.get_unchecked(src_idx) };

                if let Some(b) = base {
                    byte_val ^= unsafe { *b.get_unchecked(dst_idx + plane) };
                }

                unsafe {
                    *(result_ptr as *mut u8).add(dst_idx + plane) = byte_val;
                }
            }
        }
    }

    result
}

#[inline]
pub fn compress_bitx(
    target: &[u8],
    base: Option<&[u8]>,
    item_size: usize,
    level: i32,
) -> Result<Vec<u8>, String> {
    if let Some(b) = base {
        if target.len() != b.len() {
            return Err("Target and base must have the same length".into());
        }
    }

    let t0 = std::time::Instant::now();

    let transposed = if item_size == 2 {
        compress_fused_regroup_xor(target, base)
    } else {
        compress_fused_transpose_xor(target, base, item_size)
    };

    let t1 = std::time::Instant::now();
    BITX_FUSED_NS.fetch_add(t1.duration_since(t0).as_nanos() as u64, Ordering::Relaxed);

    let result = zstd::bulk::compress(&transposed, level)
        .map_err(|e| format!("Zstd compression failed: {}", e));

    let t2 = std::time::Instant::now();
    BITX_ZSTD_NS.fetch_add(t2.duration_since(t1).as_nanos() as u64, Ordering::Relaxed);
    BITX_CALLS.fetch_add(1, Ordering::Relaxed);

    result
}

#[inline]
pub fn decompress_bitx(
    compressed: &[u8],
    base: Option<&[u8]>,
    item_size: usize,
) -> Result<Vec<u8>, String> {
    let transposed = zstd::stream::decode_all(compressed)
        .map_err(|e| format!("Zstd decompression failed: {}", e))?;

    if let Some(b) = base {
        if transposed.len() != b.len() {
            return Err("Decompressed data and base must have the same length".into());
        }
    }

    let result = if item_size == 2 {
        decompress_fused_regroup_xor(&transposed, base)
    } else {
        decompress_fused_transpose_xor(&transposed, base, item_size)
    };

    Ok(result)
}

#[pyfunction]
#[pyo3(signature = (target, base=None, item_size=2, level=3))]
pub fn compress_bitx_rust(
    _py: Python,
    target: &[u8],
    base: Option<&[u8]>,
    item_size: usize,
    level: i32,
) -> PyResult<Vec<u8>> {
    compress_bitx(target, base, item_size, level)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))
}

#[pyfunction]
#[pyo3(signature = (compressed, base=None, item_size=2))]
pub fn decompress_bitx_rust(
    _py: Python,
    compressed: &[u8],
    base: Option<&[u8]>,
    item_size: usize,
) -> PyResult<Vec<u8>> {
    decompress_bitx(compressed, base, item_size)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))
}
