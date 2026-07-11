//! TensorX: XOR Delta + Byte-Plane Split + Zstd
//!
//! Pipeline:
//!   Compress: XOR(target, base) → byte-plane split → Zstd level 1
//!   Decompress: Zstd decode → byte-plane merge → XOR(result, base)
//!
//! XOR is its own inverse, so no ZigZag encoding is needed.
//! The byte-plane split concentrates zeros in the high byte planes
//! (since small XOR diffs have leading zero bytes), making Zstd very effective.
//!
//! Optimizations:
//!   - Fused SSE2 SIMD kernel: XOR + byte-split in a single pass (i16: 8 elems/iter)
//!   - Rayon parallel chunking for large tensors (>= PAR_THRESHOLD elements)
//!   - Auto sub-chunking for NUMA-scale inputs (>= 2MB)

use pyo3::prelude::*;
use rayon::prelude::*;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

const PAR_THRESHOLD: usize = 16384;
const SUB_CHUNK_BYTES: usize = 2 * 1024 * 1024;
const MAGIC_TX: u32 = 0x5458_4350; // "TXCP"

// =============================================================================
// Fused XOR + Byte-Plane Split kernels
// =============================================================================

// ── i16: XOR + split into 2 byte planes ──

#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn xor_split_i16_chunk_simd(
    target: &[u8],
    base: &[u8],
    range: std::ops::Range<usize>,
    p0_ptr: *mut u8, // low bytes
    p1_ptr: *mut u8, // high bytes
) {
    let mask_ff = _mm_set1_epi16(0x00FF);
    let mut i = range.start;
    let end = range.end;

    // Process 8 i16 elements (16 bytes) per iteration
    while i + 8 <= end {
        let byte_off = i * 2;
        let t_vec = _mm_loadu_si128(target.as_ptr().add(byte_off) as *const __m128i);
        let b_vec = _mm_loadu_si128(base.as_ptr().add(byte_off) as *const __m128i);

        // XOR delta
        let xor = _mm_xor_si128(t_vec, b_vec);

        // Byte-plane split
        let lo = _mm_and_si128(xor, mask_ff);
        let hi = _mm_srli_epi16(xor, 8);

        let lo_packed = _mm_packus_epi16(lo, lo);
        let hi_packed = _mm_packus_epi16(hi, hi);

        _mm_storel_epi64(p0_ptr.add(i) as *mut __m128i, lo_packed);
        _mm_storel_epi64(p1_ptr.add(i) as *mut __m128i, hi_packed);

        i += 8;
    }

    // Scalar tail
    while i < end {
        let byte_off = i * 2;
        let t_lo = *target.as_ptr().add(byte_off);
        let t_hi = *target.as_ptr().add(byte_off + 1);
        let b_lo = *base.as_ptr().add(byte_off);
        let b_hi = *base.as_ptr().add(byte_off + 1);
        *p0_ptr.add(i) = t_lo ^ b_lo;
        *p1_ptr.add(i) = t_hi ^ b_hi;
        i += 1;
    }
}

#[cfg(not(target_arch = "x86_64"))]
#[inline]
unsafe fn xor_split_i16_chunk_simd(
    target: &[u8],
    base: &[u8],
    range: std::ops::Range<usize>,
    p0_ptr: *mut u8,
    p1_ptr: *mut u8,
) {
    xor_split_i16_chunk_scalar(target, base, range, p0_ptr, p1_ptr);
}

#[inline]
unsafe fn xor_split_i16_chunk_scalar(
    target: &[u8],
    base: &[u8],
    range: std::ops::Range<usize>,
    p0_ptr: *mut u8,
    p1_ptr: *mut u8,
) {
    let mut i = range.start;
    let end = range.end;

    // Block of 8 via u64
    while i + 8 <= end {
        let byte_off = i * 2;
        let t_ptr = target.as_ptr().add(byte_off) as *const u64;
        let b_ptr = base.as_ptr().add(byte_off) as *const u64;

        let xor_0 = t_ptr.read_unaligned() ^ b_ptr.read_unaligned();
        let xor_1 = t_ptr.add(1).read_unaligned() ^ b_ptr.add(1).read_unaligned();

        let mut p0_pack: u64 = 0;
        let mut p1_pack: u64 = 0;

        macro_rules! step {
            ($j:expr, $chunk:expr) => {{
                let shift = ($j % 4) * 16;
                let val = (($chunk >> shift) & 0xFFFF) as u16;
                p0_pack |= ((val & 0xFF) as u64) << ($j * 8);
                p1_pack |= ((val >> 8) as u64) << ($j * 8);
            }};
        }

        step!(0, xor_0);
        step!(1, xor_0);
        step!(2, xor_0);
        step!(3, xor_0);
        step!(4, xor_1);
        step!(5, xor_1);
        step!(6, xor_1);
        step!(7, xor_1);

        (p0_ptr.add(i) as *mut u64).write_unaligned(p0_pack.to_le());
        (p1_ptr.add(i) as *mut u64).write_unaligned(p1_pack.to_le());
        i += 8;
    }

    while i < end {
        let byte_off = i * 2;
        *p0_ptr.add(i) = *target.as_ptr().add(byte_off) ^ *base.as_ptr().add(byte_off);
        *p1_ptr.add(i) = *target.as_ptr().add(byte_off + 1) ^ *base.as_ptr().add(byte_off + 1);
        i += 1;
    }
}

// ── i32: XOR + split into 4 byte planes ──

#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn xor_split_i32_chunk_simd(
    target: &[u8],
    base: &[u8],
    range: std::ops::Range<usize>,
    p0: *mut u8,
    p1: *mut u8,
    p2: *mut u8,
    p3: *mut u8,
) {
    let mut i = range.start;
    let end = range.end;

    // Process 4 i32 elements (16 bytes) per iteration
    // XOR result: [B0a B1a B2a B3a | B0b B1b B2b B3b | B0c B1c B2c B3c | B0d B1d B2d B3d]
    // Want: p0=[B0a,B0b,B0c,B0d], p1=[B1a,...], p2=[B2a,...], p3=[B3a,...]
    while i + 4 <= end {
        let byte_off = i * 4;
        let t_vec = _mm_loadu_si128(target.as_ptr().add(byte_off) as *const __m128i);
        let b_vec = _mm_loadu_si128(base.as_ptr().add(byte_off) as *const __m128i);

        let xor = _mm_xor_si128(t_vec, b_vec);

        // Scalar extract from SIMD register — 4 elements is too few for
        // multi-stage pack chains to outperform direct extraction.
        let v = std::mem::transmute::<__m128i, [u32; 4]>(xor);
        *(p0.add(i) as *mut u32) =
            (v[0] & 0xFF) | ((v[1] & 0xFF) << 8) | ((v[2] & 0xFF) << 16) | ((v[3] & 0xFF) << 24);
        *(p1.add(i) as *mut u32) = ((v[0] >> 8) & 0xFF)
            | (((v[1] >> 8) & 0xFF) << 8)
            | (((v[2] >> 8) & 0xFF) << 16)
            | (((v[3] >> 8) & 0xFF) << 24);
        *(p2.add(i) as *mut u32) = ((v[0] >> 16) & 0xFF)
            | (((v[1] >> 16) & 0xFF) << 8)
            | (((v[2] >> 16) & 0xFF) << 16)
            | (((v[3] >> 16) & 0xFF) << 24);
        *(p3.add(i) as *mut u32) =
            (v[0] >> 24) | ((v[1] >> 24) << 8) | ((v[2] >> 24) << 16) | ((v[3] >> 24) << 24);

        i += 4;
    }

    // Scalar tail
    while i < end {
        let byte_off = i * 4;
        let tp = target.as_ptr().add(byte_off);
        let bp = base.as_ptr().add(byte_off);
        *p0.add(i) = *tp ^ *bp;
        *p1.add(i) = *tp.add(1) ^ *bp.add(1);
        *p2.add(i) = *tp.add(2) ^ *bp.add(2);
        *p3.add(i) = *tp.add(3) ^ *bp.add(3);
        i += 1;
    }
}

#[cfg(not(target_arch = "x86_64"))]
#[inline]
unsafe fn xor_split_i32_chunk_simd(
    target: &[u8],
    base: &[u8],
    range: std::ops::Range<usize>,
    p0: *mut u8,
    p1: *mut u8,
    p2: *mut u8,
    p3: *mut u8,
) {
    let mut i = range.start;
    let end = range.end;
    while i < end {
        let byte_off = i * 4;
        let tp = target.as_ptr().add(byte_off);
        let bp = base.as_ptr().add(byte_off);
        *p0.add(i) = *tp ^ *bp;
        *p1.add(i) = *tp.add(1) ^ *bp.add(1);
        *p2.add(i) = *tp.add(2) ^ *bp.add(2);
        *p3.add(i) = *tp.add(3) ^ *bp.add(3);
        i += 1;
    }
}

// =============================================================================
// Fused XOR + Byte-Plane Merge (decompress) kernels
// =============================================================================

// ── i16 merge ──

#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn xor_merge_i16_chunk_simd(
    p0: &[u8], // low bytes
    p1: &[u8], // high bytes
    base: &[u8],
    range: std::ops::Range<usize>,
    out_ptr: *mut u8,
) {
    let mut i = range.start;
    let end = range.end;

    while i + 8 <= end {
        // Load 8 bytes from each plane
        let lo = _mm_loadl_epi64(p0.as_ptr().add(i) as *const __m128i);
        let hi = _mm_loadl_epi64(p1.as_ptr().add(i) as *const __m128i);

        // Unpack: interleave low and high bytes → 8 x u16
        let merged = _mm_unpacklo_epi8(lo, hi);

        // Load base and XOR
        let byte_off = i * 2;
        let b_vec = _mm_loadu_si128(base.as_ptr().add(byte_off) as *const __m128i);
        let result = _mm_xor_si128(merged, b_vec);

        _mm_storeu_si128(out_ptr.add(byte_off) as *mut __m128i, result);
        i += 8;
    }

    // Scalar tail
    while i < end {
        let byte_off = i * 2;
        let lo = *p0.as_ptr().add(i);
        let hi = *p1.as_ptr().add(i);
        *out_ptr.add(byte_off) = lo ^ *base.as_ptr().add(byte_off);
        *out_ptr.add(byte_off + 1) = hi ^ *base.as_ptr().add(byte_off + 1);
        i += 1;
    }
}

#[cfg(not(target_arch = "x86_64"))]
#[inline]
unsafe fn xor_merge_i16_chunk_simd(
    p0: &[u8],
    p1: &[u8],
    base: &[u8],
    range: std::ops::Range<usize>,
    out_ptr: *mut u8,
) {
    let mut i = range.start;
    let end = range.end;
    while i < end {
        let byte_off = i * 2;
        *out_ptr.add(byte_off) = *p0.as_ptr().add(i) ^ *base.as_ptr().add(byte_off);
        *out_ptr.add(byte_off + 1) = *p1.as_ptr().add(i) ^ *base.as_ptr().add(byte_off + 1);
        i += 1;
    }
}

// ── i32 merge ──

#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn xor_merge_i32_chunk_simd(
    p0: &[u8],
    p1: &[u8],
    p2: &[u8],
    p3: &[u8],
    base: &[u8],
    range: std::ops::Range<usize>,
    out_ptr: *mut u8,
) {
    let mut i = range.start;
    let end = range.end;

    while i + 4 <= end {
        let byte_off = i * 4;

        // Load 4 bytes from each plane
        let b0 = _mm_loadl_epi64(p0.as_ptr().add(i) as *const __m128i); // [b0_0, b0_1, b0_2, b0_3, ...]
        let b1 = _mm_loadl_epi64(p1.as_ptr().add(i) as *const __m128i);
        let b2 = _mm_loadl_epi64(p2.as_ptr().add(i) as *const __m128i);
        let b3 = _mm_loadl_epi64(p3.as_ptr().add(i) as *const __m128i);

        // Interleave: b0,b1 → u16 pairs, b2,b3 → u16 pairs
        let lo16 = _mm_unpacklo_epi8(b0, b1); // [b0_0,b1_0, b0_1,b1_1, ...]
        let hi16 = _mm_unpacklo_epi8(b2, b3); // [b2_0,b3_0, b2_1,b3_1, ...]

        // Interleave u16 pairs → u32
        let merged = _mm_unpacklo_epi16(lo16, hi16); // [b0_0,b1_0,b2_0,b3_0, ...]

        // Load base and XOR
        let base_vec = _mm_loadu_si128(base.as_ptr().add(byte_off) as *const __m128i);
        let result = _mm_xor_si128(merged, base_vec);

        _mm_storeu_si128(out_ptr.add(byte_off) as *mut __m128i, result);
        i += 4;
    }

    // Scalar tail
    while i < end {
        let byte_off = i * 4;
        let bp = base.as_ptr().add(byte_off);
        *out_ptr.add(byte_off) = *p0.as_ptr().add(i) ^ *bp;
        *out_ptr.add(byte_off + 1) = *p1.as_ptr().add(i) ^ *bp.add(1);
        *out_ptr.add(byte_off + 2) = *p2.as_ptr().add(i) ^ *bp.add(2);
        *out_ptr.add(byte_off + 3) = *p3.as_ptr().add(i) ^ *bp.add(3);
        i += 1;
    }
}

#[cfg(not(target_arch = "x86_64"))]
#[inline]
unsafe fn xor_merge_i32_chunk_simd(
    p0: &[u8],
    p1: &[u8],
    p2: &[u8],
    p3: &[u8],
    base: &[u8],
    range: std::ops::Range<usize>,
    out_ptr: *mut u8,
) {
    let mut i = range.start;
    let end = range.end;
    while i < end {
        let byte_off = i * 4;
        let bp = base.as_ptr().add(byte_off);
        *out_ptr.add(byte_off) = *p0.as_ptr().add(i) ^ *bp;
        *out_ptr.add(byte_off + 1) = *p1.as_ptr().add(i) ^ *bp.add(1);
        *out_ptr.add(byte_off + 2) = *p2.as_ptr().add(i) ^ *bp.add(2);
        *out_ptr.add(byte_off + 3) = *p3.as_ptr().add(i) ^ *bp.add(3);
        i += 1;
    }
}

// =============================================================================
// Top-level compress/decompress with Rayon parallelism
// =============================================================================

#[inline(always)]
pub fn compress_fused_i16(target: &[u8], base: &[u8]) -> Vec<u8> {
    let count = target.len() / 2;
    let len_bytes = target.len();
    let mut result = Vec::with_capacity(len_bytes);
    unsafe {
        result.set_len(len_bytes);
    }

    let (plane0, plane1) = result.split_at_mut(count);

    if count >= PAR_THRESHOLD && rayon::current_thread_index().is_none() {
        let p0_addr = plane0.as_mut_ptr() as usize;
        let p1_addr = plane1.as_mut_ptr() as usize;
        let num_threads = rayon::current_num_threads();
        let chunk_size = ((count + num_threads - 1) / num_threads + 7) & !7;

        (0..num_threads).into_par_iter().for_each(|t| {
            let start = t * chunk_size;
            let end = (start + chunk_size).min(count);
            if start < end {
                unsafe {
                    xor_split_i16_chunk_simd(
                        target,
                        base,
                        start..end,
                        p0_addr as *mut u8,
                        p1_addr as *mut u8,
                    );
                }
            }
        });
    } else {
        unsafe {
            xor_split_i16_chunk_simd(
                target,
                base,
                0..count,
                plane0.as_mut_ptr(),
                plane1.as_mut_ptr(),
            );
        }
    }

    result
}

#[inline(always)]
pub fn compress_fused_i32(target: &[u8], base: &[u8]) -> Vec<u8> {
    let count = target.len() / 4;
    let len_bytes = target.len();
    let mut result = Vec::with_capacity(len_bytes);
    unsafe {
        result.set_len(len_bytes);
    }

    let (p0_slice, rest) = result.split_at_mut(count);
    let (p1_slice, rest) = rest.split_at_mut(count);
    let (p2_slice, p3_slice) = rest.split_at_mut(count);

    if count >= PAR_THRESHOLD && rayon::current_thread_index().is_none() {
        let p0_addr = p0_slice.as_mut_ptr() as usize;
        let p1_addr = p1_slice.as_mut_ptr() as usize;
        let p2_addr = p2_slice.as_mut_ptr() as usize;
        let p3_addr = p3_slice.as_mut_ptr() as usize;
        let num_threads = rayon::current_num_threads();
        let chunk_size = ((count + num_threads - 1) / num_threads + 3) & !3;

        (0..num_threads).into_par_iter().for_each(|t| {
            let start = t * chunk_size;
            let end = (start + chunk_size).min(count);
            if start < end {
                unsafe {
                    xor_split_i32_chunk_simd(
                        target,
                        base,
                        start..end,
                        p0_addr as *mut u8,
                        p1_addr as *mut u8,
                        p2_addr as *mut u8,
                        p3_addr as *mut u8,
                    );
                }
            }
        });
    } else {
        unsafe {
            xor_split_i32_chunk_simd(
                target,
                base,
                0..count,
                p0_slice.as_mut_ptr(),
                p1_slice.as_mut_ptr(),
                p2_slice.as_mut_ptr(),
                p3_slice.as_mut_ptr(),
            );
        }
    }

    result
}

#[inline(always)]
pub fn decompress_fused_i16(transposed: &[u8], base: &[u8]) -> Vec<u8> {
    let count = transposed.len() / 2;
    let len_bytes = transposed.len();
    let mut result = Vec::with_capacity(len_bytes);
    unsafe {
        result.set_len(len_bytes);
    }

    let plane0 = &transposed[..count];
    let plane1 = &transposed[count..];

    if count >= PAR_THRESHOLD && rayon::current_thread_index().is_none() {
        let out_addr = result.as_mut_ptr() as usize;
        let num_threads = rayon::current_num_threads();
        let chunk_size = ((count + num_threads - 1) / num_threads + 7) & !7;

        (0..num_threads).into_par_iter().for_each(|t| {
            let start = t * chunk_size;
            let end = (start + chunk_size).min(count);
            if start < end {
                unsafe {
                    xor_merge_i16_chunk_simd(plane0, plane1, base, start..end, out_addr as *mut u8);
                }
            }
        });
    } else {
        unsafe {
            xor_merge_i16_chunk_simd(plane0, plane1, base, 0..count, result.as_mut_ptr());
        }
    }

    result
}

#[inline(always)]
pub fn decompress_fused_i32(transposed: &[u8], base: &[u8]) -> Vec<u8> {
    let count = transposed.len() / 4;
    let len_bytes = transposed.len();
    let mut result = Vec::with_capacity(len_bytes);
    unsafe {
        result.set_len(len_bytes);
    }

    let p0 = &transposed[..count];
    let p1 = &transposed[count..2 * count];
    let p2 = &transposed[2 * count..3 * count];
    let p3 = &transposed[3 * count..];

    if count >= PAR_THRESHOLD && rayon::current_thread_index().is_none() {
        let out_addr = result.as_mut_ptr() as usize;
        let num_threads = rayon::current_num_threads();
        let chunk_size = ((count + num_threads - 1) / num_threads + 3) & !3;

        (0..num_threads).into_par_iter().for_each(|t| {
            let start = t * chunk_size;
            let end = (start + chunk_size).min(count);
            if start < end {
                unsafe {
                    xor_merge_i32_chunk_simd(p0, p1, p2, p3, base, start..end, out_addr as *mut u8);
                }
            }
        });
    } else {
        unsafe {
            xor_merge_i32_chunk_simd(p0, p1, p2, p3, base, 0..count, result.as_mut_ptr());
        }
    }

    result
}

// =============================================================================
// Single-chunk compress/decompress (XOR + split + Zstd)
// =============================================================================

pub fn compress_tensorx(
    target: &[u8],
    base: &[u8],
    item_size: usize,
    level: i32,
) -> Result<Vec<u8>, String> {
    if target.len() != base.len() {
        return Err("target and base must have the same length".to_string());
    }

    let transposed = match item_size {
        2 => compress_fused_i16(target, base),
        4 => compress_fused_i32(target, base),
        _ => return Err(format!("Unsupported item_size: {}", item_size)),
    };

    zstd::bulk::compress(&transposed, level).map_err(|e| format!("Zstd compression failed: {}", e))
}

pub fn decompress_tensorx(
    compressed: &[u8],
    base: &[u8],
    item_size: usize,
) -> Result<Vec<u8>, String> {
    let transposed = zstd::stream::decode_all(compressed)
        .map_err(|e| format!("Zstd decompression failed: {}", e))?;

    match item_size {
        2 => Ok(decompress_fused_i16(&transposed, base)),
        4 => Ok(decompress_fused_i32(&transposed, base)),
        _ => Err(format!("Unsupported item_size: {}", item_size)),
    }
}

// =============================================================================
// Parallel auto-chunked wrappers
// =============================================================================

pub fn compress_tensorx_parallel(
    target: &[u8],
    base: &[u8],
    item_size: usize,
    level: i32,
) -> Result<Vec<u8>, String> {
    if target.len() != base.len() {
        return Err("target and base must have the same length".to_string());
    }

    let len = target.len();

    // Small tensor: single-chunk
    if len <= SUB_CHUNK_BYTES {
        return compress_tensorx(target, base, item_size, level);
    }

    // Split into aligned sub-chunks
    let chunk_size = (SUB_CHUNK_BYTES / item_size) * item_size;
    let mut offsets = Vec::new();
    let mut off = 0;
    while off < len {
        let end = (off + chunk_size).min(len);
        offsets.push((off, end));
        off = end;
    }
    let num_chunks = offsets.len();

    let compressed_chunks: Vec<Result<Vec<u8>, String>> = offsets
        .par_iter()
        .map(|&(start, end)| {
            compress_tensorx(&target[start..end], &base[start..end], item_size, level)
        })
        .collect();

    let mut chunks = Vec::with_capacity(num_chunks);
    for (i, result) in compressed_chunks.into_iter().enumerate() {
        chunks.push(result.map_err(|e| format!("sub-chunk {} failed: {}", i, e))?);
    }

    // Build framed output: magic + num_chunks + total_bytes + sizes + data
    let header_size = 4 + 4 + 8 + 4 * num_chunks;
    let body_size: usize = chunks.iter().map(|c| c.len()).sum();
    let mut out = Vec::with_capacity(header_size + body_size);

    out.extend_from_slice(&MAGIC_TX.to_le_bytes());
    out.extend_from_slice(&(num_chunks as u32).to_le_bytes());
    out.extend_from_slice(&(len as u64).to_le_bytes());
    for chunk in &chunks {
        out.extend_from_slice(&(chunk.len() as u32).to_le_bytes());
    }
    for chunk in &chunks {
        out.extend_from_slice(chunk);
    }

    Ok(out)
}

pub fn decompress_tensorx_parallel(
    compressed: &[u8],
    base: &[u8],
    item_size: usize,
) -> Result<Vec<u8>, String> {
    // Check for chunked magic
    if compressed.len() >= 4 {
        let magic = u32::from_le_bytes(compressed[0..4].try_into().unwrap());
        if magic == MAGIC_TX {
            return decompress_tensorx_chunked(compressed, base, item_size);
        }
    }
    decompress_tensorx(compressed, base, item_size)
}

fn decompress_tensorx_chunked(
    compressed: &[u8],
    base: &[u8],
    item_size: usize,
) -> Result<Vec<u8>, String> {
    if compressed.len() < 16 {
        return Err("Chunked data too short for header".to_string());
    }

    let num_chunks = u32::from_le_bytes(compressed[4..8].try_into().unwrap()) as usize;
    let original_total = u64::from_le_bytes(compressed[8..16].try_into().unwrap()) as usize;

    let sizes_end = 16 + 4 * num_chunks;
    if compressed.len() < sizes_end {
        return Err("Chunked data too short for size table".to_string());
    }

    let mut chunk_sizes = Vec::with_capacity(num_chunks);
    for i in 0..num_chunks {
        let off = 16 + 4 * i;
        chunk_sizes.push(u32::from_le_bytes(compressed[off..off + 4].try_into().unwrap()) as usize);
    }

    let mut chunk_offsets = Vec::with_capacity(num_chunks);
    let mut data_off = sizes_end;
    for &sz in &chunk_sizes {
        chunk_offsets.push(data_off);
        data_off += sz;
    }
    if data_off > compressed.len() {
        return Err(format!(
            "Chunked data truncated: need {} bytes, have {}",
            data_off,
            compressed.len()
        ));
    }

    let chunk_bytes = (SUB_CHUNK_BYTES / item_size) * item_size;
    let mut orig_offsets = Vec::with_capacity(num_chunks);
    let mut orig_off = 0;
    for _ in 0..num_chunks {
        let end = (orig_off + chunk_bytes).min(original_total);
        orig_offsets.push((orig_off, end));
        orig_off = end;
    }

    let decompressed_chunks: Vec<Result<Vec<u8>, String>> = (0..num_chunks)
        .into_par_iter()
        .map(|i| {
            let c_start = chunk_offsets[i];
            let c_end = c_start + chunk_sizes[i];
            let (o_start, o_end) = orig_offsets[i];
            decompress_tensorx(
                &compressed[c_start..c_end],
                &base[o_start..o_end],
                item_size,
            )
        })
        .collect();

    let mut result = Vec::with_capacity(original_total);
    for (i, chunk_result) in decompressed_chunks.into_iter().enumerate() {
        let chunk = chunk_result.map_err(|e| format!("sub-chunk {} failed: {}", i, e))?;
        result.extend_from_slice(&chunk);
    }

    Ok(result)
}

// =============================================================================
// Python Bindings
// =============================================================================

#[pyfunction]
#[pyo3(signature = (target, base, item_size=2, level=1))]
pub fn compress_tensorx_rust(
    py: Python,
    target: &[u8],
    base: &[u8],
    item_size: usize,
    level: i32,
) -> PyResult<PyObject> {
    // Return `bytes`, not a `Vec<u8>` (which PyO3 boxes into a list[int] —
    // catastrophically slow to build and consume for MB-scale blobs).
    let out = compress_tensorx_parallel(target, base, item_size, level)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))?;
    Ok(pyo3::types::PyBytes::new(py, &out).to_object(py))
}

#[pyfunction]
#[pyo3(signature = (compressed, base, item_size=2))]
pub fn decompress_tensorx_rust(
    py: Python,
    compressed: &[u8],
    base: &[u8],
    item_size: usize,
) -> PyResult<PyObject> {
    let out = decompress_tensorx_parallel(compressed, base, item_size)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))?;
    Ok(pyo3::types::PyBytes::new(py, &out).to_object(py))
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensorx_roundtrip_i16() {
        let target = vec![0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF0];
        let base = vec![0x10, 0x30, 0x50, 0x70, 0x90, 0xB0, 0xD0, 0xE0];

        let compressed = compress_tensorx(&target, &base, 2, 1).unwrap();
        let decompressed = decompress_tensorx(&compressed, &base, 2).unwrap();
        assert_eq!(target, decompressed);
    }

    #[test]
    fn test_tensorx_roundtrip_i32() {
        let target = vec![0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08];
        let base = vec![0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00];

        let compressed = compress_tensorx(&target, &base, 4, 1).unwrap();
        let decompressed = decompress_tensorx(&compressed, &base, 4).unwrap();
        assert_eq!(target, decompressed);
    }

    #[test]
    fn test_tensorx_roundtrip_large_parallel() {
        // Generate enough data to trigger parallel path
        let n = PAR_THRESHOLD * 4;
        let mut target = vec![0u8; n * 2];
        let mut base = vec![0u8; n * 2];
        for i in 0..n {
            let t: u16 = (i as u16).wrapping_mul(31);
            let b: u16 = (i as u16).wrapping_mul(29);
            target[i * 2] = (t & 0xFF) as u8;
            target[i * 2 + 1] = (t >> 8) as u8;
            base[i * 2] = (b & 0xFF) as u8;
            base[i * 2 + 1] = (b >> 8) as u8;
        }

        let compressed = compress_tensorx_parallel(&target, &base, 2, 1).unwrap();
        let decompressed = decompress_tensorx_parallel(&compressed, &base, 2).unwrap();
        assert_eq!(target, decompressed);
    }
}
