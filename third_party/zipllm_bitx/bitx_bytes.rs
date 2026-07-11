// Vendored verbatim from ZipLLM (https://github.com/ds2-lab/ZipLLM),
// src/bitx/bitx_bytes.rs — the official BitX implementation, Apache-2.0.
// Used by examples/zipllm_bitx_bench.rs to benchmark the baseline AS RELEASED.
//
// Minimal edits for standalone use (marked VENDOR below):
//   1. `use crate::config::CONFIG` removed; zstd worker count comes from
//      std::thread::available_parallelism() (same as ZipLLM's default config).
//   2. `use log::debug` removed; debug! timing lines commented out.
// No algorithmic or performance-relevant change.

use rayon::prelude::*;
use zstd::stream::{Encoder, Decoder};
use std::io::{Write, Read};
use std::time::Instant;
use anyhow::Result;

pub fn bitx_compress(data1: &[u8], data2: &[u8]) -> (Vec<u8>, Vec<u8>) {
    let min_len = std::cmp::min(data1.len(), data2.len()) & !1;
    assert!(min_len >= 2, "Input files too small");
    let chunk_count = min_len / 2;

    // Step 1: XOR and collect (exponent, sign_mantissa)
    let xor_start = Instant::now();
    let results: Vec<(u8, u8)> = (0..chunk_count).into_par_iter().map(|i| {
        let idx = i * 2;
        let v1 = u16::from_le_bytes([data1[idx], data1[idx + 1]]);
        let v2 = u16::from_le_bytes([data2[idx], data2[idx + 1]]);
        let xor = v1 ^ v2;

        let sign = ((xor >> 15) & 0x1) as u8;
        let exponent = ((xor >> 7) & 0xFF) as u8;
        let mantissa = (xor & 0x7F) as u8;
        let sign_mantissa = (sign << 7) | mantissa;

        (exponent, sign_mantissa)
    }).collect();
    let _xor_time = xor_start.elapsed();

    // Step 2: unzip into separate buffers
    let unzip_start = Instant::now();
    let mut exp_output = vec![0u8; chunk_count];
    let mut sm_output = vec![0u8; chunk_count];
    let chunk_size = 4096;
    exp_output
        .par_chunks_mut(chunk_size)
        .zip(sm_output.par_chunks_mut(chunk_size))
        .zip(results.par_chunks(chunk_size))
        .for_each(|((exp_chunk, sm_chunk), chunk_data)| {
            for ((e, s), (exp, sm)) in exp_chunk.iter_mut().zip(sm_chunk.iter_mut()).zip(chunk_data.iter()) {
                *e = *exp;
                *s = *sm;
            }
        });
    let _unzip_time = unzip_start.elapsed();

    // Step 3: compress in parallel using join
    let zstd_start = Instant::now();
    let (compressed_exp, compressed_sm): (Vec<u8>, Vec<u8>) = rayon::join(
        || zstd_compress_data(&exp_output, 3),
        || zstd_compress_data(&sm_output, 3),
    );
    let _zstd_time = zstd_start.elapsed();
    // VENDOR: debug!("XOR Time: {:.3}s", _xor_time.as_secs_f64());
    // VENDOR: debug!("Unzip Time: {:.3}s", _unzip_time.as_secs_f64());
    // VENDOR: debug!("ZSTD Total Time: {:.3}s", _zstd_time.as_secs_f64());

    (compressed_exp, compressed_sm)
}

pub fn bitx_decompress(data1: &[u8], compressed_exp: &[u8], compressed_sm: &[u8]) -> Vec<u8> {


    // decompress in parallel
    let (exp_output, sm_output): (Vec<u8>, Vec<u8>) = rayon::join(
        || zstd_decompress_data(compressed_exp),
        || zstd_decompress_data(compressed_sm),
    );
    
    // check if the decompressed lengths are the same
    assert_eq!(exp_output.len(), sm_output.len());
    let chunk_count = exp_output.len();

    // Combined reconstruct XOR and write output in one parallel step
    let mut output = vec![0u8; chunk_count * 2];
    
    output.par_chunks_mut(2).enumerate().for_each(|(i, chunk)| {
        // Reconstruct XOR result directly
        let exp = exp_output[i];
        let sm = sm_output[i];
        let sign = ((sm >> 7) & 0x1) as u16;
        let mantissa = (sm & 0x7F) as u16;
        let xor_result = (sign << 15) | ((exp as u16) << 7) | mantissa;
        
        // Apply XOR with base data and write to output
        let idx = i * 2;
        let orig1 = u16::from_le_bytes([data1[idx], data1[idx + 1]]);
        let recovered = xor_result ^ orig1;
        chunk[0] = (recovered & 0xFF) as u8;
        chunk[1] = (recovered >> 8) as u8;
    });
    


    output
}

/// ZSTD compression (supports multithreading)
pub fn zstd_compress_data(data: &[u8], level: i32) -> Vec<u8> {
    let mut output = Vec::with_capacity(data.len() / 3); // Estimated compression ratio
    let mut encoder = Encoder::new(&mut output, level).expect("Zstd encoder failed");
    // VENDOR: ZipLLM reads thread count from its config (default = all cores)
    let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    encoder.multithread(threads as u32).expect("Failed to set threads");
    encoder.write_all(data).expect("Write failed");
    encoder.finish().expect("Finish failed");
    output
}

/// ZSTD decompression
pub fn zstd_decompress_data(data: &[u8]) -> Vec<u8> {
    let mut decoder = Decoder::new(data).expect("Failed to create decoder");
    let mut decompressed = Vec::new();
    decoder.read_to_end(&mut decompressed).expect("Failed to decompress");
    decompressed
}

// optimized version 
/// BitX transform function that only returns the separated exponent and sign_mantissa arrays
pub fn bitx_transform_only(data1: &[u8], data2: &[u8]) -> Result<(Vec<u8>, Vec<u8>)> {
    let min_len = std::cmp::min(data1.len(), data2.len()) & !1;
    if min_len < 2 { return Err(anyhow::anyhow!("Input data too small for bitx transform")); }
    let pair_count = min_len / 2;
    let mut exp_output = vec![0u8; pair_count];
    let mut sm_output = vec![0u8; pair_count];

    let chunk_size = {
        let bytes = std::env::var("BITX_CHUNK_SIZE").ok().and_then(|s| s.parse::<usize>().ok()).filter(|&v| v > 0).unwrap_or(1_048_576);
        std::cmp::max(1, bytes / 2)
    };
    exp_output
        .par_chunks_mut(chunk_size)
        .zip(sm_output.par_chunks_mut(chunk_size))
        .enumerate()
        .for_each(|(chunk_idx, (exp_chunk, sm_chunk))| {
            let start = chunk_idx * chunk_size;
            let mut processed = 0usize;
            #[cfg(all(target_arch = "x86_64"))]
            {
                if is_x86_feature_detected!("avx2") {
                    unsafe {
                        use core::arch::x86_64::*;
                        let mut i = 0usize;
                        while i + 16 <= exp_chunk.len() {
                            let global_i = start + i;
                            let byte_off = global_i * 2;
                            let a = _mm256_loadu_si256(data1.as_ptr().add(byte_off) as *const __m256i);
                            let b = _mm256_loadu_si256(data2.as_ptr().add(byte_off) as *const __m256i);
                            let x = _mm256_xor_si256(a, b);
                            let exp16 = _mm256_and_si256(_mm256_srli_epi16(x, 7), _mm256_set1_epi16(0x00FF));
                            let mant16 = _mm256_and_si256(x, _mm256_set1_epi16(0x007F));
                            let sign16 = _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(x, 15), _mm256_set1_epi16(0x0001)), 7);
                            let sm16 = _mm256_or_si256(sign16, mant16);
                            let exp_lo = _mm256_castsi256_si128(exp16);
                            let exp_hi = _mm256_extracti128_si256(exp16, 1);
                            let exp_packed = _mm_packus_epi16(exp_lo, exp_hi);
                            _mm_storeu_si128(exp_chunk.as_mut_ptr().add(i) as *mut __m128i, exp_packed);
                            let sm_lo = _mm256_castsi256_si128(sm16);
                            let sm_hi = _mm256_extracti128_si256(sm16, 1);
                            let sm_packed = _mm_packus_epi16(sm_lo, sm_hi);
                            _mm_storeu_si128(sm_chunk.as_mut_ptr().add(i) as *mut __m128i, sm_packed);
                            i += 16;
                        }
                        processed = i;
                    }
                }
            }
            for j in processed..exp_chunk.len() {
                let global_i = start + j;
                let idx2 = global_i * 2;
                let v1 = u16::from_le_bytes([data1[idx2], data1[idx2 + 1]]);
                let v2 = u16::from_le_bytes([data2[idx2], data2[idx2 + 1]]);
                let xor = v1 ^ v2;
                let sign = ((xor >> 15) & 0x1) as u8;
                let exponent = ((xor >> 7) & 0xFF) as u8;
                let mantissa = (xor & 0x7F) as u8;
                exp_chunk[j] = exponent;
                sm_chunk[j] = (sign << 7) | mantissa;
            }
        });

    Ok((exp_output, sm_output))
}
/// Optimized BitX compression that handles smaller tensors
pub fn optimized_bitx_compress(data1: &[u8], data2: &[u8]) -> Result<(Vec<u8>, Vec<u8>)> {
    let min_len = std::cmp::min(data1.len(), data2.len()) & !1;
    if min_len < 2 {
        return Err(anyhow::anyhow!("Input data too small for bitx compression"));
    }
    let chunk_count = min_len / 2;

    // Pre-allocate the final output buffers, write in-place in parallel to avoid intermediate allocations
    let mut exp_output = vec![0u8; chunk_count];
    let mut sm_output = vec![0u8; chunk_count];

    // Use a configurable chunk size from env (BITX_CHUNK_SIZE bytes); convert to pairs (u16) count
    let chunk_size = {
        let bytes = std::env::var("BITX_CHUNK_SIZE")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .filter(|&v| v > 0)
            .unwrap_or(1_048_576);
        std::cmp::max(1, bytes / 2)
    };
        
    exp_output
        .par_chunks_mut(chunk_size)
        .zip(sm_output.par_chunks_mut(chunk_size))
        .enumerate()
        .for_each(|(chunk_idx, (exp_chunk, sm_chunk))| {
            let start = chunk_idx * chunk_size;
            let _end = start + exp_chunk.len();
            let mut processed = 0usize;

            // Use AVX2 SIMD (on x86_64 and if CPU supports it)
            #[cfg(all(target_arch = "x86_64"))]
            {
                if is_x86_feature_detected!("avx2") {
                    unsafe {
                        use core::arch::x86_64::*;
                        let mut i = 0usize;
                        while i + 16 <= exp_chunk.len() {
                            let global_i = start + i;
                            let byte_off = global_i * 2;

                            let a = _mm256_loadu_si256(data1.as_ptr().add(byte_off) as *const __m256i);
                            let b = _mm256_loadu_si256(data2.as_ptr().add(byte_off) as *const __m256i);
                            let x = _mm256_xor_si256(a, b);

                            // exponent = ((x >> 7) & 0x00FF)
                            let exp16 = _mm256_and_si256(_mm256_srli_epi16(x, 7), _mm256_set1_epi16(0x00FF));
                            // mantissa = (x & 0x007F)
                            let mant16 = _mm256_and_si256(x, _mm256_set1_epi16(0x007F));
                            // sign << 7 = ((x >> 15) & 1) << 7
                            let sign16 = _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(x, 15), _mm256_set1_epi16(0x0001)), 7);
                            let sm16 = _mm256_or_si256(sign16, mant16);

                            // Pack the low bytes of 16 u16s into 16 u8s
                            let exp_lo = _mm256_castsi256_si128(exp16);
                            let exp_hi = _mm256_extracti128_si256(exp16, 1);
                            let exp_packed = _mm_packus_epi16(exp_lo, exp_hi);
                            _mm_storeu_si128(exp_chunk.as_mut_ptr().add(i) as *mut __m128i, exp_packed);

                            let sm_lo = _mm256_castsi256_si128(sm16);
                            let sm_hi = _mm256_extracti128_si256(sm16, 1);
                            let sm_packed = _mm_packus_epi16(sm_lo, sm_hi);
                            _mm_storeu_si128(sm_chunk.as_mut_ptr().add(i) as *mut __m128i, sm_packed);

                            i += 16;
                        }
                        processed = i;
                    }
                }
            }

            // Handle the remainder (or non-AVX2 environment)
            for j in processed..exp_chunk.len() {
                let global_i = start + j;
                let idx2 = global_i * 2;
                let v1 = u16::from_le_bytes([data1[idx2], data1[idx2 + 1]]);
                let v2 = u16::from_le_bytes([data2[idx2], data2[idx2 + 1]]);
                let xor = v1 ^ v2;

                let sign = ((xor >> 15) & 0x1) as u8;
                let exponent = ((xor >> 7) & 0xFF) as u8;
                let mantissa = (xor & 0x7F) as u8;

                exp_chunk[j] = exponent;
                sm_chunk[j] = (sign << 7) | mantissa;
            }
        });

    const LARGE_DATA_THRESHOLD: usize = 10 * 1024 * 1024; // 10MB
    
    if data2.len() > LARGE_DATA_THRESHOLD {
        let compression_chunk_size = std::env::var("BITX_CHUNK_SIZE")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .filter(|&v| v > 0)
            .unwrap_or(1_048_576);
        let zstd_level = 3;

        let (compressed_exp_chunks, compressed_sm_chunks): (Vec<Vec<u8>>, Vec<Vec<u8>>) = rayon::join(
            || exp_output
                .par_chunks(compression_chunk_size)
                .map(|chunk| zstd_compress_data(chunk, zstd_level))
                .collect(),
            || sm_output
                .par_chunks(compression_chunk_size)
                .map(|chunk| zstd_compress_data(chunk, zstd_level))
                .collect(),
        );

        let total_exp_size: usize = compressed_exp_chunks.iter().map(|v| v.len()).sum();
        let total_sm_size: usize = compressed_sm_chunks.iter().map(|v| v.len()).sum();

        let mut compressed_exp = Vec::with_capacity(total_exp_size);
        let mut compressed_sm = Vec::with_capacity(total_sm_size);

        for chunk in compressed_exp_chunks { compressed_exp.extend_from_slice(&chunk); }
        for chunk in compressed_sm_chunks { compressed_sm.extend_from_slice(&chunk); }

        Ok((compressed_exp, compressed_sm))
    } else {
        let (compressed_exp, compressed_sm): (Vec<u8>, Vec<u8>) = rayon::join(
            || zstd_compress_data(&exp_output, 3),
            || zstd_compress_data(&sm_output, 3),
        );

        Ok((compressed_exp, compressed_sm))
    }
}


/// Result structure for BitX compression benchmark
#[derive(Debug)]
pub struct BitxBenchmarkResult {
    pub total_tensors: usize,
    pub processed_tensors: usize,
    pub skipped_tensors: usize,
    pub total_original_size_mb: f64,
    pub total_compressed_size_mb: f64,
    pub compression_ratio: f64,
    pub compression_time_seconds: f64,
    pub throughput_mb_per_sec: f64,
}

/// Structure to represent a tensor in a model
#[derive(Debug, Clone)]
pub struct ModelTensor {
    pub name: String,
    pub data: Vec<u8>,
    pub shape: Vec<usize>,
    pub dtype: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;
    use std::fs;


    #[test]
    fn test_bitx_compress() {
        let size = 3*1024 * 1024 * 1024; // 1GB
        let mut data1 = vec![0u8; size];
        let mut data2 = vec![0u8; size];
        
        use rand::Rng;
        let mut rng = rand::thread_rng();
        rng.fill(&mut data1[..]);
        rng.fill(&mut data2[..]);
        

        // throughput
        let start: Instant = Instant::now();
        let (_compressed_exp, _compressed_sm) = bitx_compress(&data1, &data2);
        let end = Instant::now();
        let duration = end.duration_since(start);
        println!("Throughput: {} GB/s", 3.0 / duration.as_secs_f64());
    }
    #[test]
    fn test_bitx_decompress() {
        let size = 1024 * 1024 * 1024; // 1GB
        let mut data1 = vec![0u8; size];
        let mut data2 = vec![0u8; size];
        
        use rand::Rng;
        let mut rng = rand::thread_rng();
        rng.fill(&mut data1[..]);
        rng.fill(&mut data2[..]);
        
        let (compressed_exp, compressed_sm) = bitx_compress(&data1, &data2);
        let decompressed = bitx_decompress(&data1, &compressed_exp, &compressed_sm);
        //check if decompressed is equal to data2, don't use assert_eq! because it will print the whole vectors
        for i in 0..size {
            if decompressed[i] != data2[i] {
                println!("Decompressed and data2 mismatch at index {}", i);
                break;
            }
        }
    }
    #[test]
    fn test_bitx_throughput() {
        //init log
        let _ = env_logger::builder().is_test(true).try_init();
        let size = 3*1024 * 1024 * 1024; // 1GB
        let mut data1 = vec![0u8; size];
        let mut data2 = vec![0u8; size];
        let mut rng = rand::thread_rng();
        rng.fill(&mut data1[..]);
        rng.fill(&mut data2[..]);

        let start = Instant::now();
        let (compressed_exp, compressed_sm) = bitx_compress(&data1, &data2);
        let duration = start.elapsed();
        println!("Compression Throughput: {:.2} GB/s", size as f64 / 1e9 / duration.as_secs_f64());

        let start_decomp = Instant::now();
        let decompressed = bitx_decompress(&data1, &compressed_exp, &compressed_sm);
        let duration_decomp = start_decomp.elapsed();
        println!("Decompression Throughput: {:.2} GB/s", size as f64 / 1e9 / duration_decomp.as_secs_f64());

        assert_eq!(decompressed, data2);
    }   
    #[test]
    fn test_bitx_throughput_real_files() {
        let _ = env_logger::builder().is_test(true).try_init();

        let path1 = "input1.bin";
        let path2 = "input2.bin";

        let start_read = Instant::now();
        let data1 = fs::read(path1).expect("Failed to read file1");
        let data2 = fs::read(path2).expect("Failed to read file2");
        let read_time = start_read.elapsed().as_secs_f64();

        let size = std::cmp::min(data1.len(), data2.len());

        println!(
            "Read {} MB from each file in {:.2} s ({:.2} MB/s)",
            size / 1024 / 1024,
            read_time,
            size as f64 / 1024.0 / 1024.0 / read_time
        );

        let start = Instant::now();
        let (compressed_exp, compressed_sm) = bitx_compress(&data1, &data2);
        let duration = start.elapsed();
        println!(
            "Compression Throughput: {:.2} GB/s",
            size as f64 / 1e9 / duration.as_secs_f64()
        );

        let start_decomp = Instant::now();
        let decompressed = bitx_decompress(&data1, &compressed_exp, &compressed_sm);
        let duration_decomp = start_decomp.elapsed();
        println!(
            "Decompression Throughput: {:.2} GB/s",
            size as f64 / 1e9 / duration_decomp.as_secs_f64()
        );

        assert_eq!(decompressed.len(), size);
        assert_eq!(&decompressed[..size], &data2[..size]);
    }

}