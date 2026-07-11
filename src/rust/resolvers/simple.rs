use super::{TensorBatch, TensorFileResolver, TensorSlice};
use memmap2::Mmap;
use rayon::prelude::*;
use safetensors::SafeTensors;
use std::collections::HashSet;
use std::env;
use std::fs::File;
use std::io::Read;
use std::path::Path;
use std::sync::Arc;

#[cfg(target_os = "linux")]
use std::os::unix::fs::FileExt;

#[cfg(not(target_os = "linux"))]
use safetensors::tensor::Dtype;

const ENV_FADVISE_MODE: &str = "TENSORDEX_RESOLVER_FADVISE_MODE";

/// Simple tensor file resolver that assumes tensors are stored as individual safetensors files
pub struct SimpleTensorResolver {
    pub tensor_dir: String,
}

impl SimpleTensorResolver {
    pub fn new(tensor_dir: String) -> Self {
        Self { tensor_dir }
    }

    fn resolve_tensor_file_path(&self, tensor_id: &str) -> Result<String, String> {
        let hashed_path = TensorFileResolver::resolve_tensor_path(self, tensor_id)?;

        if Path::new(&hashed_path).exists() {
            Ok(hashed_path)
        } else {
            Err(format!(
                "Tensor file not found for id {}. Expected at {}",
                tensor_id, hashed_path
            ))
        }
    }

    /// Create an mmap for the given file WITHOUT pre-populating pages.
    /// Callers should populate only the needed range after finding the tensor offset.
    fn create_mmap(&self, path: &str) -> Result<(File, Mmap), String> {
        let file =
            File::open(path).map_err(|e| format!("Failed to open tensor file {}: {}", path, e))?;
        let mmap = unsafe { Mmap::map(&file) }
            .map_err(|e| format!("Failed to memory-map tensor file {}: {}", path, e))?;
        Ok((file, mmap))
    }

    fn prefetch_files(&self, tensor_ids: &[String]) -> Result<(), String> {
        let mut prefetched_files = HashSet::new();

        for tensor_id in tensor_ids {
            if let Ok(tensor_path) = self.resolve_tensor_file_path(tensor_id) {
                if prefetched_files.insert(tensor_path.clone()) {
                    if let Ok(file) = File::open(&tensor_path) {
                        apply_prefetch_advice(&file);
                    }
                }
            }
        }

        Ok(())
    }

    /// Load tensor data directly into dst using pread(), avoiding the
    /// mmap → MADV_POPULATE_READ → memcpy path that was 10× slower due to
    /// memory bandwidth saturation with many parallel threads.
    ///
    /// New path: read header (small) → parse → pread() tensor data → dst
    /// This gives ~5 GB/s vs ~0.5 GB/s for mmap+memcpy under high thread counts.
    #[cfg(target_os = "linux")]
    fn load_into_slice(
        &self,
        tensor_id: &str,
        dst: &mut [u8],
        expected_shape: &[usize],
        elem_size: usize,
    ) -> Result<(), String> {
        let tensor_path = self.resolve_tensor_file_path(tensor_id)?;
        let mut file = File::open(&tensor_path)
            .map_err(|e| format!("Failed to open {}: {}", tensor_path, e))?;

        // 1. Read safetensors header size (8 bytes, u64 LE)
        let mut header_size_buf = [0u8; 8];
        file.read_exact(&mut header_size_buf)
            .map_err(|e| format!("Failed to read header size from {}: {}", tensor_path, e))?;
        let header_size = u64::from_le_bytes(header_size_buf) as usize;

        // 2. Read and parse header JSON
        let mut header_buf = vec![0u8; header_size];
        file.read_exact(&mut header_buf)
            .map_err(|e| format!("Failed to read header from {}: {}", tensor_path, e))?;
        let data_section_offset = 8 + header_size;

        let (data_start, data_end, dtype_str, shape) =
            parse_safetensors_header(&header_buf, tensor_id, expected_shape)?;

        let tensor_len = data_end - data_start;

        // 3. Validate
        let dtype_size = dtype_name_to_size(&dtype_str)?;
        if dtype_size != elem_size {
            return Err(format!(
                "Element size mismatch for {}: expected {} bytes, file reports {} bytes ({})",
                tensor_id, elem_size, dtype_size, dtype_str
            ));
        }

        if !expected_shape.is_empty() {
            let expected_len = expected_shape
                .iter()
                .product::<usize>()
                .saturating_mul(elem_size);
            if expected_len != tensor_len {
                return Err(format!(
                    "Size mismatch for {}: expected {} bytes (shape {:?} × {}), got {} bytes",
                    tensor_id, expected_len, expected_shape, elem_size, tensor_len
                ));
            }
            if shape != expected_shape {
                return Err(format!(
                    "Shape mismatch for {}: expected {:?}, got {:?}",
                    tensor_id, expected_shape, shape
                ));
            }
        }

        if tensor_len != dst.len() {
            return Err(format!(
                "Destination buffer length mismatch for {}: dst has {} bytes but tensor has {} bytes",
                tensor_id, dst.len(), tensor_len
            ));
        }

        // 4. pread() tensor data directly into dst — single syscall, no extra copy
        let file_offset = (data_section_offset + data_start) as u64;
        pread_exact(&file, dst, file_offset)
            .map_err(|e| format!("pread failed for {}: {}", tensor_path, e))?;

        Ok(())
    }

    /// Fallback for non-Linux: use original mmap+copy path
    #[cfg(not(target_os = "linux"))]
    fn load_into_slice(
        &self,
        tensor_id: &str,
        dst: &mut [u8],
        expected_shape: &[usize],
        elem_size: usize,
    ) -> Result<(), String> {
        let tensor_path = self.resolve_tensor_file_path(tensor_id)?;
        let (_file, mmap) = self.create_mmap(&tensor_path)?;

        let slice = unsafe { std::slice::from_raw_parts(mmap.as_ptr(), mmap.len()) };
        let safetensors = SafeTensors::deserialize(slice)
            .map_err(|e| format!("Failed to deserialize SafeTensors: {}", e))?;

        let tensor_key = select_tensor_key(&safetensors, tensor_id, expected_shape)?;
        let tensor_view = safetensors
            .tensor(tensor_key)
            .map_err(|e| format!("Failed to get tensor '{}': {}", tensor_key, e))?;

        if !expected_shape.is_empty() && tensor_view.shape() != expected_shape {
            return Err(format!(
                "Shape mismatch for {}: expected {:?}, got {:?}",
                tensor_id,
                expected_shape,
                tensor_view.shape()
            ));
        }

        let tensor_data = tensor_view.data();
        let dtype_size = tensor_dtype_size(tensor_view.dtype());

        if dtype_size != elem_size {
            return Err(format!(
                "Element size mismatch for {}: expected {} bytes, file reports {} bytes",
                tensor_id, elem_size, dtype_size
            ));
        }

        if tensor_data.len() != dst.len() {
            return Err(format!(
                "Destination buffer length mismatch for {}: dst has {} bytes but tensor has {} bytes",
                tensor_id, dst.len(), tensor_data.len()
            ));
        }

        dst.copy_from_slice(tensor_data);
        Ok(())
    }

    fn load_tensor_mmap(
        &self,
        tensor_id: &str,
        expected_shape: &[usize],
    ) -> Result<TensorSlice, String> {
        let tensor_path = self.resolve_tensor_file_path(tensor_id)?;
        let (_file, mmap) = self.create_mmap(&tensor_path)?;
        let mmap_arc = Arc::new(mmap);

        let slice = unsafe { std::slice::from_raw_parts(mmap_arc.as_ptr(), mmap_arc.len()) };
        let safetensors = SafeTensors::deserialize(slice)
            .map_err(|e| format!("Failed to deserialize SafeTensors: {}", e))?;

        let tensor_key = select_tensor_key(&safetensors, tensor_id, expected_shape)?;

        let tensor_view = safetensors
            .tensor(tensor_key)
            .map_err(|e| format!("Failed to get tensor '{}': {}", tensor_key, e))?;

        if !expected_shape.is_empty() && tensor_view.shape() != expected_shape {
            return Err(format!(
                "Shape mismatch for {}: expected {:?}, got {:?}",
                tensor_id,
                expected_shape,
                tensor_view.shape()
            ));
        }

        let tensor_data = tensor_view.data();

        // Hint the kernel to prefetch tensor data pages asynchronously.
        // We intentionally use MADV_WILLNEED (non-blocking) instead of
        // MADV_POPULATE_READ (blocking) because:
        //   - With many parallel threads, the synchronous populate causes
        //     heavy kernel contention (mmap_lock, TLB shootdowns) and is
        //     no faster than letting pages fault lazily from warm cache.
        //   - WILLNEED starts background readahead while we return immediately,
        //     allowing the compress phase to begin sooner and overlap I/O.
        #[cfg(target_os = "linux")]
        {
            apply_mmap_willneed_range(tensor_data.as_ptr(), tensor_data.len());
        }

        Ok(TensorSlice::new(mmap_arc, tensor_data))
    }
}

impl TensorFileResolver for SimpleTensorResolver {
    fn tensor_dir(&self) -> &str {
        &self.tensor_dir
    }

    fn bulk_load_tensors_mmap(
        &self,
        requests: &[(String, Vec<usize>)],
    ) -> Result<TensorBatch, String> {
        // Prefetch all files first
        let tensor_ids: Vec<String> = requests.iter().map(|(id, _)| id.clone()).collect();
        self.prefetch_files(&tensor_ids)?;

        let results: Result<Vec<TensorSlice>, String> = requests
            .iter() // Note: cannot use par_iter with lifetime constraints
            .map(|(tensor_id, shape)| self.load_tensor_mmap(tensor_id, shape))
            .collect();
        results
    }

    fn bulk_load_slices(
        &self,
        requests: &[(String, usize, usize, Vec<usize>)],
        dsts: &mut [u8],
    ) -> Result<(), String> {
        // Prefetch all files first
        let tensor_ids: Vec<String> = requests.iter().map(|(id, _, _, _)| id.clone()).collect();
        self.prefetch_files(&tensor_ids)?;

        let base_ptr_addr = dsts.as_mut_ptr() as usize;
        let dst_len = dsts.len();

        requests
            .par_iter()
            .try_for_each(|(tid, off, elem_size, shape)| {
                let byte_offset = *off;
                let byte_len = shape.iter().product::<usize>() * *elem_size;

                if byte_offset + byte_len > dst_len {
                    return Err(format!(
                        "Offset {} + length {} exceeds buffer size {}",
                        byte_offset, byte_len, dst_len
                    ));
                }

                // Recover pointer in thread
                let base_ptr = base_ptr_addr as *mut u8;
                let dst =
                    unsafe { std::slice::from_raw_parts_mut(base_ptr.add(byte_offset), byte_len) };
                self.load_into_slice(tid, dst, shape, *elem_size)
            })
    }
}

#[cfg(not(target_os = "linux"))]
fn tensor_dtype_size(dtype: Dtype) -> usize {
    dtype.size()
}

pub fn select_tensor_key<'a>(
    safetensors: &'a SafeTensors<'a>,
    tensor_id: &str,
    expected_shape: &[usize],
) -> Result<&'a str, String> {
    let mut fallback: Option<&'a str> = None;

    for name in safetensors
        .names()
        .into_iter()
        .filter(|key| !key.starts_with('_'))
    {
        let view = safetensors
            .tensor(name)
            .map_err(|e| format!("Failed to get tensor '{}': {}", name, e))?;

        if expected_shape.is_empty() || view.shape() == expected_shape {
            return Ok(name);
        }

        if fallback.is_none() {
            fallback = Some(name);
        }
    }

    fallback.ok_or_else(|| format!("No tensor payload found in file for id {}", tensor_id))
}

fn env_var(name: &str) -> Option<String> {
    env::var(name).ok()
}

// =============================================================================
// Safetensors header parsing (for pread path)
// =============================================================================

/// Minimal struct for parsing a single tensor entry from the safetensors JSON header.
#[derive(serde::Deserialize)]
struct SafetensorsHeaderEntry {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: [usize; 2],
}

/// Parse safetensors JSON header to find the best-matching tensor.
/// Returns (data_start, data_end, dtype_string, shape).
/// Uses the same key-selection logic as `select_tensor_key`.
fn parse_safetensors_header(
    header_buf: &[u8],
    tensor_id: &str,
    expected_shape: &[usize],
) -> Result<(usize, usize, String, Vec<usize>), String> {
    let header: serde_json::Map<String, serde_json::Value> = serde_json::from_slice(header_buf)
        .map_err(|e| format!("Failed to parse safetensors header: {}", e))?;

    let mut best: Option<(&str, &serde_json::Value)> = None;
    let mut fallback: Option<(&str, &serde_json::Value)> = None;

    for (key, value) in &header {
        if key.starts_with('_') {
            continue;
        }
        // Try to parse shape from this entry
        if let Some(shape_val) = value.get("shape") {
            if let Some(shape_arr) = shape_val.as_array() {
                let shape: Vec<usize> = shape_arr
                    .iter()
                    .filter_map(|v| v.as_u64().map(|n| n as usize))
                    .collect();

                if fallback.is_none() {
                    fallback = Some((key.as_str(), value));
                }

                if expected_shape.is_empty() || shape == expected_shape {
                    best = Some((key.as_str(), value));
                    break;
                }
            }
        }
    }

    let (_key, entry_value) = best
        .or(fallback)
        .ok_or_else(|| format!("No tensor payload found in header for id {}", tensor_id))?;

    let entry: SafetensorsHeaderEntry = serde_json::from_value(entry_value.clone())
        .map_err(|e| format!("Failed to parse tensor entry: {}", e))?;

    Ok((
        entry.data_offsets[0],
        entry.data_offsets[1],
        entry.dtype,
        entry.shape,
    ))
}

/// Map safetensors dtype name string to byte size.
fn dtype_name_to_size(dtype: &str) -> Result<usize, String> {
    match dtype {
        "BOOL" | "U8" | "I8" => Ok(1),
        "F16" | "BF16" | "I16" | "U16" => Ok(2),
        "F32" | "I32" | "U32" => Ok(4),
        "F64" | "I64" | "U64" => Ok(8),
        _ => Err(format!("Unknown safetensors dtype: {}", dtype)),
    }
}

/// pread() loop that reads the entire requested range into dst.
/// Unlike a single pread() call which may return fewer bytes, this loops
/// until all bytes are read or an error occurs.
#[cfg(target_os = "linux")]
fn pread_exact(file: &File, dst: &mut [u8], mut offset: u64) -> Result<(), String> {
    let mut total_read = 0usize;
    while total_read < dst.len() {
        let n = file
            .read_at(&mut dst[total_read..], offset)
            .map_err(|e| format!("pread error at offset {}: {}", offset, e))?;
        if n == 0 {
            return Err(format!(
                "Unexpected EOF: read {} of {} bytes at offset {}",
                total_read,
                dst.len(),
                offset
            ));
        }
        total_read += n;
        offset += n as u64;
    }
    Ok(())
}

#[cfg(target_os = "linux")]
fn apply_prefetch_advice(file: &File) {
    linux::apply_prefetch_advice(file);
}

#[cfg(not(target_os = "linux"))]
fn apply_prefetch_advice(_file: &File) {}

/// Pre-populate only a specific range of mmap'd pages into memory.
/// The pointer and length are page-aligned internally.
#[cfg(target_os = "linux")]
fn apply_mmap_populate_range(ptr: *const u8, len: usize) {
    linux::apply_mmap_populate_range(ptr, len);
}

/// Hint the kernel to prefetch a range of pages asynchronously (non-blocking).
/// Unlike MADV_POPULATE_READ, this returns immediately and lets the kernel
/// perform readahead in the background, avoiding mmap_lock contention.
#[cfg(target_os = "linux")]
fn apply_mmap_willneed_range(ptr: *const u8, len: usize) {
    linux::apply_mmap_willneed_range(ptr, len);
}

#[cfg(target_os = "linux")]
mod linux {
    use super::{env_var, ENV_FADVISE_MODE};
    use ::once_cell::sync::Lazy;
    use std::fs::File;
    use std::os::raw::{c_int, c_void};
    use std::os::unix::io::AsRawFd;

    const POSIX_FADV_RANDOM: c_int = 1;
    const POSIX_FADV_SEQUENTIAL: c_int = 2;
    const POSIX_FADV_WILLNEED: c_int = 3;
    const MADV_WILLNEED: c_int = 3;
    const MADV_POPULATE_READ: c_int = 22; // Linux 5.14+
    const PAGE_SIZE: usize = 4096;

    #[derive(Copy, Clone, Debug)]
    enum AdvicePattern {
        Random,
        Sequential,
    }

    static ADVICE_PATTERN: Lazy<AdvicePattern> = Lazy::new(|| {
        env_var(ENV_FADVISE_MODE)
            .as_deref()
            .map(|mode| mode.eq_ignore_ascii_case("sequential"))
            .map(|is_seq| {
                if is_seq {
                    AdvicePattern::Sequential
                } else {
                    AdvicePattern::Random
                }
            })
            .unwrap_or(AdvicePattern::Random)
    });

    extern "C" {
        fn posix_fadvise(fd: c_int, offset: i64, len: i64, advice: c_int) -> c_int;
        fn madvise(addr: *mut c_void, length: usize, advice: c_int) -> c_int;
    }

    pub(super) fn apply_prefetch_advice(file: &File) {
        let fd = file.as_raw_fd();
        if fd >= 0 {
            unsafe {
                let _ = posix_fadvise(fd, 0, 0, POSIX_FADV_WILLNEED);
                let pattern = match *ADVICE_PATTERN {
                    AdvicePattern::Sequential => POSIX_FADV_SEQUENTIAL,
                    AdvicePattern::Random => POSIX_FADV_RANDOM,
                };
                let _ = posix_fadvise(fd, 0, 0, pattern);
            }
        }
    }

    /// Pre-populate a specific range of mmap pages synchronously.
    /// Aligns the address down to page boundary and length up to page boundary.
    /// Uses MADV_POPULATE_READ (Linux 5.14+), falls back to MADV_WILLNEED.
    pub(super) fn apply_mmap_populate_range(ptr: *const u8, len: usize) {
        if len == 0 {
            return;
        }
        let addr = ptr as usize;
        let aligned_addr = addr & !(PAGE_SIZE - 1);
        let end = addr + len;
        let aligned_len = end - aligned_addr;

        unsafe {
            let ret = madvise(aligned_addr as *mut c_void, aligned_len, MADV_POPULATE_READ);
            if ret != 0 {
                let _ = madvise(aligned_addr as *mut c_void, aligned_len, MADV_WILLNEED);
            }
        }
    }

    /// Async hint: tell the kernel to start readahead for a range (non-blocking).
    pub(super) fn apply_mmap_willneed_range(ptr: *const u8, len: usize) {
        if len == 0 {
            return;
        }
        let addr = ptr as usize;
        let aligned_addr = addr & !(PAGE_SIZE - 1);
        let end = addr + len;
        let aligned_len = end - aligned_addr;

        unsafe {
            let _ = madvise(aligned_addr as *mut c_void, aligned_len, MADV_WILLNEED);
        }
    }
}
