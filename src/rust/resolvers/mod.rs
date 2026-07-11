use memmap2::Mmap;
/// Tensor file resolver traits and implementations for TensorDex
///
/// This module provides a modular and extensible system for loading and decoding
/// tensor data from various storage formats and sources.
use std::sync::Arc;

pub mod s3;
pub mod simple;

pub use s3::S3TensorResolver;
pub use simple::SimpleTensorResolver;

pub enum TensorResolver {
    Simple(SimpleTensorResolver),
    S3(S3TensorResolver),
}

enum TensorBacking {
    Mmap(Arc<Mmap>),
    Owned(Arc<[u8]>),
}

/// Zero-copy tensor slice that holds a reference to backing storage
pub struct TensorSlice {
    _backing: TensorBacking,
    data_ptr: *const u8,
    data_len: usize,
}

unsafe impl Send for TensorSlice {}
unsafe impl Sync for TensorSlice {}

impl TensorSlice {
    /// Create a new TensorSlice with the given mmap and data slice
    pub fn new(mmap: Arc<Mmap>, data: &[u8]) -> Self {
        Self {
            _backing: TensorBacking::Mmap(mmap),
            data_ptr: data.as_ptr(),
            data_len: data.len(),
        }
    }

    /// Create a tensor slice backed by owned heap memory
    pub fn from_owned_bytes(bytes: Arc<[u8]>, offset: usize, len: usize) -> Self {
        let data_ptr = unsafe { bytes.as_ptr().add(offset) };
        Self {
            _backing: TensorBacking::Owned(bytes),
            data_ptr,
            data_len: len,
        }
    }

    /// Get the tensor data as a byte slice
    pub fn data(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.data_ptr, self.data_len) }
    }

    /// Get the length of the tensor data
    pub fn len(&self) -> usize {
        self.data_len
    }

    /// Check if the tensor data is empty
    pub fn is_empty(&self) -> bool {
        self.data_len == 0
    }
}

/// A batch of zero-copy tensor slices
pub type TensorBatch = Vec<TensorSlice>;

/// Main trait for resolving and loading tensor data
///
/// This trait defines the interface for different tensor loading strategies,
/// allowing for pluggable implementations that can handle various data formats,
/// quantization schemes, and storage backends.
pub trait TensorFileResolver: Send + Sync {
    fn tensor_dir(&self) -> &str;

    fn resolve_tensor_path(&self, tensor_id: &str) -> Result<String, String> {
        if tensor_id.len() < 2 {
            return Err(format!("Invalid tensor id: {}", tensor_id));
        }

        let prefix1 = &tensor_id[..2];
        let prefix2 = if tensor_id.len() >= 4 {
            &tensor_id[2..4]
        } else {
            "00"
        };

        // Try 2-level sharding first (standard local layout)
        let path2 = format!(
            "{}/blobs/{}/{}/{}.safetensors",
            self.tensor_dir(),
            prefix1,
            prefix2,
            tensor_id
        );

        if std::path::Path::new(&path2).exists() {
            return Ok(path2);
        }

        // Fallback to 1-level sharding (standard S3 or migrated layout)
        let path1 = format!(
            "{}/blobs/{}/{}.safetensors",
            self.tensor_dir(),
            prefix1,
            tensor_id
        );

        Ok(path1)
    }

    fn bulk_load_tensors_mmap(
        &self,
        requests: &[(String, Vec<usize>)],
    ) -> Result<TensorBatch, String>;

    fn bulk_load_slices(
        &self,
        requests: &[(String, usize, usize, Vec<usize>)],
        dsts: &mut [u8],
    ) -> Result<(), String>;
}
