use std::sync::Arc;

use aws_config::BehaviorVersion;
use aws_sdk_s3::{config::Region, Client};
use safetensors::tensor::{Dtype, TensorView};
use safetensors::SafeTensors;
use tokio::io::AsyncReadExt;
use tokio::runtime::Runtime;

use super::simple::select_tensor_key;
use super::{TensorBatch, TensorFileResolver, TensorSlice};

/// Tensor resolver that downloads whole safetensors files from S3.
pub struct S3TensorResolver {
    client: Client,
    bucket: String,
    prefix: String,
    runtime: Arc<Runtime>,
}

impl S3TensorResolver {
    /// Create a new resolver backed by the given bucket/prefix.
    pub fn new(bucket: String, prefix: String, region: Option<String>) -> Result<Self, String> {
        let normalized_prefix = prefix.trim_matches('/').to_string();
        let runtime =
            Runtime::new().map_err(|e| format!("Failed to create tokio runtime: {}", e))?;
        let config = runtime.block_on(async {
            let mut loader = aws_config::defaults(BehaviorVersion::latest());
            if let Some(region_name) = region {
                loader = loader.region(Region::new(region_name));
            }
            loader.load().await
        });

        // Use the shared runtime to create the client if necessary, or just rely on it being available
        // for subsequent calls. Note that `Client::new` is synchronous but the underlying
        // hyper client might be bound to the runtime context if not careful.
        // However, standard usage suggests `Client` is `Send + Sync` and decoupled.
        // The issue likely was creating NEW runtimes that didn't know about the client's state or vice versa.
        // By using ONE runtime for everything, we align them.

        Ok(Self {
            client: Client::new(&config),
            bucket,
            prefix: normalized_prefix,
            runtime: Arc::new(runtime),
        })
    }

    fn build_key(&self, tensor_id: &str) -> Result<String, String> {
        if tensor_id.len() < 2 {
            return Err(format!("Invalid tensor id: {}", tensor_id));
        }

        let shard = &tensor_id[..2];

        let mut key = String::new();
        if !self.prefix.is_empty() {
            key.push_str(&self.prefix);
            key.push('/');
        }
        key.push_str("blobs/");
        key.push_str(shard);
        key.push('/');
        key.push_str(tensor_id);
        key.push_str(".safetensors");
        Ok(key)
    }

    async fn download_object(&self, key: &str) -> Result<Vec<u8>, String> {
        let response = self
            .client
            .get_object()
            .bucket(&self.bucket)
            .key(key)
            .send()
            .await
            .map_err(|e| format!("Failed to GET s3://{}/{}: {:?}", self.bucket, key, e))?;

        let mut data = Vec::new();
        let mut reader = response.body.into_async_read();
        reader
            .read_to_end(&mut data)
            .await
            .map_err(|e| format!("Failed to read object body for {}: {}", key, e))?;
        Ok(data)
    }

    fn load_tensor_bytes<'a>(
        &self,
        safetensors: &'a SafeTensors<'a>,
        tensor_id: &str,
        expected_shape: &[usize],
    ) -> Result<TensorView<'a>, String> {
        let tensor_key = select_tensor_key(safetensors, tensor_id, expected_shape)?;
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

        Ok(tensor_view)
    }

    fn dtype_size(dtype: Dtype) -> usize {
        dtype.size()
    }
}

impl TensorFileResolver for S3TensorResolver {
    fn tensor_dir(&self) -> &str {
        &self.prefix
    }

    fn resolve_tensor_path(&self, tensor_id: &str) -> Result<String, String> {
        self.build_key(tensor_id)
    }

    fn bulk_load_tensors_mmap(
        &self,
        requests: &[(String, Vec<usize>)],
    ) -> Result<TensorBatch, String> {
        if requests.is_empty() {
            return Ok(Vec::new());
        }

        let mut batch = Vec::with_capacity(requests.len());

        for (tensor_id, shape) in requests {
            let key = self.build_key(tensor_id)?;
            let object_bytes = self
                .runtime
                .block_on(self.download_object(&key))
                .map_err(|e| format!("Failed to download {}: {}", tensor_id, e))?;
            let arc_bytes: Arc<[u8]> = object_bytes.into_boxed_slice().into();
            let (offset, len) = {
                let safetensors = SafeTensors::deserialize(arc_bytes.as_ref())
                    .map_err(|e| format!("Failed to deserialize {}: {}", tensor_id, e))?;
                let tensor_view = self.load_tensor_bytes(&safetensors, tensor_id, shape)?;
                let data = tensor_view.data();
                let base_ptr = arc_bytes.as_ptr();
                let data_ptr = data.as_ptr();
                let rel = unsafe { data_ptr.offset_from(base_ptr) };
                let offset = usize::try_from(rel).map_err(|_| {
                    format!(
                        "Tensor slice for {} not contained in downloaded object",
                        tensor_id
                    )
                })?;
                (offset, data.len())
            };

            batch.push(TensorSlice::from_owned_bytes(arc_bytes, offset, len));
        }

        Ok(batch)
    }

    fn bulk_load_slices(
        &self,
        requests: &[(String, usize, usize, Vec<usize>)],
        dsts: &mut [u8],
    ) -> Result<(), String> {
        if requests.is_empty() {
            return Ok(());
        }

        let base_ptr_addr = dsts.as_mut_ptr() as usize;
        let dst_len = dsts.len();

        for (tensor_id, offset, elem_size, shape) in requests {
            let key = self.build_key(tensor_id)?;
            let object_bytes = self
                .runtime
                .block_on(self.download_object(&key))
                .map_err(|e| format!("Failed to download {}: {}", tensor_id, e))?;
            let safetensors = SafeTensors::deserialize(&object_bytes)
                .map_err(|e| format!("Failed to deserialize {}: {}", tensor_id, e))?;
            let tensor_view = self.load_tensor_bytes(&safetensors, tensor_id, shape)?;
            let tensor_data = tensor_view.data();

            let dtype_size = Self::dtype_size(tensor_view.dtype());
            if dtype_size != *elem_size {
                return Err(format!(
                    "Element size mismatch for {}: expected {} bytes, file reports {} bytes",
                    tensor_id, elem_size, dtype_size
                ));
            }

            if !shape.is_empty() {
                let expected_len = shape.iter().product::<usize>().saturating_mul(*elem_size);
                if expected_len != tensor_data.len() {
                    return Err(format!(
                        "Size mismatch for {}: expected {} bytes, got {} bytes",
                        tensor_id,
                        expected_len,
                        tensor_data.len()
                    ));
                }
            }

            let byte_offset = *offset;
            let byte_len = tensor_data.len();
            if byte_offset + byte_len > dst_len {
                return Err(format!(
                    "Offset {} + length {} exceeds buffer size {}",
                    byte_offset, byte_len, dst_len
                ));
            }

            let base_ptr = base_ptr_addr as *mut u8;
            let dst =
                unsafe { std::slice::from_raw_parts_mut(base_ptr.add(byte_offset), byte_len) };
            dst.copy_from_slice(tensor_data);
        }

        Ok(())
    }
}
