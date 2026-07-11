//! Memory-mapped safetensors reader.
//!
//! A single `OpenSafetensors` holds a mmap + absolute-offset descriptors.
//! Slices handed out via `tensor_bytes` are zero-copy views into the mmap,
//! so ingest's hash / BCS / blob-write phases never allocate tensor payloads.

use std::collections::HashSet;
use std::fs::File;
use std::path::Path;

use anyhow::{anyhow, Context, Result};
use memmap2::Mmap;
use safetensors::{Dtype, SafeTensors};

/// Produce the ``torch.<name>`` string that Python's `str(t.dtype)` emits.
/// The Python ingest stored these verbatim; Rust ingest must match exactly.
pub fn dtype_to_torch_str(dt: Dtype) -> &'static str {
    match dt {
        Dtype::BOOL => "torch.bool",
        Dtype::U8 => "torch.uint8",
        Dtype::I8 => "torch.int8",
        Dtype::I16 => "torch.int16",
        Dtype::U16 => "torch.uint16",
        Dtype::I32 => "torch.int32",
        Dtype::U32 => "torch.uint32",
        Dtype::I64 => "torch.int64",
        Dtype::U64 => "torch.uint64",
        Dtype::F16 => "torch.float16",
        Dtype::BF16 => "torch.bfloat16",
        Dtype::F32 => "torch.float32",
        Dtype::F64 => "torch.float64",
        _ => "torch.unknown",
    }
}

/// Match Python's `json.dumps(list(map(int, shape)), separators=(",", ":"))`.
pub fn shape_to_json(shape: &[usize]) -> String {
    let mut s = String::from("[");
    for (i, d) in shape.iter().enumerate() {
        if i > 0 {
            s.push(',');
        }
        s.push_str(&d.to_string());
    }
    s.push(']');
    s
}

/// Per-tensor descriptor within a mmap'd safetensors file.
pub struct TensorSlot {
    pub name: String,
    pub shape: Vec<usize>,
    pub dtype: Dtype,
    pub data_offset: usize,
    pub data_len: usize,
}

pub struct OpenSafetensors {
    _file: File,
    mmap: Mmap,
    pub slots: Vec<TensorSlot>,
}

impl OpenSafetensors {
    pub fn open(path: &Path) -> Result<Self> {
        let file = File::open(path).with_context(|| format!("open safetensors file {:?}", path))?;
        let mmap = unsafe { Mmap::map(&file) }
            .with_context(|| format!("mmap safetensors file {:?}", path))?;
        let st = SafeTensors::deserialize(&mmap)
            .map_err(|e| anyhow!("parse safetensors {:?}: {}", path, e))?;

        let mmap_base = mmap.as_ptr() as usize;
        let slots: Vec<TensorSlot> = st
            .tensors()
            .into_iter()
            .map(|(name, view)| {
                let data = view.data();
                let data_offset = data.as_ptr() as usize - mmap_base;
                TensorSlot {
                    name: name.to_string(),
                    shape: view.shape().to_vec(),
                    dtype: view.dtype(),
                    data_offset,
                    data_len: data.len(),
                }
            })
            .collect();

        Ok(Self {
            _file: file,
            mmap,
            slots,
        })
    }

    /// Borrow a tensor's payload by its index in ``slots``.
    #[inline]
    pub fn slot_bytes(&self, slot: &TensorSlot) -> &[u8] {
        &self.mmap[slot.data_offset..slot.data_offset + slot.data_len]
    }

    /// Filter slots by an optional name set (kept in insertion order).
    pub fn filtered_slots<'a>(
        &'a self,
        filter: Option<&'a HashSet<String>>,
    ) -> Vec<&'a TensorSlot> {
        self.slots
            .iter()
            .filter(|s| filter.map_or(true, |f| f.contains(&s.name)))
            .collect()
    }
}
