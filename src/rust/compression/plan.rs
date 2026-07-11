use serde::{Deserialize, Serialize};

/// Rust equivalent of Python's CompressionPlan dataclass
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionPlan {
    pub param_name: String,
    pub target_tensor_id: String,
    pub base_tensor_id: String,
    pub target_shape: Vec<usize>,
    pub target_dtype: String,
    pub similarity_score: Option<f64>,
    pub cluster_id: Option<i32>,
    // Execution metadata (filled during compression)
    pub diff_size: Option<usize>,
    pub offset: Option<usize>,
    pub original_size: Option<usize>,
    pub element_size: Option<usize>,
}

impl CompressionPlan {
    pub fn element_size(&self) -> Result<usize, String> {
        self.element_size
            .or_else(|| match self.target_dtype.as_str() {
                "float32" | "f32" | "torch.float32" | "int32" | "i32" | "torch.int32" | "F32"
                | "I32" => Some(4),
                "float64" | "f64" | "torch.float64" | "int64" | "i64" | "torch.int64" | "F64"
                | "I64" => Some(8),
                "float16" | "f16" | "torch.float16" | "bfloat16" | "bf16" | "torch.bfloat16"
                | "F16" | "BF16" => Some(2),
                "uint8" | "u8" | "torch.uint8" | "int8" | "i8" | "torch.int8" | "U8" | "I8" => {
                    Some(1)
                }
                _ => None,
            })
            .ok_or_else(|| format!("Unknown element size for dtype: {}", self.target_dtype))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LegacyPairReport {
    pub target_id: String,
    pub base_id: String,
    pub bytes_in: u64,
    pub bytes_out: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PairReport {
    pub target_tensor_id: String,
    pub base_tensor_id: String,
    pub bytes_out: u64,
}

pub fn per_pair_records_to_json(records: &[PairReport]) -> Option<String> {
    serde_json::to_string(records).ok()
}
