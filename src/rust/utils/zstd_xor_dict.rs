/// XOR-based Dictionary Trainer for Zstd compression
///
/// This module provides functions to generate training samples from XOR data
/// and train Zstd dictionaries specifically for tensor compression workloads.
use crate::codec::{CompressionError, CompressionResult};
use crate::compression::plan::CompressionPlan;
use crate::kernels::xor::compute_xor_bytes_rust;
use crate::resolvers::TensorFileResolver;
use numpy::{IntoPyArray, PyArray1};
use pyo3::prelude::*;
use std::collections::HashMap;

/// Configuration for XOR-based dictionary training
#[derive(Debug, Clone)]
pub struct XorDictTrainingConfig {
    /// Maximum number of XOR samples to collect for training
    pub max_samples: usize,
    /// Maximum dictionary size in bytes
    pub dict_size: usize,
    /// Maximum size of each training sample in bytes
    pub max_sample_size: usize,
    /// Skip factor for sampling (1 = use all samples, 2 = use every 2nd, etc.)
    pub skip_factor: usize,
}

impl Default for XorDictTrainingConfig {
    fn default() -> Self {
        Self {
            max_samples: 1000,
            dict_size: 64 * 1024,      // 64KB dictionary
            max_sample_size: 4 * 1024, // 4KB per sample
            skip_factor: 1,
        }
    }
}

/// Generate training samples from XOR deltas using compression plans
///
/// This function reuses the same data iteration logic as compression_execution.rs
/// to ensure sample boundaries and distribution match production usage.
pub fn generate_xor_training_samples<T: TensorFileResolver>(
    plans: &[CompressionPlan],
    resolver: &T,
    config: &XorDictTrainingConfig,
) -> CompressionResult<Vec<Vec<u8>>> {
    let mut training_samples = Vec::new();
    let mut samples_collected = 0;

    Python::with_gil(|py| -> CompressionResult<()> {
        for (i, plan) in plans.iter().enumerate() {
            // Skip samples based on skip_factor
            if i % config.skip_factor != 0 {
                continue;
            }

            if samples_collected >= config.max_samples {
                break;
            }

            // Load target and base tensors using the same logic as compression_execution
            let target_request = (plan.target_tensor_id.clone(), plan.target_shape.clone());
            let base_request = (plan.base_tensor_id.clone(), plan.target_shape.clone());

            let target_batch = resolver
                .bulk_load_tensors_mmap(&[target_request])
                .map_err(|e| {
                    CompressionError::CompressionFailed(format!(
                        "Failed to load target tensor: {}",
                        e
                    ))
                })?;
            let base_batch = resolver
                .bulk_load_tensors_mmap(&[base_request])
                .map_err(|e| {
                    CompressionError::CompressionFailed(format!(
                        "Failed to load base tensor: {}",
                        e
                    ))
                })?;

            if target_batch.is_empty() || base_batch.is_empty() {
                continue;
            }

            let target_slice = &target_batch[0];
            let base_slice = &base_batch[0];

            // Convert byte slices for XOR computation - support multiple data types
            let element_size = plan
                .element_size()
                .map_err(|e| CompressionError::InvalidInput(e))?;

            let target_len = target_slice.len() / element_size;
            let base_len = base_slice.len() / element_size;

            if target_len != base_len || target_len == 0 {
                continue;
            }

            // Handle different tensor data types
            let xor_bytes = match element_size {
                4 => {
                    // f32 tensors
                    let target_f32: &[f32] = bytemuck::cast_slice(target_slice.data());
                    let base_f32: &[f32] = bytemuck::cast_slice(base_slice.data());

                    let target_array = PyArray1::from_slice(py, target_f32);
                    let base_array = PyArray1::from_slice(py, base_f32);

                    compute_xor_bytes_rust(py, target_array.readonly(), base_array.readonly())
                        .map_err(|e| {
                            CompressionError::CompressionFailed(format!(
                                "XOR computation failed: {}",
                                e
                            ))
                        })?
                }
                2 => {
                    // bfloat16 or f16 tensors - process as raw bytes for XOR training
                    // This gives us the raw bit patterns which is what we want for dictionary training
                    let target_bytes = target_slice.data();
                    let base_bytes = base_slice.data();

                    if target_bytes.len() != base_bytes.len() {
                        continue;
                    }

                    // Simple XOR at byte level for non-f32 data
                    target_bytes
                        .iter()
                        .zip(base_bytes.iter())
                        .map(|(a, b)| a ^ b)
                        .collect::<Vec<u8>>()
                }
                _ => {
                    // Skip unsupported data types
                    continue;
                }
            };

            // Limit sample size to prevent overly large training data
            let sample_size = std::cmp::min(xor_bytes.len(), config.max_sample_size);
            let mut sample = xor_bytes;
            sample.truncate(sample_size);

            if !sample.is_empty() {
                training_samples.push(sample);
                samples_collected += 1;
            }
        }
        Ok(())
    })?;

    if training_samples.is_empty() {
        return Err(CompressionError::InvalidInput(
            "No valid XOR training samples could be generated".to_string(),
        ));
    }

    Ok(training_samples)
}

/// Train a Zstd dictionary from XOR training samples
///
/// Uses zstd::dict::from_samples to create an optimized dictionary
/// for compressing XOR deltas.
pub fn train_zstd_dictionary(
    training_samples: &[Vec<u8>],
    dict_size: usize,
) -> CompressionResult<Vec<u8>> {
    if training_samples.is_empty() {
        return Err(CompressionError::InvalidInput(
            "Cannot train dictionary from empty samples".to_string(),
        ));
    }

    // Convert samples to the format expected by zstd::dict::from_samples
    let sample_refs: Vec<&[u8]> = training_samples.iter().map(|s| s.as_slice()).collect();

    // Train the dictionary
    let dictionary = zstd::dict::from_samples(&sample_refs, dict_size).map_err(|e| {
        CompressionError::CompressionFailed(format!("Dictionary training failed: {}", e))
    })?;

    Ok(dictionary)
}

/// Complete pipeline: generate XOR samples and train dictionary
///
/// This is a convenience function that combines sample generation and dictionary training.
pub fn train_xor_dictionary<T: TensorFileResolver>(
    plans: &[CompressionPlan],
    resolver: &T,
    config: Option<XorDictTrainingConfig>,
) -> CompressionResult<Vec<u8>> {
    let config = config.unwrap_or_default();

    // Generate training samples from XOR deltas
    let training_samples = generate_xor_training_samples(plans, resolver, &config)?;

    // Train dictionary from samples
    train_zstd_dictionary(&training_samples, config.dict_size)
}

/// Analyze XOR sample characteristics for dictionary training optimization
///
/// Returns statistics about the training samples to help optimize dictionary parameters.
pub fn analyze_xor_samples(samples: &[Vec<u8>]) -> HashMap<String, f64> {
    let mut stats = HashMap::new();

    if samples.is_empty() {
        return stats;
    }

    let total_samples = samples.len() as f64;
    let total_bytes: usize = samples.iter().map(|s| s.len()).sum();
    let avg_size = total_bytes as f64 / total_samples;

    let min_size = samples.iter().map(|s| s.len()).min().unwrap_or(0) as f64;
    let max_size = samples.iter().map(|s| s.len()).max().unwrap_or(0) as f64;

    // Calculate zero byte ratio across all samples
    let total_zero_bytes: usize = samples
        .iter()
        .map(|sample| sample.iter().filter(|&&b| b == 0).count())
        .sum();
    let zero_ratio = total_zero_bytes as f64 / total_bytes as f64;

    stats.insert("total_samples".to_string(), total_samples);
    stats.insert("total_bytes".to_string(), total_bytes as f64);
    stats.insert("avg_sample_size".to_string(), avg_size);
    stats.insert("min_sample_size".to_string(), min_size);
    stats.insert("max_sample_size".to_string(), max_size);
    stats.insert("zero_byte_ratio".to_string(), zero_ratio);

    stats
}

/// Python wrapper for XOR dictionary training
#[pyfunction]
pub fn train_xor_dictionary_rust(
    py: Python,
    plans_json: &[u8],
    tensor_dir: &str,
    max_samples: usize,
    dict_size: usize,
    max_sample_size: usize,
    skip_factor: usize,
) -> PyResult<PyObject> {
    use crate::resolvers::SimpleTensorResolver;
    use serde_json;

    // Deserialize plans from JSON
    let plans: Vec<CompressionPlan> = serde_json::from_slice(plans_json).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "Failed to deserialize plans: {}",
            e
        ))
    })?;

    // Create tensor resolver
    let resolver = SimpleTensorResolver::new(tensor_dir.to_string());

    // Create configuration
    let config = XorDictTrainingConfig {
        max_samples,
        dict_size,
        max_sample_size,
        skip_factor,
    };

    // Train dictionary
    let dictionary = train_xor_dictionary(&plans, &resolver, Some(config)).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
            "Dictionary training failed: {}",
            e
        ))
    })?;

    // Convert to Python bytes
    Ok(pyo3::types::PyBytes::new(py, &dictionary).to_object(py))
}

/// Simple Python API for XOR dictionary training that mirrors compression example inputs
#[pyfunction]
pub fn train_zstd_xor_dictionary(
    py: Python,
    plans_json: &[u8],
    tensor_dir: &str,
    dict_size: Option<usize>,
    max_samples: Option<usize>,
    max_sample_bytes: Option<usize>,
    seed: Option<u64>,
    num_threads: Option<usize>,
    save_path: Option<&str>,
) -> PyResult<PyObject> {
    use crate::resolvers::SimpleTensorResolver;
    use serde_json;
    use std::fs;

    // Set defaults
    let dict_size = dict_size.unwrap_or(112_640);
    let max_samples = max_samples.unwrap_or(1000);
    let max_sample_bytes = max_sample_bytes.unwrap_or(4 * 1024);
    let skip_factor = 1; // Always use all available samples up to max_samples

    // Deserialize plans from JSON
    let plans: Vec<CompressionPlan> = serde_json::from_slice(plans_json).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "Failed to deserialize plans: {}",
            e
        ))
    })?;

    // Create tensor resolver
    let resolver = SimpleTensorResolver::new(tensor_dir.to_string());

    // Create configuration
    let config = XorDictTrainingConfig {
        max_samples,
        dict_size,
        max_sample_size: max_sample_bytes,
        skip_factor,
    };

    // Train dictionary
    let dictionary = train_xor_dictionary(&plans, &resolver, Some(config)).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
            "Dictionary training failed: {}",
            e
        ))
    })?;

    // Save to file if path provided
    if let Some(path) = save_path {
        fs::write(path, &dictionary).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyIOError, _>(format!(
                "Failed to save dictionary to {}: {}",
                path, e
            ))
        })?;
    }

    // Convert to Python bytes
    Ok(pyo3::types::PyBytes::new(py, &dictionary).to_object(py))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_xor_dict_training_config() {
        let config = XorDictTrainingConfig::default();
        assert_eq!(config.max_samples, 1000);
        assert_eq!(config.dict_size, 64 * 1024);
        assert_eq!(config.max_sample_size, 4 * 1024);
        assert_eq!(config.skip_factor, 1);
    }

    #[test]
    fn test_train_zstd_dictionary() {
        // Create mock training samples with repetitive patterns
        let samples = vec![
            b"repeating data pattern for compression testing".to_vec(),
            b"repeating data pattern with slight variations".to_vec(),
            b"another repeating data pattern for testing".to_vec(),
            b"another repeating data pattern with changes".to_vec(),
        ];

        let dict_size = 1024;
        let dictionary = train_zstd_dictionary(&samples, dict_size).unwrap();

        // Dictionary should be created successfully
        assert!(!dictionary.is_empty());
        assert!(dictionary.len() <= dict_size + 100); // Allow some overhead
    }

    #[test]
    fn test_analyze_xor_samples() {
        let samples = vec![
            vec![0, 1, 2, 3, 0, 0],
            vec![1, 0, 3, 4, 5],
            vec![0, 0, 0, 1],
        ];

        let stats = analyze_xor_samples(&samples);

        assert_eq!(stats["total_samples"], 3.0);
        assert_eq!(stats["total_bytes"], 15.0);
        assert_eq!(stats["avg_sample_size"], 5.0);
        assert_eq!(stats["min_sample_size"], 4.0);
        assert_eq!(stats["max_sample_size"], 6.0);

        // 5 zero bytes out of 15 total = 1/3
        assert!((stats["zero_byte_ratio"] - (5.0 / 15.0)).abs() < 0.001);
    }

    #[test]
    fn test_empty_samples() {
        let empty_samples: Vec<Vec<u8>> = vec![];
        let result = train_zstd_dictionary(&empty_samples, 1024);
        assert!(result.is_err());

        let stats = analyze_xor_samples(&empty_samples);
        assert!(stats.is_empty());
    }
}
