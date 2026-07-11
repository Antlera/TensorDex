use std::error::Error;
use std::fmt;

/// Generic compression result type
pub type CompressionResult<T> = Result<T, CompressionError>;

/// Compression error types
#[derive(Debug, Clone)]
pub enum CompressionError {
    CompressionFailed(String),
    DecompressionFailed(String),
    InvalidInput(String),
    ConfigurationError(String),
}

impl fmt::Display for CompressionError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            CompressionError::CompressionFailed(msg) => write!(f, "Compression failed: {}", msg),
            CompressionError::DecompressionFailed(msg) => {
                write!(f, "Decompression failed: {}", msg)
            }
            CompressionError::InvalidInput(msg) => write!(f, "Invalid input: {}", msg),
            CompressionError::ConfigurationError(msg) => write!(f, "Configuration error: {}", msg),
        }
    }
}

impl Error for CompressionError {}

/// Configuration for compression strategies
#[derive(Debug, Clone)]
pub struct CompressionConfig {
    pub compression_level: i32,
    pub num_threads: usize,
    pub chunk_size: usize,
    pub enable_checksum: bool,
    pub collect_entropy: bool,
    pub dictionary: Option<Vec<u8>>,
    pub dictionary_path: Option<String>,
}

impl Default for CompressionConfig {
    fn default() -> Self {
        Self {
            compression_level: 3,
            num_threads: num_cpus::get(),
            chunk_size: 64 * 1024, // 64KB chunks
            enable_checksum: false,
            collect_entropy: false,
            dictionary: None,
            dictionary_path: None,
        }
    }
}

/// Generic compressor trait that all compression strategies must implement
pub trait Compressor: Send + Sync {
    /// Return the name of the compression algorithm
    fn name(&self) -> &'static str;

    /// Compress the input data
    fn compress(&self, data: &[u8]) -> CompressionResult<Vec<u8>>;

    /// Decompress the input data
    fn decompress(&self, compressed_data: &[u8]) -> CompressionResult<Vec<u8>>;

    /// Get the current configuration
    fn config(&self) -> &CompressionConfig;

    /// Update the configuration
    fn update_config(&mut self, config: CompressionConfig) -> CompressionResult<()>;

    /// Get compression statistics if available
    fn get_stats(&self) -> Option<CompressionStats> {
        None
    }

    /// Check if the compressor supports streaming
    fn supports_streaming(&self) -> bool {
        false
    }
}

/// Compression statistics
#[derive(Debug, Clone)]
pub struct CompressionStats {
    pub original_size: usize,
    pub compressed_size: usize,
    pub compression_ratio: f64,
    pub compression_time_ms: u128,
    pub throughput_mb_per_sec: f64,
}

impl CompressionStats {
    pub fn new(original_size: usize, compressed_size: usize, compression_time_ms: u128) -> Self {
        let compression_ratio = if original_size > 0 {
            compressed_size as f64 / original_size as f64
        } else {
            0.0
        };

        let throughput_mb_per_sec = if compression_time_ms > 0 {
            (original_size as f64 / (1024.0 * 1024.0)) / (compression_time_ms as f64 / 1000.0)
        } else {
            0.0
        };

        Self {
            original_size,
            compressed_size,
            compression_ratio,
            compression_time_ms,
            throughput_mb_per_sec,
        }
    }
}

/// Factory for creating compressor instances
pub struct CompressorFactory;

impl CompressorFactory {
    pub fn create_compressor(
        algorithm: &str,
        config: CompressionConfig,
    ) -> CompressionResult<Box<dyn Compressor>> {
        match algorithm.to_lowercase().as_str() {
            "zstd" => Ok(Box::new(crate::codec::zstd::ZstdCompressor::new(config)?)),
            "lz4" => Ok(Box::new(crate::codec::lz4::Lz4Compressor::new(config)?)),
            _ => Err(CompressionError::ConfigurationError(format!(
                "Unsupported compression algorithm: {}. Supported algorithms: {}",
                algorithm,
                Self::available_algorithms().join(", ")
            ))),
        }
    }

    pub fn available_algorithms() -> Vec<&'static str> {
        vec!["zstd", "lz4"]
    }
}

pub mod experimental;
pub mod lz4;
/// Sub-modules for different compression implementations
pub mod zstd;
