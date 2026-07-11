use super::{CompressionConfig, CompressionError, CompressionResult, CompressionStats, Compressor};
use lz4_flex::compress_prepend_size;
use lz4_flex::decompress_size_prepended;

/// LZ4 compression implementation
pub struct Lz4Compressor {
    config: CompressionConfig,
}

impl Lz4Compressor {
    /// Create a new LZ4 compressor with the given configuration
    pub fn new(config: CompressionConfig) -> CompressionResult<Self> {
        // Validate configuration
        if config.compression_level < 1 || config.compression_level > 12 {
            return Err(CompressionError::ConfigurationError(
                "LZ4 compression level must be between 1 and 12".to_string(),
            ));
        }

        if config.num_threads == 0 {
            return Err(CompressionError::ConfigurationError(
                "Number of threads must be greater than 0".to_string(),
            ));
        }

        Ok(Self { config })
    }

    /// Create a new LZ4 compressor with default configuration
    pub fn with_level(compression_level: i32) -> CompressionResult<Self> {
        let mut config = CompressionConfig::default();
        config.compression_level = compression_level;
        Self::new(config)
    }

    /// Create a new LZ4 compressor with custom thread count
    pub fn with_threads(compression_level: i32, num_threads: usize) -> CompressionResult<Self> {
        let mut config = CompressionConfig::default();
        config.compression_level = compression_level;
        config.num_threads = num_threads;
        Self::new(config)
    }
}

impl Compressor for Lz4Compressor {
    fn name(&self) -> &'static str {
        "lz4"
    }

    fn compress(&self, data: &[u8]) -> CompressionResult<Vec<u8>> {
        let start_time = std::time::Instant::now();

        // LZ4 compression using lz4_flex
        // Note: lz4_flex doesn't support compression levels in the same way as zstd
        // We use compress_prepend_size which includes the original size for decompression
        let compressed_data = compress_prepend_size(data);

        let compression_time = start_time.elapsed().as_millis();

        // Note: Statistics are not stored in the compressor for thread safety
        // They can be calculated externally if needed

        Ok(compressed_data)
    }

    fn decompress(&self, compressed_data: &[u8]) -> CompressionResult<Vec<u8>> {
        decompress_size_prepended(compressed_data).map_err(|e| {
            CompressionError::DecompressionFailed(format!("Failed to decompress: {}", e))
        })
    }

    fn config(&self) -> &CompressionConfig {
        &self.config
    }

    fn update_config(&mut self, config: CompressionConfig) -> CompressionResult<()> {
        // Validate new configuration
        if config.compression_level < 1 || config.compression_level > 12 {
            return Err(CompressionError::ConfigurationError(
                "LZ4 compression level must be between 1 and 12".to_string(),
            ));
        }

        if config.num_threads == 0 {
            return Err(CompressionError::ConfigurationError(
                "Number of threads must be greater than 0".to_string(),
            ));
        }

        self.config = config;
        Ok(())
    }

    fn get_stats(&self) -> Option<CompressionStats> {
        None // Stats not stored for thread safety
    }

    fn supports_streaming(&self) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lz4_compression() {
        let config = CompressionConfig::default();
        let compressor = Lz4Compressor::new(config).unwrap();

        let test_data = b"Hello, world! This is a test string for compression.";
        let compressed = compressor.compress(test_data).unwrap();
        let decompressed = compressor.decompress(&compressed).unwrap();

        assert_eq!(test_data, decompressed.as_slice());
        assert!(compressed.len() < test_data.len());
    }

    #[test]
    fn test_lz4_config_validation() {
        let mut config = CompressionConfig::default();
        config.compression_level = 15; // Invalid level

        let result = Lz4Compressor::new(config);
        assert!(result.is_err());

        let mut config = CompressionConfig::default();
        config.num_threads = 0; // Invalid thread count

        let result = Lz4Compressor::new(config);
        assert!(result.is_err());
    }

    #[test]
    fn test_lz4_stats() {
        let config = CompressionConfig::default();
        let compressor = Lz4Compressor::new(config).unwrap();

        let test_data = b"Hello, world! This is a test string for compression.";
        let compressed = compressor.compress(test_data).unwrap();

        // Stats are not stored in the compressor for thread safety
        // But we can verify the compression worked
        assert!(compressed.len() > 0);
        assert!(compressor.get_stats().is_none());
    }

    #[test]
    fn test_lz4_with_level() {
        let compressor = Lz4Compressor::with_level(6).unwrap();
        assert_eq!(compressor.config().compression_level, 6);

        let test_data = b"This is test data for compression level testing.";
        let compressed = compressor.compress(test_data).unwrap();
        let decompressed = compressor.decompress(&compressed).unwrap();

        assert_eq!(test_data, decompressed.as_slice());
    }

    #[test]
    fn test_lz4_with_threads() {
        let compressor = Lz4Compressor::with_threads(4, 2).unwrap();
        assert_eq!(compressor.config().compression_level, 4);
        assert_eq!(compressor.config().num_threads, 2);

        let test_data = b"This is test data for thread configuration testing.";
        let compressed = compressor.compress(test_data).unwrap();
        let decompressed = compressor.decompress(&compressed).unwrap();

        assert_eq!(test_data, decompressed.as_slice());
    }

    #[test]
    fn test_lz4_update_config() {
        let config = CompressionConfig::default();
        let mut compressor = Lz4Compressor::new(config).unwrap();

        let mut new_config = CompressionConfig::default();
        new_config.compression_level = 8;
        new_config.num_threads = 4;

        compressor.update_config(new_config).unwrap();
        assert_eq!(compressor.config().compression_level, 8);
        assert_eq!(compressor.config().num_threads, 4);

        // Test invalid config update
        let mut invalid_config = CompressionConfig::default();
        invalid_config.compression_level = 20; // Invalid

        let result = compressor.update_config(invalid_config);
        assert!(result.is_err());
    }

    #[test]
    fn test_lz4_supports_streaming() {
        let config = CompressionConfig::default();
        let compressor = Lz4Compressor::new(config).unwrap();

        assert_eq!(compressor.supports_streaming(), false);
    }

    #[test]
    fn test_lz4_name() {
        let config = CompressionConfig::default();
        let compressor = Lz4Compressor::new(config).unwrap();

        assert_eq!(compressor.name(), "lz4");
    }
}
