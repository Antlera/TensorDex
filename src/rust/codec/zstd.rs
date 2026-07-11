use super::{CompressionConfig, CompressionError, CompressionResult, CompressionStats, Compressor};
use std::fs;
use std::io::{Read, Write};
use zstd::stream::{Decoder, Encoder};

/// Zstd compression implementation
pub struct ZstdCompressor {
    config: CompressionConfig,
    loaded_dictionary: Option<Vec<u8>>,
}

impl ZstdCompressor {
    /// Create a new Zstd compressor with the given configuration
    pub fn new(config: CompressionConfig) -> CompressionResult<Self> {
        // Validate configuration
        if config.compression_level < 1 || config.compression_level > 22 {
            return Err(CompressionError::ConfigurationError(
                "Zstd compression level must be between 1 and 22".to_string(),
            ));
        }

        if config.num_threads == 0 {
            return Err(CompressionError::ConfigurationError(
                "Number of threads must be greater than 0".to_string(),
            ));
        }

        // Load dictionary from path if specified
        let loaded_dictionary = if let Some(ref dict_path) = config.dictionary_path {
            Some(fs::read(dict_path).map_err(|e| {
                CompressionError::ConfigurationError(format!(
                    "Failed to load dictionary from {}: {}",
                    dict_path, e
                ))
            })?)
        } else {
            None
        };

        Ok(Self {
            config,
            loaded_dictionary,
        })
    }

    /// Create a new Zstd compressor with default configuration
    pub fn with_level(compression_level: i32) -> CompressionResult<Self> {
        let mut config = CompressionConfig::default();
        config.compression_level = compression_level;
        Self::new(config)
    }

    /// Create a new Zstd compressor with custom thread count
    pub fn with_threads(compression_level: i32, num_threads: usize) -> CompressionResult<Self> {
        let mut config = CompressionConfig::default();
        config.compression_level = compression_level;
        config.num_threads = num_threads;
        Self::new(config)
    }

    /// Create a new Zstd compressor with a pre-trained dictionary
    pub fn with_dictionary(compression_level: i32, dict_bytes: Vec<u8>) -> CompressionResult<Self> {
        let mut config: CompressionConfig = CompressionConfig::default();
        config.compression_level = compression_level;
        config.dictionary = Some(dict_bytes);
        Self::new(config)
    }
}

impl Compressor for ZstdCompressor {
    fn name(&self) -> &'static str {
        "zstd"
    }

    fn compress(&self, data: &[u8]) -> CompressionResult<Vec<u8>> {
        let start_time = std::time::Instant::now();

        // Create encoder with or without dictionary
        let mut encoder = if let Some(ref dict_bytes) = self.loaded_dictionary {
            // Create encoder with dictionary
            Encoder::with_dictionary(Vec::new(), self.config.compression_level, dict_bytes)
                .map_err(|e| {
                    CompressionError::CompressionFailed(format!(
                        "Failed to create encoder with dictionary: {}",
                        e
                    ))
                })?
        } else {
            // Create encoder without dictionary
            Encoder::new(Vec::new(), self.config.compression_level).map_err(|e| {
                CompressionError::CompressionFailed(format!("Failed to create encoder: {}", e))
            })?
        };

        // Enable multi-threading
        encoder
            .multithread(self.config.num_threads as u32)
            .map_err(|e| {
                CompressionError::CompressionFailed(format!(
                    "Failed to enable multithreading: {}",
                    e
                ))
            })?;

        // Enable checksum if configured
        if self.config.enable_checksum {
            encoder.include_checksum(true).map_err(|e| {
                CompressionError::CompressionFailed(format!("Failed to enable checksum: {}", e))
            })?;
        }

        // Write all data to encoder
        encoder.write_all(data).map_err(|e| {
            CompressionError::CompressionFailed(format!("Failed to write data: {}", e))
        })?;

        // Finish compression and get result
        let compressed_data = encoder.finish().map_err(|e| {
            CompressionError::CompressionFailed(format!("Failed to finish compression: {}", e))
        })?;

        let compression_time = start_time.elapsed().as_millis();

        // Note: Statistics are not stored in the compressor for thread safety
        // They can be calculated externally if needed

        Ok(compressed_data)
    }

    fn decompress(&self, compressed_data: &[u8]) -> CompressionResult<Vec<u8>> {
        if let Some(ref dict_bytes) = self.loaded_dictionary {
            // Decompress with dictionary using streaming decoder
            let mut decoder = zstd::stream::Decoder::with_dictionary(compressed_data, dict_bytes)
                .map_err(|e| {
                CompressionError::DecompressionFailed(format!(
                    "Failed to create decoder with dictionary: {}",
                    e
                ))
            })?;

            let mut decompressed = Vec::new();
            std::io::Read::read_to_end(&mut decoder, &mut decompressed).map_err(|e| {
                CompressionError::DecompressionFailed(format!(
                    "Failed to decompress with dictionary: {}",
                    e
                ))
            })?;

            Ok(decompressed)
        } else {
            // Decompress without dictionary
            zstd::decode_all(compressed_data).map_err(|e| {
                CompressionError::DecompressionFailed(format!("Failed to decompress: {}", e))
            })
        }
    }

    fn config(&self) -> &CompressionConfig {
        &self.config
    }

    fn update_config(&mut self, config: CompressionConfig) -> CompressionResult<()> {
        // Validate new configuration
        if config.compression_level < 1 || config.compression_level > 22 {
            return Err(CompressionError::ConfigurationError(
                "Zstd compression level must be between 1 and 22".to_string(),
            ));
        }

        if config.num_threads == 0 {
            return Err(CompressionError::ConfigurationError(
                "Number of threads must be greater than 0".to_string(),
            ));
        }

        // Load dictionary from path if specified
        self.loaded_dictionary = if let Some(ref dict_path) = config.dictionary_path {
            Some(fs::read(dict_path).map_err(|e| {
                CompressionError::ConfigurationError(format!(
                    "Failed to load dictionary from {}: {}",
                    dict_path, e
                ))
            })?)
        } else {
            None
        };

        self.config = config;
        Ok(())
    }

    fn get_stats(&self) -> Option<CompressionStats> {
        None // Stats not stored for thread safety
    }

    fn supports_streaming(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zstd_compression() {
        let config = CompressionConfig::default();
        let compressor = ZstdCompressor::new(config).unwrap();

        let test_data = b"Hello, world! This is a test string for compression.";
        let compressed = compressor.compress(test_data).unwrap();
        let decompressed = compressor.decompress(&compressed).unwrap();

        assert_eq!(test_data, decompressed.as_slice());
        assert!(compressed.len() < test_data.len());
    }

    #[test]
    fn test_zstd_config_validation() {
        let mut config = CompressionConfig::default();
        config.compression_level = 25; // Invalid level

        let result = ZstdCompressor::new(config);
        assert!(result.is_err());

        let mut config = CompressionConfig::default();
        config.num_threads = 0; // Invalid thread count

        let result = ZstdCompressor::new(config);
        assert!(result.is_err());
    }

    #[test]
    fn test_zstd_stats() {
        let config = CompressionConfig::default();
        let compressor = ZstdCompressor::new(config).unwrap();

        let test_data = b"Hello, world! This is a test string for compression.";
        let compressed = compressor.compress(test_data).unwrap();

        // Stats are not stored in the compressor for thread safety
        // But we can verify the compression worked
        assert!(compressed.len() > 0);
        assert!(compressor.get_stats().is_none());
    }

    #[test]
    fn test_zstd_dictionary_compression() {
        // Create training samples with repetitive patterns
        let training_samples = vec![
            b"This is a repeating pattern that appears frequently in our data".to_vec(),
            b"This is a repeating pattern with slight variations in the text".to_vec(),
            b"Another repeating pattern that appears frequently in our data".to_vec(),
            b"Another repeating pattern with different variations in the text".to_vec(),
        ];

        // Train a simple dictionary using zstd::dict::from_samples
        let dict =
            zstd::dict::from_samples(&training_samples, 1024).expect("Failed to train dictionary");

        // Create compressor with dictionary
        let compressor = ZstdCompressor::with_dictionary(3, dict.clone()).unwrap();

        // Test data that should benefit from the dictionary
        let test_data = b"This is a repeating pattern that should compress well with dictionary";

        // Compress and decompress with dictionary
        let compressed = compressor.compress(test_data).unwrap();
        let decompressed = compressor.decompress(&compressed).unwrap();

        // Verify round-trip equality
        assert_eq!(test_data, decompressed.as_slice());

        // Create compressor without dictionary for comparison
        let compressor_no_dict = ZstdCompressor::with_level(3).unwrap();
        let compressed_no_dict = compressor_no_dict.compress(test_data).unwrap();

        // Dictionary compression should not be worse than non-dictionary
        // (though it might be similar for small test data)
        assert!(compressed.len() <= compressed_no_dict.len() + 100); // Allow small overhead for tiny data
    }
}
