//! Content-addressed tensor IDs.
//!
//! The id is the 128-bit XXH3 hash of the raw tensor bytes, hex-encoded
//! (32 chars, big-endian). This is byte-for-byte identical to Python's
//! `xxhash.xxh128_hexdigest` (seed 0) — the hash the metadata store and the
//! published `results.db` cache key on — so a freshly ingested tensor lands
//! on the exact same id as the paper's evaluation trace. XXH3 is not
//! cryptographic, but for a model hub deduplicating its own content it is the
//! right trade-off: an order of magnitude faster than BLAKE3 and identity-
//! preserving against the existing corpus.

use pyo3::prelude::*;
use xxhash_rust::xxh3::xxh3_128;

/// Hash raw tensor bytes to a 32-char hex content id (XXH3-128, big-endian).
///
/// Matches `xxhash.xxh128_hexdigest(bytes)`: the 128-bit value is emitted as
/// `(high64 << 64) | low64` in lowercase hex, zero-padded to 32 chars.
#[inline]
pub fn content_hash_hex(bytes: &[u8]) -> String {
    format!("{:032x}", xxh3_128(bytes))
}

/// Python-exposed twin of [`content_hash_hex`] — lets the engine verify a
/// tensor's bytes against its content-addressed id on read, using the exact
/// hash ingest assigned.
#[pyfunction]
pub fn content_hash(bytes: &[u8]) -> String {
    content_hash_hex(bytes)
}

#[cfg(test)]
mod tests {
    use super::content_hash_hex;

    /// Ground-truth vectors from Python `xxhash.xxh128_hexdigest` (seed 0).
    #[test]
    fn matches_python_xxh128() {
        assert_eq!(content_hash_hex(b""), "99aa06d3014798d86001c324468d497f");
        assert_eq!(content_hash_hex(b"hello"), "b5e9c1ad071b3e7fc779cfaa5e523818");
        assert_eq!(content_hash_hex(&[0, 1, 2, 3]), "eb70bf5fc779e9e6a6111d53e80a3db5");
    }
}
