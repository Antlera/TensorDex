//! Ingest process — reads .safetensors files, deduplicates against the
//! `MetadataStore`, computes BCS fingerprints, writes blobs, and commits
//! all DB rows in a single transaction. Python's only job on this path is
//! to hand over file paths + a model name.

pub mod blob_writer;
pub mod hash;
pub mod pipeline;
pub mod safetensors_reader;

pub use pipeline::ingest_from_safetensors_files;
