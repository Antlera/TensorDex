//! Metadata layer — owns all persistent state about tensors.
//!
//! Submodules:
//!   `schema`       — SQLite DDL + PRAGMAs
//!   `store`        — MetadataStore pyclass (rusqlite Connection + CRUD)
//!   `fingerprint`  — in-memory BCS fingerprint arena (FingerprintStore pyclass)

pub mod fingerprint;
pub mod schema;
pub mod store;
