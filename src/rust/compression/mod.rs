//! Compression orchestration: plan schema, (future) planner, and batch executor.
//!
//! - `plan`    — `CompressionPlan` data structure + per-pair reports
//! - `execute` — runs a batch of plans by dispatching to kernels + resolvers
//!
//! Lower layers: `crate::kernels` (algorithms), `crate::codec` (generic zstd/lz4
//! backends), `crate::resolvers` (tensor I/O).

pub mod execute;
pub mod plan;
pub mod planner;
