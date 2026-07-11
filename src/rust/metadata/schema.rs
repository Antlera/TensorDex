//! SQLite schema — DDL + PRAGMAs applied on open.

pub const PRAGMAS: &[(&str, &str)] = &[
    ("foreign_keys", "ON"),
    ("journal_mode", "WAL"),
    ("synchronous", "NORMAL"),
];

pub const SCHEMA_DDL: &[&str] = &[
    "CREATE TABLE IF NOT EXISTS tensors (
        id TEXT PRIMARY KEY,
        shape TEXT NOT NULL,
        dtype TEXT NOT NULL,
        size_bytes INTEGER NOT NULL,
        storage_uri TEXT NOT NULL,
        fingerprint BLOB,
        created_at TEXT NOT NULL
    )",
    "CREATE TABLE IF NOT EXISTS model_mappings (
        model_name TEXT NOT NULL,
        param_name TEXT NOT NULL,
        tensor_id TEXT NOT NULL,
        created_at TEXT NOT NULL,
        PRIMARY KEY (model_name, param_name),
        FOREIGN KEY(tensor_id) REFERENCES tensors(id) ON DELETE CASCADE
    )",
    "CREATE TABLE IF NOT EXISTS model_meta (
        model_name TEXT PRIMARY KEY,
        status TEXT NOT NULL,
        total_tensors INTEGER NOT NULL DEFAULT 0,
        metadata TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )",
    "CREATE INDEX IF NOT EXISTS idx_model_mappings_tensor_id
        ON model_mappings(tensor_id)",
    // The delta-base graph: a row here means `tensor_id`'s blob is a
    // delta encoded against `base_tensor_id` with `codec`. Raw tensors
    // have no row. Keeping this in SQL (not just in blob headers) lets
    // gc and manifest building run as indexed queries instead of
    // scanning every blob's safetensors header.
    "CREATE TABLE IF NOT EXISTS tensor_deltas (
        tensor_id TEXT PRIMARY KEY,
        base_tensor_id TEXT NOT NULL,
        codec TEXT NOT NULL,
        FOREIGN KEY(tensor_id) REFERENCES tensors(id) ON DELETE CASCADE
    )",
    "CREATE INDEX IF NOT EXISTS idx_tensor_deltas_base
        ON tensor_deltas(base_tensor_id)",
];
