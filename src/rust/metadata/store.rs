//! MetadataStore — owns the SQLite connection and all CRUD.
//!
//! Python engine holds this as `self.metadata` and calls through; Python
//! never opens a `sqlite3` handle of its own. All row tuples cross FFI in
//! positional form to keep the boundary lean.

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Mutex;

use chrono::Utc;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use rusqlite::{params, params_from_iter, Connection, OptionalExtension};

use super::fingerprint::FingerprintStore;
use super::schema::{PRAGMAS, SCHEMA_DDL};

/// Tensor metadata tuple WITHOUT the fingerprint blob — the blob never
/// crosses the FFI boundary as data; it flows directly into
/// `FingerprintStore` via dedicated methods.
///
/// `(id, shape_json, dtype, size_bytes, storage_uri)`
pub type TensorTuple = (String, String, String, i64, String);

/// `(model_name, param_name, tensor_id)`
pub type MappingTuple = (String, String, String);

/// `(id, shape_json, dtype, size_bytes, storage_uri, fingerprint, created_at)`
pub type TensorInsertRow = (String, String, String, i64, String, Option<Vec<u8>>, String);

/// `(model_name, param_name, tensor_id, created_at)`
pub type MappingInsertRow = (String, String, String, String);

fn sql_err(e: rusqlite::Error) -> PyErr {
    PyRuntimeError::new_err(format!("sqlite: {}", e))
}

#[pyclass(module = "tensordex._ops")]
pub struct MetadataStore {
    conn: Mutex<Connection>,
}

impl MetadataStore {
    fn apply_schema(conn: &Connection) -> rusqlite::Result<()> {
        for (name, value) in PRAGMAS {
            conn.pragma_update(None, name, value)?;
        }
        for stmt in SCHEMA_DDL {
            conn.execute(stmt, [])?;
        }
        Ok(())
    }
}

#[pymethods]
impl MetadataStore {
    /// Open (or create) a SQLite file and apply schema + PRAGMAs.
    #[new]
    fn new(path: &str) -> PyResult<Self> {
        let conn = Connection::open(path).map_err(sql_err)?;
        Self::apply_schema(&conn).map_err(sql_err)?;
        Ok(Self {
            conn: Mutex::new(conn),
        })
    }

    // ------------------------------------------------------------------
    // Hydration
    //
    // Tensor *metadata* crosses FFI as tuples (cheap — strings + ints);
    // fingerprint *blobs* are shoveled straight into FingerprintStore on
    // the Rust side via `absorb_blob`, never touching Python.
    // ------------------------------------------------------------------

    /// Returns `(tensor_rows, mapping_rows)` with no fingerprint payload.
    fn hydrate_metadata(&self) -> PyResult<(Vec<TensorTuple>, Vec<MappingTuple>)> {
        let conn = self.conn.lock().unwrap();

        let tensors: Vec<TensorTuple> = {
            let mut stmt = conn
                .prepare(
                    "SELECT id, shape, dtype, size_bytes, storage_uri
                     FROM tensors ORDER BY created_at ASC",
                )
                .map_err(sql_err)?;
            let rows = stmt
                .query_map([], |row| {
                    Ok((
                        row.get::<_, String>(0)?,
                        row.get::<_, String>(1)?,
                        row.get::<_, String>(2)?,
                        row.get::<_, i64>(3)?,
                        row.get::<_, String>(4)?,
                    ))
                })
                .map_err(sql_err)?;
            rows.collect::<rusqlite::Result<_>>().map_err(sql_err)?
        };

        let mappings: Vec<MappingTuple> = {
            let mut stmt = conn
                .prepare("SELECT model_name, param_name, tensor_id FROM model_mappings")
                .map_err(sql_err)?;
            let rows = stmt
                .query_map([], |row| {
                    Ok((
                        row.get::<_, String>(0)?,
                        row.get::<_, String>(1)?,
                        row.get::<_, String>(2)?,
                    ))
                })
                .map_err(sql_err)?;
            rows.collect::<rusqlite::Result<_>>().map_err(sql_err)?
        };

        Ok((tensors, mappings))
    }

    /// Pour every non-null fingerprint blob directly into ``fp_store``.
    /// Returns ``(ok, skipped)`` — blobs with wrong length are skipped.
    fn load_fingerprints_into(
        &self,
        mut fp_store: PyRefMut<FingerprintStore>,
    ) -> PyResult<(usize, usize)> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn
            .prepare(
                "SELECT id, fingerprint FROM tensors
                 WHERE fingerprint IS NOT NULL
                 ORDER BY created_at ASC",
            )
            .map_err(sql_err)?;
        let mut rows = stmt.query([]).map_err(sql_err)?;
        let (mut ok, mut skipped) = (0usize, 0usize);
        while let Some(row) = rows.next().map_err(sql_err)? {
            let id: String = row.get(0).map_err(sql_err)?;
            let blob: Vec<u8> = row.get(1).map_err(sql_err)?;
            if fp_store.absorb_blob(id, &blob) {
                ok += 1;
            } else {
                skipped += 1;
            }
        }
        Ok((ok, skipped))
    }

    /// Same as ``load_fingerprints_into`` but scoped to an explicit id list.
    fn load_fingerprints_by_ids_into(
        &self,
        ids: Vec<String>,
        mut fp_store: PyRefMut<FingerprintStore>,
    ) -> PyResult<(usize, usize)> {
        if ids.is_empty() {
            return Ok((0, 0));
        }
        let conn = self.conn.lock().unwrap();
        let (mut ok, mut skipped) = (0usize, 0usize);
        for chunk in ids.chunks(900) {
            let placeholders = vec!["?"; chunk.len()].join(",");
            let query = format!(
                "SELECT id, fingerprint FROM tensors
                 WHERE fingerprint IS NOT NULL AND id IN ({})",
                placeholders
            );
            let mut stmt = conn.prepare(&query).map_err(sql_err)?;
            let mut rows = stmt
                .query(params_from_iter(chunk.iter()))
                .map_err(sql_err)?;
            while let Some(row) = rows.next().map_err(sql_err)? {
                let id: String = row.get(0).map_err(sql_err)?;
                let blob: Vec<u8> = row.get(1).map_err(sql_err)?;
                if fp_store.absorb_blob(id, &blob) {
                    ok += 1;
                } else {
                    skipped += 1;
                }
            }
        }
        Ok((ok, skipped))
    }

    // ------------------------------------------------------------------
    // Ingest — atomic batch write
    // ------------------------------------------------------------------

    /// Insert new tensor rows + upsert model mappings in a single
    /// transaction. Tensor rows collide on primary key id → `INSERT OR
    /// IGNORE`; mappings collide on `(model_name, param_name)` → overwrite.
    pub fn ingest_batch(
        &self,
        tensor_rows: Vec<TensorInsertRow>,
        mapping_rows: Vec<MappingInsertRow>,
    ) -> PyResult<()> {
        let mut conn = self.conn.lock().unwrap();
        let tx = conn.transaction().map_err(sql_err)?;
        {
            let mut ins_tensor = tx
                .prepare(
                    "INSERT OR IGNORE INTO tensors
                     (id, shape, dtype, size_bytes, storage_uri, fingerprint, created_at)
                     VALUES (?, ?, ?, ?, ?, ?, ?)",
                )
                .map_err(sql_err)?;
            for row in &tensor_rows {
                ins_tensor
                    .execute(params![row.0, row.1, row.2, row.3, row.4, row.5, row.6])
                    .map_err(sql_err)?;
            }
            let mut ins_map = tx
                .prepare(
                    "INSERT INTO model_mappings (model_name, param_name, tensor_id, created_at)
                     VALUES (?, ?, ?, ?)
                     ON CONFLICT(model_name, param_name) DO UPDATE SET
                         tensor_id=excluded.tensor_id,
                         created_at=excluded.created_at",
                )
                .map_err(sql_err)?;
            for row in &mapping_rows {
                ins_map
                    .execute(params![row.0, row.1, row.2, row.3])
                    .map_err(sql_err)?;
            }
        }
        tx.commit().map_err(sql_err)
    }

    /// Return `(id, storage_uri)` for the subset of ``ids`` already present.
    /// Chunks internally to stay under SQLite's parameter limit.
    pub fn existing_tensor_ids(&self, ids: Vec<String>) -> PyResult<Vec<(String, String)>> {
        if ids.is_empty() {
            return Ok(Vec::new());
        }
        let conn = self.conn.lock().unwrap();
        let mut out = Vec::new();
        for chunk in ids.chunks(900) {
            let placeholders = vec!["?"; chunk.len()].join(",");
            let query = format!(
                "SELECT id, storage_uri FROM tensors WHERE id IN ({})",
                placeholders
            );
            let mut stmt = conn.prepare(&query).map_err(sql_err)?;
            let rows = stmt
                .query_map(params_from_iter(chunk.iter()), |row| {
                    Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
                })
                .map_err(sql_err)?;
            for r in rows {
                out.push(r.map_err(sql_err)?);
            }
        }
        Ok(out)
    }

    // ------------------------------------------------------------------
    // Individual lookups (fallbacks from Python's in-memory caches)
    // ------------------------------------------------------------------

    fn lookup_tensor_id(&self, model_name: &str, param_name: &str) -> PyResult<Option<String>> {
        let conn = self.conn.lock().unwrap();
        conn.query_row(
            "SELECT tensor_id FROM model_mappings WHERE model_name = ? AND param_name = ?",
            params![model_name, param_name],
            |row| row.get::<_, String>(0),
        )
        .optional()
        .map_err(sql_err)
    }

    fn get_storage_uri(&self, tensor_id: &str) -> PyResult<Option<String>> {
        let conn = self.conn.lock().unwrap();
        conn.query_row(
            "SELECT storage_uri FROM tensors WHERE id = ?",
            params![tensor_id],
            |row| row.get::<_, String>(0),
        )
        .optional()
        .map_err(sql_err)
    }

    /// Selective hydration helper — fetch metadata rows by explicit id list.
    /// Fingerprints travel via ``load_fingerprints_by_ids_into``.
    fn select_tensors_by_ids(&self, ids: Vec<String>) -> PyResult<Vec<TensorTuple>> {
        if ids.is_empty() {
            return Ok(Vec::new());
        }
        let conn = self.conn.lock().unwrap();
        let mut out = Vec::new();
        for chunk in ids.chunks(900) {
            let placeholders = vec!["?"; chunk.len()].join(",");
            let query = format!(
                "SELECT id, shape, dtype, size_bytes, storage_uri
                 FROM tensors WHERE id IN ({})",
                placeholders
            );
            let mut stmt = conn.prepare(&query).map_err(sql_err)?;
            let rows = stmt
                .query_map(params_from_iter(chunk.iter()), |row| {
                    Ok((
                        row.get::<_, String>(0)?,
                        row.get::<_, String>(1)?,
                        row.get::<_, String>(2)?,
                        row.get::<_, i64>(3)?,
                        row.get::<_, String>(4)?,
                    ))
                })
                .map_err(sql_err)?;
            for r in rows {
                out.push(r.map_err(sql_err)?);
            }
        }
        Ok(out)
    }

    // ------------------------------------------------------------------
    // Model lifecycle
    // ------------------------------------------------------------------

    fn init_model(&self, model_name: &str, metadata_json: Option<String>) -> PyResult<()> {
        let now = Utc::now().to_rfc3339();
        let mut conn = self.conn.lock().unwrap();
        let tx = conn.transaction().map_err(sql_err)?;
        tx.execute(
            "DELETE FROM model_mappings WHERE model_name = ?",
            params![model_name],
        )
        .map_err(sql_err)?;
        tx.execute(
            "DELETE FROM model_meta WHERE model_name = ?",
            params![model_name],
        )
        .map_err(sql_err)?;
        tx.execute(
            "INSERT INTO model_meta (model_name, status, total_tensors, metadata, created_at, updated_at)
             VALUES (?, ?, 0, ?, ?, ?)",
            params![model_name, "ingesting", metadata_json, now, now],
        )
        .map_err(sql_err)?;
        tx.commit().map_err(sql_err)
    }

    fn commit_model(&self, model_name: &str) -> PyResult<()> {
        let now = Utc::now().to_rfc3339();
        let conn = self.conn.lock().unwrap();
        let count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM model_mappings WHERE model_name = ?",
                params![model_name],
                |row| row.get(0),
            )
            .map_err(sql_err)?;
        conn.execute(
            "INSERT INTO model_meta (model_name, status, total_tensors, metadata, created_at, updated_at)
             VALUES (?, ?, ?, NULL, ?, ?)
             ON CONFLICT(model_name) DO UPDATE SET
                 status=excluded.status,
                 total_tensors=excluded.total_tensors,
                 updated_at=excluded.updated_at",
            params![model_name, "ready", count, now, now],
        )
        .map_err(sql_err)?;
        Ok(())
    }

    fn fail_model(&self, model_name: &str) -> PyResult<()> {
        let now = Utc::now().to_rfc3339();
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "INSERT INTO model_meta (model_name, status, total_tensors, metadata, created_at, updated_at)
             VALUES (?, 'failed', 0, NULL, ?, ?)
             ON CONFLICT(model_name) DO UPDATE SET
                 status='failed',
                 updated_at=excluded.updated_at",
            params![model_name, now, now],
        )
        .map_err(sql_err)?;
        Ok(())
    }

    /// Returns `(model_name, status, total_tensors, metadata_json, created_at, updated_at)`
    /// or ``None``. Python side parses the JSON if needed.
    fn get_model_state(
        &self,
        model_name: &str,
    ) -> PyResult<Option<(String, String, i64, Option<String>, String, String)>> {
        let conn = self.conn.lock().unwrap();
        conn.query_row(
            "SELECT model_name, status, total_tensors, metadata, created_at, updated_at
             FROM model_meta WHERE model_name = ?",
            params![model_name],
            |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, i64>(2)?,
                    row.get::<_, Option<String>>(3)?,
                    row.get::<_, String>(4)?,
                    row.get::<_, String>(5)?,
                ))
            },
        )
        .optional()
        .map_err(sql_err)
    }

    fn list_model_tensors(&self, model_name: &str) -> PyResult<Vec<(String, String)>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn
            .prepare("SELECT param_name, tensor_id FROM model_mappings WHERE model_name = ?")
            .map_err(sql_err)?;
        let rows = stmt
            .query_map(params![model_name], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
            })
            .map_err(sql_err)?;
        rows.collect::<rusqlite::Result<_>>().map_err(sql_err)
    }

    /// Returns `[(model_name, status, total_tensors, created_at, updated_at), ...]`
    /// ordered by `created_at` ascending. Drives `tensordex ls`.
    fn list_models(&self) -> PyResult<Vec<(String, String, i64, String, String)>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn
            .prepare(
                "SELECT model_name, status, total_tensors, created_at, updated_at
                 FROM model_meta ORDER BY created_at ASC",
            )
            .map_err(sql_err)?;
        let rows = stmt
            .query_map([], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, i64>(2)?,
                    row.get::<_, String>(3)?,
                    row.get::<_, String>(4)?,
                ))
            })
            .map_err(sql_err)?;
        rows.collect::<rusqlite::Result<_>>().map_err(sql_err)
    }

    /// Summed `size_bytes` over all tensors referenced by ``model_name`` (distinct).
    /// Powers ``tensordex info`` / ``stats`` without forcing Python to join.
    fn model_total_bytes(&self, model_name: &str) -> PyResult<i64> {
        let conn = self.conn.lock().unwrap();
        conn.query_row(
            "SELECT COALESCE(SUM(size_bytes), 0) FROM tensors
             WHERE id IN (SELECT DISTINCT tensor_id FROM model_mappings WHERE model_name = ?)",
            params![model_name],
            |row| row.get::<_, i64>(0),
        )
        .map_err(sql_err)
    }

    /// Transactionally drop every row belonging to ``model_name`` from
    /// ``model_mappings`` and ``model_meta``. Tensor rows and blobs are
    /// **not** touched — run ``gc_orphans`` afterwards to reclaim them
    /// once no other model references them.
    ///
    /// Returns ``(mappings_deleted, meta_deleted)``.
    fn delete_model(&self, model_name: &str) -> PyResult<(i64, i64)> {
        let mut conn = self.conn.lock().unwrap();
        let tx = conn.transaction().map_err(sql_err)?;
        let mappings = tx
            .execute(
                "DELETE FROM model_mappings WHERE model_name = ?",
                params![model_name],
            )
            .map_err(sql_err)? as i64;
        let meta = tx
            .execute(
                "DELETE FROM model_meta WHERE model_name = ?",
                params![model_name],
            )
            .map_err(sql_err)? as i64;
        tx.commit().map_err(sql_err)?;
        Ok((mappings, meta))
    }

    /// Find and delete every tensor row that no ``model_mappings`` row
    /// references, except those still needed as a delta base. Protected
    /// bases come from the ``tensor_deltas`` table (the SQL delta graph);
    /// the optional ``protect`` list adds extra ids on top, but callers
    /// normally pass ``None`` now that the base graph lives in SQL.
    ///
    /// Returns the ``(id, storage_uri)`` pairs that were actually deleted,
    /// so the caller can unlink the corresponding blobs.
    #[pyo3(signature = (protect=None))]
    fn gc_orphans(&self, protect: Option<Vec<String>>) -> PyResult<Vec<(String, String)>> {
        let mut protect_set: HashSet<String> = protect.unwrap_or_default().into_iter().collect();

        let mut conn = self.conn.lock().unwrap();
        let tx = conn.transaction().map_err(sql_err)?;

        // Protect every tensor any delta still depends on as a base —
        // read straight from the SQL delta graph, no blob-header scan.
        {
            let mut stmt = tx
                .prepare("SELECT DISTINCT base_tensor_id FROM tensor_deltas")
                .map_err(sql_err)?;
            let rows = stmt
                .query_map([], |row| row.get::<_, String>(0))
                .map_err(sql_err)?;
            for r in rows {
                protect_set.insert(r.map_err(sql_err)?);
            }
        }

        let candidates: Vec<(String, String)> = {
            let mut stmt = tx
                .prepare(
                    "SELECT id, storage_uri FROM tensors
                     WHERE id NOT IN (SELECT DISTINCT tensor_id FROM model_mappings)",
                )
                .map_err(sql_err)?;
            let rows = stmt
                .query_map([], |row| {
                    Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
                })
                .map_err(sql_err)?;
            rows.collect::<rusqlite::Result<_>>().map_err(sql_err)?
        };

        let deletable: Vec<(String, String)> = candidates
            .into_iter()
            .filter(|(id, _)| !protect_set.contains(id))
            .collect();

        if !deletable.is_empty() {
            let ids: Vec<&String> = deletable.iter().map(|(id, _)| id).collect();
            for chunk in ids.chunks(900) {
                let placeholders = vec!["?"; chunk.len()].join(",");
                let query = format!("DELETE FROM tensors WHERE id IN ({})", placeholders);
                tx.execute(&query, params_from_iter(chunk.iter()))
                    .map_err(sql_err)?;
            }
        }
        tx.commit().map_err(sql_err)?;
        Ok(deletable)
    }

    /// Update ``size_bytes`` / ``storage_uri`` for a tensor row, used
    /// after ``compress_pair`` rewrites the underlying blob.
    fn update_tensor_storage(
        &self,
        tensor_id: &str,
        size_bytes: i64,
        storage_uri: &str,
    ) -> PyResult<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "UPDATE tensors SET size_bytes = ?, storage_uri = ? WHERE id = ?",
            params![size_bytes, storage_uri, tensor_id],
        )
        .map_err(sql_err)?;
        Ok(())
    }

    // ------------------------------------------------------------------
    // Delta graph — base dependency edges (control plane, not blob headers)
    // ------------------------------------------------------------------

    /// Record that ``tensor_id``'s blob is a delta against ``base_tensor_id``.
    /// Called by ``compress_pair`` right after the blob is rewritten.
    fn set_tensor_delta(&self, tensor_id: &str, base_tensor_id: &str, codec: &str) -> PyResult<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "INSERT INTO tensor_deltas (tensor_id, base_tensor_id, codec)
             VALUES (?, ?, ?)
             ON CONFLICT(tensor_id) DO UPDATE SET
                 base_tensor_id=excluded.base_tensor_id,
                 codec=excluded.codec",
            params![tensor_id, base_tensor_id, codec],
        )
        .map_err(sql_err)?;
        Ok(())
    }

    /// Distinct base tensor ids referenced by any delta — the set gc must
    /// protect. Powers the ``bases_protected`` stat without a header scan.
    fn protected_base_ids(&self) -> PyResult<Vec<String>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn
            .prepare("SELECT DISTINCT base_tensor_id FROM tensor_deltas")
            .map_err(sql_err)?;
        let rows = stmt
            .query_map([], |row| row.get::<_, String>(0))
            .map_err(sql_err)?;
        rows.collect::<rusqlite::Result<_>>().map_err(sql_err)
    }

    /// Backfill ``tensor_deltas`` from explicit ``(tensor_id, base, codec)``
    /// triples — a one-time migration for hubs whose blobs were compressed
    /// before the delta graph moved into SQL. Idempotent.
    fn backfill_deltas(&self, rows: Vec<(String, String, String)>) -> PyResult<usize> {
        if rows.is_empty() {
            return Ok(0);
        }
        let mut conn = self.conn.lock().unwrap();
        let tx = conn.transaction().map_err(sql_err)?;
        {
            let mut stmt = tx
                .prepare(
                    "INSERT INTO tensor_deltas (tensor_id, base_tensor_id, codec)
                     VALUES (?, ?, ?)
                     ON CONFLICT(tensor_id) DO UPDATE SET
                         base_tensor_id=excluded.base_tensor_id, codec=excluded.codec",
                )
                .map_err(sql_err)?;
            for (tid, base, codec) in &rows {
                stmt.execute(params![tid, base, codec]).map_err(sql_err)?;
            }
        }
        tx.commit().map_err(sql_err)?;
        Ok(rows.len())
    }

    /// Build the full blob set a client needs to reconstruct the tensors in
    /// ``direct_ids`` — i.e. the closure over delta base chains — entirely
    /// from SQL. Replaces per-blob safetensors-header peeking on the server.
    ///
    /// Each row is
    /// ``(tensor_id, size_bytes, shape_json, dtype, codec?, base_tensor_id?)``;
    /// ``codec``/``base`` are non-null exactly for delta (compressed) blobs.
    fn manifest_blobs(
        &self,
        direct_ids: Vec<String>,
    ) -> PyResult<Vec<(String, i64, String, String, Option<String>, Option<String>)>> {
        let conn = self.conn.lock().unwrap();

        // 1. Close over the base chain via indexed point lookups.
        let mut seen: HashSet<String> = HashSet::new();
        let mut queue: VecDeque<String> = direct_ids.into_iter().collect();
        while let Some(id) = queue.pop_front() {
            if !seen.insert(id.clone()) {
                continue;
            }
            let base: Option<String> = conn
                .query_row(
                    "SELECT base_tensor_id FROM tensor_deltas WHERE tensor_id = ?",
                    params![id],
                    |row| row.get::<_, String>(0),
                )
                .optional()
                .map_err(sql_err)?;
            if let Some(b) = base {
                if !seen.contains(&b) {
                    queue.push_back(b);
                }
            }
        }

        // 2. Fetch the descriptor for every id in the closure in one shot.
        let ids: Vec<String> = seen.into_iter().collect();
        let mut out = Vec::with_capacity(ids.len());
        for chunk in ids.chunks(900) {
            let placeholders = vec!["?"; chunk.len()].join(",");
            let query = format!(
                "SELECT t.id, t.size_bytes, t.shape, t.dtype, d.codec, d.base_tensor_id
                 FROM tensors t
                 LEFT JOIN tensor_deltas d ON d.tensor_id = t.id
                 WHERE t.id IN ({})",
                placeholders
            );
            let mut stmt = conn.prepare(&query).map_err(sql_err)?;
            let rows = stmt
                .query_map(params_from_iter(chunk.iter()), |row| {
                    Ok((
                        row.get::<_, String>(0)?,
                        row.get::<_, i64>(1)?,
                        row.get::<_, String>(2)?,
                        row.get::<_, String>(3)?,
                        row.get::<_, Option<String>>(4)?,
                        row.get::<_, Option<String>>(5)?,
                    ))
                })
                .map_err(sql_err)?;
            for r in rows {
                out.push(r.map_err(sql_err)?);
            }
        }
        Ok(out)
    }

    // ------------------------------------------------------------------
    // Stats
    // ------------------------------------------------------------------

    fn count_tensors(&self) -> PyResult<i64> {
        self.conn
            .lock()
            .unwrap()
            .query_row("SELECT COUNT(*) FROM tensors", [], |row| row.get(0))
            .map_err(sql_err)
    }

    fn count_models(&self) -> PyResult<i64> {
        self.conn
            .lock()
            .unwrap()
            .query_row("SELECT COUNT(*) FROM model_meta", [], |row| row.get(0))
            .map_err(sql_err)
    }

    /// Returns `{shape_json: count}` grouped at the SQL level so Python only
    /// has to pretty-print the keys.
    fn shape_distribution(&self) -> PyResult<HashMap<String, i64>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn
            .prepare("SELECT shape, COUNT(*) FROM tensors GROUP BY shape")
            .map_err(sql_err)?;
        let rows = stmt
            .query_map([], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?))
            })
            .map_err(sql_err)?;
        let mut out = HashMap::new();
        for r in rows {
            let (k, v) = r.map_err(sql_err)?;
            out.insert(k, v);
        }
        Ok(out)
    }
}
