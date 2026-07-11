"""
Shared database query helpers for TensorDex charts.

Call ``init(db_path)`` once before any queries.
Uses a connection pool to avoid reconnection + PRAGMA overhead per query.
"""

import queue
import sqlite3

_DB_PATH = None
_pool = queue.Queue()
_POOL_SIZE = 4


def init(db_path: str):
    """Set the database path and pre-warm the connection pool."""
    global _DB_PATH
    _DB_PATH = db_path
    for _ in range(_POOL_SIZE):
        _pool.put(_create_conn())


def _create_conn():
    """Create a new read-only connection with optimal settings."""
    conn = sqlite3.connect(_DB_PATH, timeout=30, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")   # wait up to 30s if locked
    conn.execute("PRAGMA query_only=ON")         # read-only safety
    conn.execute("PRAGMA mmap_size=4294967296")  # 4GB mmap for fast IO
    conn.execute("PRAGMA cache_size=-2097152")   # 2GB page cache (negative = KB)
    conn.execute("PRAGMA temp_store=MEMORY")     # temp tables in memory
    return conn


def _connect():
    """Get a connection from the pool, or create a new one if empty."""
    try:
        return _pool.get_nowait()
    except queue.Empty:
        return _create_conn()


def _release(conn):
    """Return a connection to the pool."""
    _pool.put(conn)


def query(sql: str, params=None):
    """Execute *sql* and return all rows."""
    conn = _connect()
    try:
        return conn.execute(sql, params or []).fetchall()
    finally:
        _release(conn)


def query_one(sql: str, params=None):
    """Execute *sql* and return the first row."""
    conn = _connect()
    try:
        return conn.execute(sql, params or []).fetchone()
    finally:
        _release(conn)


# ── Shared SQL fragments ─────────────────────────────────────────────

LAYER_TYPE_CASE = """
    CASE
        WHEN param_name LIKE '%q_proj%' THEN 'q_proj'
        WHEN param_name LIKE '%k_proj%' THEN 'k_proj'
        WHEN param_name LIKE '%v_proj%' THEN 'v_proj'
        WHEN param_name LIKE '%o_proj%' THEN 'o_proj'
        WHEN param_name LIKE '%gate_proj%' OR param_name LIKE '%gate_up%' THEN 'gate_proj'
        WHEN param_name LIKE '%up_proj%' THEN 'up_proj'
        WHEN param_name LIKE '%down_proj%' THEN 'down_proj'
        WHEN param_name LIKE '%embed%' THEN 'embed'
        WHEN param_name LIKE '%lm_head%' OR param_name LIKE '%embed_out%' THEN 'lm_head'
        WHEN param_name LIKE '%layernorm%' OR param_name LIKE '%layer_norm%'
             OR param_name LIKE '%norm%' THEN 'norm'
        WHEN param_name LIKE '%mlp%' OR param_name LIKE '%fc%' THEN 'other'
        ELSE 'other'
    END
"""
