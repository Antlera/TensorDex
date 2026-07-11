"""TensorDex engine: wiring of MetadataStore + StorageBackend + FingerprintStore.

The engine owns no SQL, no blob I/O, and no ingest compute — it threads
calls through the Rust ``MetadataStore`` (persistent state), one
``StorageBackend`` instance (read path + S3 writes), the Rust
``FingerprintStore`` (in-memory BCS arena), and ``ingest_from_safetensors_files``
(atomic ingest on local disk).
"""

from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch

from tensordex import _ops
from tensordex._ops import FingerprintStore, MetadataStore
from tensordex.core._serde import (
    deserialize_metadata,
    serialize_metadata,
    shape_from_json,
)
from tensordex.core.codec import (
    CODEC_TENSORX,
    SUPPORTED_CODECS,
    CompressedBlob,
    load_blob,
    probe_codec,
    read_base_id,
    save_compressed,
)
from tensordex.core.metadata import (
    BCS_DEFAULT_D,
    BCS_DEFAULT_W,
)
from tensordex.core.storage import (
    LocalStorageBackend,
    S3StorageBackend,
    StorageBackend,
)

_DTYPE_BY_NAME: Dict[str, torch.dtype] = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "float64": torch.float64,
    "int8": torch.int8,
    "int16": torch.int16,
    "int32": torch.int32,
    "int64": torch.int64,
    "uint8": torch.uint8,
    "bool": torch.bool,
}

# Hybrid CR predictor v3 — must match `DEFAULT_HYBRID_COEFFS` in the Rust
# planner. Coefficients are fitted on 710K real TensorX (tratio) pairs.
HYBRID_COEFFS_V3: Tuple[float, float, float, float] = (
    -23.727944,
    0.522466,
    1.966862,
    -0.043132,
)

# Same default as `FlexSplit.STANDALONE_ZSTD_CR`.
DEFAULT_ATTACH_CR_THRESHOLD = 0.70


def _dtype_from_name(name: str) -> torch.dtype:
    try:
        return _DTYPE_BY_NAME[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported dtype name: {name!r}") from exc


def _tensor_to_bytes(tensor: torch.Tensor) -> bytes:
    """Raw little-endian bytes, matching what the Rust codecs expect."""
    return bytes(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())


def _bytes_to_tensor(
    payload: bytes, dtype: torch.dtype, shape: Tuple[int, ...]
) -> torch.Tensor:
    view = torch.frombuffer(bytearray(payload), dtype=torch.uint8).clone()
    return view.view(dtype).reshape(shape)


def _item_size_for_dtype(dtype: torch.dtype) -> int:
    return torch.empty((), dtype=dtype).element_size()

logger = logging.getLogger(__name__)


class ModelNotReadyError(RuntimeError):
    """Raised when attempting to access tensors for a model that isn't ready."""


class IntegrityError(RuntimeError):
    """Raised when a tensor's bytes don't hash to its content-addressed id."""


class TensorDex:
    """Content-addressable tensor engine: MetadataStore + StorageBackend + FingerprintStore."""

    def __init__(
        self,
        storage_dir: str = "./data/tensordex",
        backend: str | StorageBackend = "local",
        *,
        bcs_d: int = BCS_DEFAULT_D,
        bcs_w: int = BCS_DEFAULT_W,
        backend_options: Optional[Dict[str, Any]] = None,
        hydrate_all: bool = False,
        **_legacy_kwargs: Any,
    ) -> None:
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device("cpu")
        self.bcs_d = bcs_d
        self.bcs_w = bcs_w
        self.k = bcs_d * bcs_w
        self.backend_kind = "local"
        self.backend_options = backend_options or {}
        self._lock = threading.RLock()

        self.storage_backend = self._resolve_backend(backend)
        self._db_path = self.storage_dir / "metadata.db"
        self.metadata = MetadataStore(str(self._db_path))

        # The Rust MetadataStore (SQLite) is the single source of truth for
        # tensor metadata, storage URIs, and model→tensor mappings; the engine
        # queries it on demand rather than keeping shadow dicts in sync. Only
        # the fingerprint arena is held in memory.
        self.fingerprints = FingerprintStore(self.k)

        # Lazy by default: fingerprints load on demand, so opening a huge hub
        # is cheap. Pass hydrate_all=True to eagerly load the whole fingerprint
        # arena (e.g. for repeated whole-hub similarity scans).
        if hydrate_all:
            self._hydrate_state()
        else:
            logger.debug("TensorDex opened with lazy hydration")

    @classmethod
    def open(cls, storage_dir: str, **kwargs: Any) -> TensorDex:
        """Open an existing TensorDex storage directory. Raises if it does not exist."""
        if not os.path.exists(storage_dir):
            raise FileNotFoundError(f"Storage directory {storage_dir} not found")
        return cls(storage_dir=storage_dir, **kwargs)

    # ------------------------------------------------------------------
    # Initialization helpers
    # ------------------------------------------------------------------

    def _resolve_backend(self, backend: str | StorageBackend) -> StorageBackend:
        if isinstance(backend, StorageBackend):
            return backend
        if backend == "local":
            self.backend_kind = "local"
            root_dir = self.backend_options.get("root_dir", self.storage_dir)
            return LocalStorageBackend(root_dir)
        if backend == "s3":
            self.backend_kind = "s3"
            options = self.backend_options
            bucket = options.get("bucket")
            if not bucket:
                raise ValueError("S3 backend requires 'bucket' in backend_options")
            return S3StorageBackend(
                bucket=bucket,
                region=options.get("region"),
                prefix=options.get("prefix", ""),
            )
        raise ValueError(f"Unsupported backend '{backend}'")

    # ------------------------------------------------------------------
    # State hydration
    # ------------------------------------------------------------------

    def _hydrate_state(self) -> None:
        """Eagerly load every fingerprint into the in-memory arena.

        Tensor metadata + storage URIs are read straight from SQLite on
        demand, so the only thing worth pre-loading is the fingerprint
        arena (for repeated whole-hub similarity scans).
        """
        with self._lock:
            self.fingerprints.clear()
            ok, skipped = self.metadata.load_fingerprints_into(self.fingerprints)
            if ok:
                logger.info("Loaded %d fingerprints into store.", ok)
            if skipped:
                logger.warning(
                    "Skipped %d malformed fingerprint blobs during hydration", skipped
                )

    # ------------------------------------------------------------------
    # Model lifecycle — delegated to MetadataStore
    # ------------------------------------------------------------------

    def init_model(
        self, model_name: str, metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Initialize or reset ``model_name`` to ingesting state."""
        logger.info("Initializing model ingestion: %s", model_name)
        self.metadata.init_model(model_name, serialize_metadata(metadata))

    def commit_model(self, model_name: str) -> None:
        """Mark ``model_name`` as ready once ingestion succeeds."""
        self.metadata.commit_model(model_name)
        logger.info("Committed model %s", model_name)

    def fail_model(self, model_name: str) -> None:
        """Flag ``model_name`` as failed if ingestion aborts."""
        self.metadata.fail_model(model_name)
        logger.warning("Marking model %s as failed", model_name)

    def get_model_state(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Return lifecycle metadata for ``model_name`` if it exists."""
        row = self.metadata.get_model_state(model_name)
        if row is None:
            return None
        name, status, total, metadata_json, created, updated = row
        return {
            "model_name": name,
            "status": status,
            "total_tensors": int(total),
            "metadata": deserialize_metadata(metadata_json),
            "created_at": created,
            "updated_at": updated,
        }

    def _ensure_model_ready(self, model_name: str) -> None:
        state = self.get_model_state(model_name)
        if not state:
            return
        if state.get("status") != "ready":
            raise ModelNotReadyError(
                f"Model '{model_name}' is not ready (status={state.get('status')})"
            )

    def get_model_tensors(self, model_name: str) -> Dict[str, str]:
        """Return all ``(param_name -> tensor_id)`` mappings for a model."""
        return dict(self.metadata.list_model_tensors(model_name))

    # ------------------------------------------------------------------
    # Ingestion — one Rust call, one transaction
    # ------------------------------------------------------------------

    def ingest(
        self,
        files: Iterable[str],
        model_name: str,
        *,
        param_filter: Optional[Iterable[str]] = None,
    ) -> Dict[str, str]:
        """Ingest ``.safetensors`` shards into this hub.

        Rust owns every phase — hash, dedup, BCS, blob write, SQL commit —
        under one `ingest_from_safetensors_files` call. Python only hands
        over paths + (optionally) the set of tensor names to keep.

        Only the local filesystem backend is supported here; S3 ingest is a
        separate path planned for a future pass. Reading tensors back works
        over any backend.
        """
        if not isinstance(self.storage_backend, LocalStorageBackend):
            raise NotImplementedError(
                "hub.ingest(...) requires a local backend; S3 ingest is not yet wired up"
            )

        file_list = [str(Path(f).resolve()) for f in files]
        if not file_list:
            return {}
        filter_set = set(param_filter) if param_filter is not None else None

        # Rust wrote tensors, mappings, and fingerprints to SQLite + the
        # fingerprint arena directly; nothing to mirror Python-side.
        result: Dict[str, str] = _ops.ingest_from_safetensors_files(
            self.metadata,
            self.fingerprints,
            str(self.storage_backend.root_dir),
            file_list,
            model_name,
            filter_set,
            self.bcs_d,
            self.bcs_w,
        )
        return result

    # ------------------------------------------------------------------
    # Tensor retrieval
    # ------------------------------------------------------------------

    def _lookup_tensor_id_for_param(self, model_name: str, param_name: str) -> str:
        tid = self.metadata.lookup_tensor_id(model_name, param_name)
        if tid is None:
            raise KeyError(f"Tensor mapping missing for {model_name}:{param_name}")
        return tid

    def _get_storage_uri_optional(self, tensor_id: str) -> Optional[str]:
        """Return ``storage_uri`` if set; otherwise ``None`` (id-path fallback may still work)."""
        uri = self.metadata.get_storage_uri(tensor_id)
        if uri is None:
            raise KeyError(f"Tensor {tensor_id} not found in metadata store")
        return uri or None

    def _expected_shape_for_tid(self, tensor_id: str) -> Optional[Tuple[int, ...]]:
        """Shape for a tensor id, read from SQL — or ``None`` if unknown."""
        rows = self.metadata.select_tensors_by_ids([tensor_id])
        if not rows:
            return None
        try:
            return tuple(int(x) for x in shape_from_json(rows[0][1]))
        except (TypeError, ValueError):
            return None

    def _load_tensor_by_tensor_id(
        self,
        tensor_id: str,
        expected_shape: Optional[Tuple[int, ...]],
    ) -> torch.Tensor:
        """Same blob layout / key selection as Rust ``SimpleTensorResolver`` / ``S3TensorResolver``."""
        backend = self.storage_backend
        if isinstance(backend, (LocalStorageBackend, S3StorageBackend)):
            return backend.load_tensor_by_id(tensor_id, expected_shape)
        raise KeyError(
            f"Tensor {tensor_id} has no storage URI and backend {type(backend).__name__!r} "
            "does not support tensor_id path resolution"
        )

    def _resolve_local_blob_path(self, tensor_id: str) -> Path:
        """Return the canonical on-disk path for ``tensor_id``.

        Only supports the local backend today; decompression runs Python-side
        and needs a file-system path to hand to ``safe_open``. S3 reads are
        delegated to the legacy (raw-only) path below.
        """
        backend = self.storage_backend
        if not isinstance(backend, LocalStorageBackend):
            raise NotImplementedError(
                "Delta-encoded blobs currently require a local backend"
            )
        path = backend.blob_path_for_id(tensor_id)
        if not path.exists():
            legacy = backend._legacy_blob_path(tensor_id)
            if legacy.exists():
                return legacy
            raise FileNotFoundError(
                f"Blob not found for tensor_id={tensor_id} (tried {path})"
            )
        return path

    def _verify_tensor(self, tensor_id: str, tensor: torch.Tensor) -> None:
        """Recompute the content hash of ``tensor`` and check it equals its id.

        Catches on-disk corruption / bit-rot for free, since the id *is* the
        XXH3-128 of the tensor's raw bytes — byte-for-byte identical to
        ``xxhash.xxh128_hexdigest``, the hash keyed by the published cache.
        """
        actual = _ops.content_hash(_tensor_to_bytes(tensor))
        if actual != tensor_id:
            raise IntegrityError(
                f"Integrity check failed for {tensor_id}: bytes hash to {actual}. "
                "The blob is corrupt, or was written by a different hash version."
            )

    def _decode_compressed_blob(
        self,
        blob: CompressedBlob,
        *,
        _seen: Optional[set[str]] = None,
        _cache: Optional[Dict[str, torch.Tensor]] = None,
        verify: bool = False,
    ) -> torch.Tensor:
        """Reconstruct the target tensor from a delta-encoded blob."""
        seen = set() if _seen is None else _seen
        if blob.base_tensor_id in seen:
            raise RuntimeError(
                f"Cycle detected while decoding base chain at {blob.base_tensor_id}"
            )
        seen.add(blob.base_tensor_id)

        base_tensor = self.get_tensor(
            tensor_id=blob.base_tensor_id, _seen=seen, _cache=_cache, verify=verify
        )
        target_dtype = _dtype_from_name(blob.target_dtype)
        base_bytes = _tensor_to_bytes(base_tensor.to(target_dtype))

        codec = blob.codec
        if codec == CODEC_TENSORX:
            raw = _ops.decompress_tensorx_rust(
                blob.compressed_bytes, base_bytes, blob.item_size
            )
        else:
            raise ValueError(f"Unsupported codec on disk: {codec!r}")

        return _bytes_to_tensor(bytes(raw), target_dtype, blob.target_shape)

    def get_tensor(
        self,
        tensor_id: Optional[str] = None,
        model_name: Optional[str] = None,
        param_name: Optional[str] = None,
        *,
        verify: bool = False,
        _seen: Optional[set[str]] = None,
        _cache: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        if not tensor_id:
            if not model_name or not param_name:
                raise ValueError(
                    "Either tensor_id or (model_name, param_name) must be provided"
                )
            tensor_id = self._lookup_tensor_id_for_param(model_name, param_name)

        # ``_cache`` memoizes decoded tensors across one logical operation
        # (e.g. a whole-model ``pull``) so a base shared by many delta
        # targets is read + decoded once instead of once per target. A cache
        # hit was already integrity-checked when first decoded.
        if _cache is not None and tensor_id in _cache:
            return _cache[tensor_id]

        # Local backend: always go through the codec-aware loader so delta
        # blobs transparently decode; raw blobs cost one extra header peek.
        if isinstance(self.storage_backend, LocalStorageBackend):
            path = self._resolve_local_blob_path(tensor_id)
            blob = load_blob(path)
            if isinstance(blob, CompressedBlob):
                result = self._decode_compressed_blob(
                    blob, _seen=_seen, _cache=_cache, verify=verify
                )
            else:
                result = blob.tensor
            if verify:
                self._verify_tensor(tensor_id, result)
            if _cache is not None:
                _cache[tensor_id] = result
            return result

        # Non-local backends keep the original raw-only path for now.
        expected_shape = self._expected_shape_for_tid(tensor_id)
        uri = self._get_storage_uri_optional(tensor_id)
        supports_id_fallback = isinstance(self.storage_backend, S3StorageBackend)

        if uri:
            try:
                return self.storage_backend.load_tensor(uri)
            except (FileNotFoundError, KeyError, RuntimeError, OSError) as first_err:
                if supports_id_fallback:
                    try:
                        return self._load_tensor_by_tensor_id(tensor_id, expected_shape)
                    except Exception:
                        raise first_err
                raise

        if supports_id_fallback:
            return self._load_tensor_by_tensor_id(tensor_id, expected_shape)

        raise KeyError(f"Tensor {tensor_id} missing storage URI")

    # ------------------------------------------------------------------
    # Fingerprint access — thin shim over Rust FingerprintStore
    # ------------------------------------------------------------------

    def get_fingerprint(self, tensor_id: str) -> torch.Tensor:
        """Return a single fingerprint vector as ``torch.float32``."""
        arr = self.fingerprints.get(tensor_id)
        if arr is None:
            self.load_fingerprints_to_memory()
            arr = self.fingerprints.get(tensor_id)
            if arr is None:
                raise KeyError(f"Fingerprint missing for tensor {tensor_id}")
        return torch.from_numpy(arr.astype(np.float32, copy=False))

    def get_fingerprints_batch(self, tensor_ids: List[str]) -> torch.Tensor:
        """Return an ``(N, k)`` ``torch.float32`` batch ordered to match ``tensor_ids``."""
        if not tensor_ids:
            return torch.empty(0, self.k, dtype=torch.float32)
        try:
            matrix = self.fingerprints.get_batch(tensor_ids)
        except KeyError:
            self.load_fingerprints_to_memory()
            matrix = self.fingerprints.get_batch(tensor_ids)
        return torch.from_numpy(matrix.astype(np.float32, copy=False))

    def _reload_fingerprints_from_sql(self) -> None:
        self.fingerprints.clear()
        _, skipped = self.metadata.load_fingerprints_into(self.fingerprints)
        if skipped:
            logger.warning(
                "Skipped %d malformed fingerprint blobs during reload", skipped
            )

    def export_fingerprints_to_npz(self, path: Optional[str] = None) -> Path:
        """Export all fingerprints to a compressed NPZ file."""
        target = Path(path) if path else self.storage_dir / "fingerprints.npz"
        target.parent.mkdir(parents=True, exist_ok=True)
        if len(self.fingerprints) == 0:
            self._reload_fingerprints_from_sql()
        matrix = self.fingerprints.matrix()
        ids = np.array(self.fingerprints.ids(), dtype=np.str_)
        np.savez_compressed(target, ids=ids, fingerprints=matrix)
        return target

    def load_fingerprints_to_memory(self, npz_path: Optional[str] = None) -> None:
        """Load fingerprints into the store, preferring the NPZ cache if present."""
        target = Path(npz_path) if npz_path else self.storage_dir / "fingerprints.npz"
        if target.exists():
            data = np.load(target, allow_pickle=False)
            ids = data.get("ids")
            vectors = data.get("fingerprints")
            if ids is None or vectors is None:
                raise ValueError(f"Fingerprint archive at {target} is missing data")
            vectors_i32 = np.ascontiguousarray(vectors, dtype=np.int32)
            self.fingerprints.clear()
            for i, tid in enumerate(ids.tolist()):
                self.fingerprints.insert_vec(str(tid), vectors_i32[i])
            return
        self._reload_fingerprints_from_sql()

    def load_tensors_selective(
        self, tensor_ids: List[str], load_fingerprints: bool = True
    ) -> None:
        """Load fingerprints for a specific list of tensor ids into the arena.

        Metadata is read from SQL on demand, so this only warms the
        fingerprint arena (idempotent — already-loaded ids are skipped).
        """
        if not tensor_ids or not load_fingerprints:
            return
        unique_ids = list({tid for tid in tensor_ids if tid not in self.fingerprints})
        if not unique_ids:
            return
        ok, skipped = self.metadata.load_fingerprints_by_ids_into(
            unique_ids, self.fingerprints
        )
        if ok:
            logger.info("Loaded %d fingerprints selectively.", ok)
        if skipped:
            logger.warning(
                "Skipped %d malformed fingerprint blobs during selective load", skipped
            )

    # ------------------------------------------------------------------
    # User-facing lifecycle ops (drive the CLI)
    # ------------------------------------------------------------------

    def download(
        self,
        hf_model_id: str,
        *,
        stored_model_name: Optional[str] = None,
        only: Optional[Iterable[str]] = None,
        revision: Optional[str] = None,
    ) -> Dict[str, str]:
        """Download a HuggingFace repo and ingest its safetensors shards.

        ``only`` mirrors ``huggingface-cli download --include``: when set,
        ingest restricts itself to the listed parameter names. ``revision``
        selects a branch/tag/commit (e.g. a ``stepN`` training checkpoint).
        """
        from tensordex.integrations.hf_io import ingest_model, ingest_model_partial

        if only:
            target = set(only)
            return ingest_model_partial(
                self,
                hf_model_id,
                target,
                stored_model_name=stored_model_name,
                revision=revision,
            )
        return ingest_model(
            self, hf_model_id, stored_model_name=stored_model_name, revision=revision
        )

    def ls(self) -> List[Dict[str, Any]]:
        """List every model in the hub with status + basic counters."""
        return [
            {
                "model_name": row[0],
                "status": row[1],
                "total_tensors": int(row[2]),
                "created_at": row[3],
                "updated_at": row[4],
            }
            for row in self.metadata.list_models()
        ]

    def info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Return lifecycle + mapping summary for ``model_name``.

        Includes the parameter→tensor_id dict and the aggregate byte
        footprint across *distinct* tensor rows (so shared-tensor models
        don't double-count).
        """
        state = self.get_model_state(model_name)
        if state is None:
            return None
        mappings = self.get_model_tensors(model_name)
        total_bytes = int(self.metadata.model_total_bytes(model_name))
        return {
            **state,
            "mappings": mappings,
            "total_bytes": total_bytes,
            "unique_tensors": len(set(mappings.values())),
        }

    def rm(self, model_name: str) -> Dict[str, int]:
        """Delete a model's mappings + metadata. Blobs survive — ``gc`` reclaims."""
        if self.get_model_state(model_name) is None:
            raise KeyError(f"Model '{model_name}' not found")
        mappings, meta = self.metadata.delete_model(model_name)
        logger.info(
            "Removed model %s (%d mappings, %d meta row)",
            model_name,
            mappings,
            meta,
        )
        return {"mappings_deleted": int(mappings), "meta_deleted": int(meta)}

    def backfill_deltas_from_blobs(self) -> int:
        """One-time migration: populate ``tensor_deltas`` from blob headers.

        Only needed for hubs whose blobs were delta-compressed *before* the
        base graph moved into SQL. Scans every compressed blob's header once
        and records its ``(tensor_id, base, codec)`` edge. Idempotent; safe
        to run on an already-migrated hub. Returns the edge count written.
        """
        if not isinstance(self.storage_backend, LocalStorageBackend):
            return 0
        blobs_root = self.storage_backend.root_dir / "blobs"
        if not blobs_root.exists():
            return 0
        rows: List[Tuple[str, str, str]] = []
        for path in blobs_root.rglob("*.safetensors"):
            try:
                codec = probe_codec(path)
            except Exception as exc:  # noqa: BLE001 — skip malformed blobs
                logger.warning("backfill: header read failed for %s: %s", path, exc)
                continue
            if codec is None:
                continue
            try:
                base = read_base_id(path)
            except Exception as exc:  # noqa: BLE001
                logger.warning("backfill: base_id read failed for %s: %s", path, exc)
                continue
            if base:
                rows.append((path.stem, base, codec))
        return int(self.metadata.backfill_deltas(rows)) if rows else 0

    def gc(self) -> Dict[str, Any]:
        """Delete tensor rows + blobs that nothing references.

        A tensor is "referenced" if any ``model_mappings`` row points at it
        *or* it is a delta base in ``tensor_deltas``. Both graphs live in
        SQL, so the protected set + orphan delete run as one indexed Rust
        transaction — no blob-header scan. Blob cleanup is best-effort
        after the SQL commit.
        """
        protected = self.metadata.protected_base_ids()
        deleted = self.metadata.gc_orphans(None)
        if not deleted:
            return {
                "tensors_deleted": 0,
                "blobs_deleted": 0,
                "blob_errors": 0,
                "bases_protected": len(protected),
            }

        backend = self.storage_backend
        blobs_deleted, errors = 0, 0
        for tid, uri in deleted:
            target_uri = uri or self._storage_uri_fallback(tid)
            if not target_uri:
                continue
            try:
                backend.delete_tensor(target_uri)
                blobs_deleted += 1
            except Exception as exc:  # noqa: BLE001 — log and continue
                logger.warning("Failed to delete blob %s: %s", target_uri, exc)
                errors += 1

        logger.info(
            "gc: %d tensor rows deleted, %d blobs unlinked, %d protected bases",
            len(deleted),
            blobs_deleted,
            len(protected),
        )
        return {
            "tensors_deleted": len(deleted),
            "blobs_deleted": blobs_deleted,
            "blob_errors": errors,
            "bases_protected": len(protected),
        }

    def _storage_uri_fallback(self, tensor_id: str) -> Optional[str]:
        """Derive a canonical URI for a tensor whose row had an empty storage_uri."""
        if isinstance(self.storage_backend, LocalStorageBackend):
            path = self.storage_backend.blob_path_for_id(tensor_id)
            return path.relative_to(self.storage_backend.root_dir).as_posix()
        return None

    # ------------------------------------------------------------------
    # Delta compression — overwrite raw blob with codec payload + header
    # ------------------------------------------------------------------

    def compress_pair(
        self,
        target_tensor_id: str,
        base_tensor_id: str,
        *,
        codec: str = CODEC_TENSORX,
        level: int = 3,
        skip_if_compressed: bool = True,
    ) -> Dict[str, Any]:
        """Compress ``target`` as a delta against ``base`` and overwrite its blob.

        The blob at ``blobs/xx/yy/<target_tensor_id>.safetensors`` is
        rewritten atomically with the compressed payload + a
        ``__metadata__`` header describing the codec, base id, and the
        target's original shape/dtype. ``tensors.size_bytes`` is updated
        to the physical on-disk size of the new blob.

        When ``skip_if_compressed`` is True (default) and the target is
        already in a compressed state, this returns a ``status="skipped"``
        result instead of raising — so batch drivers (``auto_compress``,
        ``auto_compress_all``) can run repeatedly without failing on
        previously-compressed tensors.
        """
        if codec not in SUPPORTED_CODECS:
            raise ValueError(f"Unsupported codec {codec!r}")
        if not isinstance(self.storage_backend, LocalStorageBackend):
            raise NotImplementedError(
                "compress_pair currently requires a local backend"
            )
        if target_tensor_id == base_tensor_id:
            raise ValueError("Target and base tensor ids must differ")

        target_path = self._resolve_local_blob_path(target_tensor_id)
        if probe_codec(target_path) is not None:
            if skip_if_compressed:
                existing_size = target_path.stat().st_size
                return {
                    "status": "skipped",
                    "reason": "already_compressed",
                    "target_tensor_id": target_tensor_id,
                    "base_tensor_id": base_tensor_id,
                    "compressed_bytes": int(existing_size),
                }
            raise ValueError(
                f"Tensor {target_tensor_id} is already compressed — "
                "re-compression over a delta chain is not supported"
            )

        # A base whose delta chain leads back to the target would create a
        # decode-time cycle (A→B→…→A). Walk the chain with header-only peeks
        # and refuse such pairs — the target simply stays raw, and batch
        # drivers report the pair as skipped.
        chain_id, hops = base_tensor_id, 0
        while True:
            chain_base = read_base_id(self._resolve_local_blob_path(chain_id))
            if chain_base is None:
                break
            if chain_base == target_tensor_id or hops >= 64:
                return {
                    "status": "skipped",
                    "reason": "base_chain_reaches_target",
                    "target_tensor_id": target_tensor_id,
                    "base_tensor_id": base_tensor_id,
                }
            chain_id, hops = chain_base, hops + 1

        target_tensor = self.get_tensor(tensor_id=target_tensor_id)
        base_tensor = self.get_tensor(tensor_id=base_tensor_id)

        if target_tensor.shape != base_tensor.shape:
            raise ValueError(
                f"Shape mismatch: target {tuple(target_tensor.shape)} vs "
                f"base {tuple(base_tensor.shape)}"
            )
        if target_tensor.dtype != base_tensor.dtype:
            base_tensor = base_tensor.to(target_tensor.dtype)

        dtype = target_tensor.dtype
        item_size = _item_size_for_dtype(dtype)
        target_bytes = _tensor_to_bytes(target_tensor)
        base_bytes = _tensor_to_bytes(base_tensor)
        original_bytes = len(target_bytes)

        if codec == CODEC_TENSORX:
            compressed = _ops.compress_tensorx_rust(
                target_bytes, base_bytes, item_size, level
            )
        else:
            raise ValueError(f"Unsupported codec {codec!r}")

        target_shape = tuple(int(x) for x in target_tensor.shape)
        target_dtype_name = str(dtype).removeprefix("torch.")
        new_size = save_compressed(
            target_path,
            codec=codec,
            base_tensor_id=base_tensor_id,
            item_size=item_size,
            level=level,
            target_shape=target_shape,
            target_dtype=target_dtype_name,
            compressed_bytes=bytes(compressed),
        )

        uri = target_path.relative_to(self.storage_backend.root_dir).as_posix()
        self.metadata.update_tensor_storage(target_tensor_id, int(new_size), uri)
        # Record the delta edge in SQL so gc / manifest never has to read a
        # blob header to learn what this tensor depends on.
        self.metadata.set_tensor_delta(target_tensor_id, base_tensor_id, codec)

        ratio = (original_bytes / new_size) if new_size else 0.0
        logger.info(
            "Compressed %s against %s: %d → %d bytes (%.2fx) via %s",
            target_tensor_id,
            base_tensor_id,
            original_bytes,
            new_size,
            ratio,
            codec,
        )
        return {
            "status": "ok",
            "target_tensor_id": target_tensor_id,
            "base_tensor_id": base_tensor_id,
            "codec": codec,
            "level": level,
            "original_bytes": original_bytes,
            "compressed_bytes": int(new_size),
            "ratio": ratio,
        }

    # ------------------------------------------------------------------
    # Planner — FlexSplit attach stage over one model
    # ------------------------------------------------------------------

    @staticmethod
    def _dtype_bits(dtype_name: str) -> Optional[int]:
        """Bit-width for a stored dtype string, or ``None`` if unrecognised."""
        dtype = _DTYPE_BY_NAME.get(str(dtype_name).removeprefix("torch."))
        return None if dtype is None else 8 * _item_size_for_dtype(dtype)

    def _ordered_entry(
        self, tid: str, shape_json: str, dtype: str
    ) -> Optional[Tuple[str, str, int]]:
        """Build a planner ``(tid, shape_key, n_bits)`` entry, or None to skip."""
        bits = self._dtype_bits(dtype)
        shape = shape_from_json(shape_json)
        if bits is None or not shape:
            return None
        shape_ints = tuple(int(x) for x in shape)
        return (tid, str(shape_ints), int(np.prod(shape_ints)) * bits)

    def plan_attach(
        self,
        model_name: str,
        *,
        cr_threshold: float = DEFAULT_ATTACH_CR_THRESHOLD,
        coeffs: Tuple[float, float, float, float] = HYBRID_COEFFS_V3,
        include_existing_bases: bool = False,
    ) -> Dict[str, Any]:
        """Run FlexSplit's attach stage over ``model_name``'s tensors.

        Python assembles the ordered ``[(tid, shape_key, n_bits)]`` list
        (sorted by param_name) from SQL and hands it to the Rust planner.
        When ``include_existing_bases`` is True, every tensor already in
        the hub is prepended as a candidate base, so the new model can
        attach to pre-existing bases from other models.

        Returns a dict mirroring ``_ops.AttachPlan``.
        """
        mappings = self.get_model_tensors(model_name)
        if not mappings:
            raise ValueError(f"Model '{model_name}' has no tensor mappings")
        model_tids = set(mappings.values())

        # Make sure the planner can see a fingerprint for every candidate it
        # ranks: the model's own tensors always, the whole arena when
        # attaching against pre-existing bases from other models.
        if include_existing_bases:
            self._reload_fingerprints_from_sql()
        else:
            self.load_tensors_selective(list(model_tids), load_fingerprints=True)

        # Tied-weight tensors show up twice under different param names; dedup
        # via ``seen`` so the planner never pairs a tid against itself.
        ordered: List[Tuple[str, str, int]] = []
        seen: set[str] = set()

        if include_existing_bases:
            for tid, shape_json, dtype, _sz, _uri in self.metadata.hydrate_metadata()[0]:
                if tid in seen or tid in model_tids:
                    continue
                entry = self._ordered_entry(tid, shape_json, dtype)
                if entry is not None:
                    ordered.append(entry)
                    seen.add(tid)

        # The model's own tensors, in deterministic param-name order.
        row_by_tid = {
            r[0]: r for r in self.metadata.select_tensors_by_ids(list(model_tids))
        }
        for param in sorted(mappings.keys()):
            tid = mappings[param]
            if tid in seen:
                continue
            row = row_by_tid.get(tid)
            if row is None:
                continue
            entry = self._ordered_entry(tid, row[1], row[2])
            if entry is not None:
                ordered.append(entry)
                seen.add(tid)

        plan = _ops.plan_attach(
            self.fingerprints,
            ordered,
            cr_threshold,
            tuple(coeffs),
            self.bcs_d,
        )

        # Filter out pairs that would only compress pre-existing bases
        # (``seen_bases``) against *themselves* — not possible, but also
        # strip any pair whose target isn't in the model being planned
        # (we never want to mutate blobs outside this model).
        model_tensor_ids = set(mappings.values())
        kept_pairs = [p for p in plan.pairs if p.target_id in model_tensor_ids]

        return {
            "model_name": model_name,
            "cr_threshold": cr_threshold,
            "coeffs": list(coeffs),
            "bases": list(plan.bases),
            "pairs": [
                {
                    "target_tensor_id": p.target_id,
                    "base_tensor_id": p.base_id,
                    "distance": float(p.distance),
                    "predicted_cr": float(p.predicted_cr),
                }
                for p in kept_pairs
            ],
            "skipped_no_fp": list(plan.skipped_no_fp),
            "n_shapes": int(plan.n_shapes),
            "n_pairs": len(kept_pairs),
            "n_bases": len(plan.bases),
        }

    def auto_compress(
        self,
        model_name: str,
        *,
        cr_threshold: float = DEFAULT_ATTACH_CR_THRESHOLD,
        codec: str = CODEC_TENSORX,
        level: int = 3,
        include_existing_bases: bool = False,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """Plan + execute attach-stage compression for ``model_name``.

        ``dry_run=True`` returns the plan without touching any blobs.
        Otherwise each ``(target, base)`` pair is compressed in order
        via :meth:`compress_pair`; per-pair results carry the actual
        realised ratio so callers can compare predictions vs reality.
        """
        plan = self.plan_attach(
            model_name,
            cr_threshold=cr_threshold,
            include_existing_bases=include_existing_bases,
        )

        if dry_run:
            return {**plan, "executed": False, "results": []}

        results: List[Dict[str, Any]] = []
        total_orig = 0
        total_new = 0
        failures = 0
        skipped = 0
        for pair in plan["pairs"]:
            target_id = pair["target_tensor_id"]
            base_id = pair["base_tensor_id"]
            try:
                res = self.compress_pair(
                    target_id, base_id, codec=codec, level=level
                )
            except (ValueError, NotImplementedError, KeyError) as exc:
                logger.warning(
                    "auto_compress: %s ← %s failed (%s)", target_id, base_id, exc
                )
                failures += 1
                results.append(
                    {**pair, "status": "failed", "error": str(exc)}
                )
                continue
            if res["status"] == "skipped":
                skipped += 1
                results.append(
                    {**pair, "status": "skipped", "reason": res["reason"]}
                )
                continue
            total_orig += int(res["original_bytes"])
            total_new += int(res["compressed_bytes"])
            results.append(
                {
                    **pair,
                    "status": "ok",
                    "original_bytes": int(res["original_bytes"]),
                    "compressed_bytes": int(res["compressed_bytes"]),
                    "actual_ratio": float(res["ratio"]),
                }
            )

        realised = (total_orig / total_new) if total_new else 0.0
        return {
            **plan,
            "executed": True,
            "codec": codec,
            "level": level,
            "results": results,
            "total_original_bytes": total_orig,
            "total_compressed_bytes": total_new,
            "realised_ratio": realised,
            "failures": failures,
            "skipped": skipped,
        }

    def auto_compress_all(
        self,
        *,
        cr_threshold: float = DEFAULT_ATTACH_CR_THRESHOLD,
        codec: str = CODEC_TENSORX,
        level: int = 3,
        include_existing_bases: bool = False,
        dry_run: bool = False,
        progress: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Run :meth:`auto_compress` over every ``ready`` model in the hub.

        Idempotent — already-compressed blobs are skipped via
        :meth:`compress_pair`'s ``skip_if_compressed``. ``progress`` is an
        optional callable ``(model_name, per_model_result) -> None`` so
        the CLI can update a progress bar between models.
        """
        summary: Dict[str, Any] = {
            "models_processed": 0,
            "models_skipped": 0,
            "total_pairs": 0,
            "total_ok": 0,
            "total_skipped_pairs": 0,
            "total_failed_pairs": 0,
            "total_original_bytes": 0,
            "total_compressed_bytes": 0,
            "per_model": [],
        }
        for row in self.metadata.list_models():
            model_name, status, _total, _ca, _ua = row
            if status != "ready":
                summary["models_skipped"] += 1
                continue
            try:
                res = self.auto_compress(
                    model_name,
                    cr_threshold=cr_threshold,
                    codec=codec,
                    level=level,
                    include_existing_bases=include_existing_bases,
                    dry_run=dry_run,
                )
            except (ValueError, NotImplementedError, KeyError) as exc:
                logger.warning("auto_compress_all: %s skipped (%s)", model_name, exc)
                summary["models_skipped"] += 1
                if progress is not None:
                    progress(model_name, {"status": "skipped", "error": str(exc)})
                continue

            summary["models_processed"] += 1
            summary["total_pairs"] += res.get("n_pairs", 0)
            if res.get("executed"):
                ok_pairs = sum(1 for r in res["results"] if r["status"] == "ok")
                skipped_pairs = sum(1 for r in res["results"] if r["status"] == "skipped")
                failed_pairs = sum(1 for r in res["results"] if r["status"] == "failed")
                summary["total_ok"] += ok_pairs
                summary["total_skipped_pairs"] += skipped_pairs
                summary["total_failed_pairs"] += failed_pairs
                summary["total_original_bytes"] += int(res.get("total_original_bytes", 0))
                summary["total_compressed_bytes"] += int(res.get("total_compressed_bytes", 0))
            summary["per_model"].append(
                {
                    "model_name": model_name,
                    "n_pairs": res.get("n_pairs", 0),
                    "executed": res.get("executed", False),
                    "realised_ratio": res.get("realised_ratio", 0.0),
                    "skipped": res.get("skipped", 0),
                    "failures": res.get("failures", 0),
                }
            )
            if progress is not None:
                progress(model_name, res)

        if summary["total_compressed_bytes"] > 0:
            summary["overall_realised_ratio"] = (
                summary["total_original_bytes"] / summary["total_compressed_bytes"]
            )
        else:
            summary["overall_realised_ratio"] = 0.0
        return summary

    # ------------------------------------------------------------------
    # Bundle compression — a group of related models (e.g. checkpoints)
    # ------------------------------------------------------------------

    def _plan_bundle_flexsplit(
        self,
        per_model: List[Tuple[str, Dict[str, str]]],
        rows: Dict[str, Any],
        *,
        by_param: bool,
        cr_threshold: float,
        min_ratio: float,
    ) -> Tuple[List[Dict[str, Any]], int, int]:
        """FlexSplit bundle plan — open nearby bases per ``(param, shape)`` series.

        Returns ``(pairs, n_bases, n_tensors)``. Each pair deltas a checkpoint's
        tensor against the *nearest* opened base of the same parameter; a member
        whose nearest base is still a poor delta (``>= cr_threshold``) stays raw.
        """
        from tensordex.compression import flexsplit as _flexsplit

        # key -> ordered list of distinct tids across checkpoints, + n_bits
        groups: Dict[str, List[str]] = {}
        nbits: Dict[str, int] = {}
        for _model, mappings in per_model:  # checkpoint (creation) order
            for param in sorted(mappings.keys()):
                tid = mappings[param]
                row = rows.get(tid)
                if row is None:
                    continue
                _id, shape_json, dtype, _sz, _uri = row
                bits = self._dtype_bits(dtype)
                shape = shape_from_json(shape_json)
                if bits is None or not shape:
                    continue
                shape_ints = tuple(int(x) for x in shape)
                key = f"{param}\x00{shape_ints}" if by_param else str(shape_ints)
                series = groups.setdefault(key, [])
                if tid not in series:  # identical tensors already dedup to one id
                    series.append(tid)
                nbits.setdefault(key, int(np.prod(shape_ints)) * bits)

        pairs: List[Dict[str, Any]] = []
        n_bases = 0
        n_tensors = 0
        for key, tids in groups.items():
            n_tensors += len(tids)
            if len(tids) <= 1:
                n_bases += len(tids)
                continue
            fps = self.get_fingerprints_batch(tids).numpy()
            bidx, attach = _flexsplit.plan_group(
                fps, nbits[key], coeffs=HYBRID_COEFFS_V3,
                bcs_d=self.bcs_d, min_ratio=min_ratio,
            )
            n_bases += len(bidx)
            for ti, bi, cr in attach:
                if cr >= cr_threshold:
                    n_bases += 1  # nearest base still a poor delta → keep raw
                    continue
                pairs.append(
                    {
                        "target_tensor_id": tids[ti],
                        "base_tensor_id": tids[bi],
                        "distance": float("nan"),
                        "predicted_cr": float(cr),
                    }
                )
        return pairs, n_bases, n_tensors

    def compress_bundle(
        self,
        model_names: List[str],
        *,
        cr_threshold: float = DEFAULT_ATTACH_CR_THRESHOLD,
        codec: str = CODEC_TENSORX,
        level: int = 3,
        by_param: bool = True,
        strategy: str = "star",
        min_ratio: float = 0.0,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """Compress a *bundle* of related models (e.g. training checkpoints) together.

        Plans **once** over the union of every bundle tensor, in checkpoint
        order, keyed by ``(param, shape)`` when ``by_param=True`` so a tensor
        only attaches to the same parameter in another checkpoint. Two
        strategies, both reconstructing at depth 1 (every delta against a *raw*
        base, never a chain):

        - ``"star"`` (default) — anchor every checkpoint on the **earliest**
          one's tensor. Simple, but a long run's late checkpoints delta against
          a far-away base and compress poorly.
        - ``"flexsplit"`` — adaptively open a new raw base wherever the predicted
          byte gain beats keeping it raw, so each checkpoint attaches to a
          *nearby* base. ``min_ratio`` raises the bar for opening a base. Wins
          on long / drifting series; see :mod:`tensordex.compression.flexsplit`.
        """
        # Bundle order = model creation order (checkpoints arrive over time).
        order = [r["model_name"] for r in self.ls() if r["model_name"] in set(model_names)]
        if not order:
            raise ValueError("No matching models in the bundle")

        per_model: List[Tuple[str, Dict[str, str]]] = [
            (m, self.get_model_tensors(m)) for m in order
        ]
        all_tids = [tid for _m, mp in per_model for tid in mp.values()]
        self.load_tensors_selective(all_tids, load_fingerprints=True)
        rows = {r[0]: r for r in self.metadata.select_tensors_by_ids(list(set(all_tids)))}

        if strategy == "flexsplit":
            pairs, n_bases, n_tensors = self._plan_bundle_flexsplit(
                per_model, rows, by_param=by_param,
                cr_threshold=cr_threshold, min_ratio=min_ratio,
            )
        elif strategy == "star":
            ordered: List[Tuple[str, str, int]] = []
            seen: set[str] = set()
            for _model, mappings in per_model:
                for param in sorted(mappings.keys()):
                    tid = mappings[param]
                    if tid in seen:
                        continue
                    row = rows.get(tid)
                    if row is None:
                        continue
                    _id, shape_json, dtype, _sz, _uri = row
                    bits = self._dtype_bits(dtype)
                    shape = shape_from_json(shape_json)
                    if bits is None or not shape:
                        continue
                    shape_ints = tuple(int(x) for x in shape)
                    n_bits = int(np.prod(shape_ints)) * bits
                    key = f"{param}\x00{shape_ints}" if by_param else str(shape_ints)
                    ordered.append((tid, key, n_bits))
                    seen.add(tid)
            plan = _ops.plan_attach(
                self.fingerprints, ordered, cr_threshold, HYBRID_COEFFS_V3, self.bcs_d
            )
            pairs = [
                {
                    "target_tensor_id": p.target_id,
                    "base_tensor_id": p.base_id,
                    "distance": float(p.distance),
                    "predicted_cr": float(p.predicted_cr),
                }
                for p in plan.pairs
            ]
            n_bases = len(plan.bases)
            n_tensors = len(ordered)
        else:
            raise ValueError(
                f"unknown bundle strategy {strategy!r} (use 'star' or 'flexsplit')"
            )

        base = {
            "models": order,
            "strategy": strategy,
            "n_tensors": n_tensors,
            "n_bases": n_bases,
            "n_pairs": len(pairs),
            "cr_threshold": cr_threshold,
        }
        if dry_run:
            return {**base, "executed": False, "pairs": pairs, "results": []}

        results: List[Dict[str, Any]] = []
        total_orig = total_new = failures = skipped = 0
        for pair in pairs:
            try:
                res = self.compress_pair(
                    pair["target_tensor_id"],
                    pair["base_tensor_id"],
                    codec=codec,
                    level=level,
                )
            except (ValueError, NotImplementedError, KeyError) as exc:
                failures += 1
                results.append({**pair, "status": "failed", "error": str(exc)})
                continue
            if res["status"] == "skipped":
                skipped += 1
                results.append({**pair, "status": "skipped"})
                continue
            total_orig += int(res["original_bytes"])
            total_new += int(res["compressed_bytes"])
            results.append({**pair, "status": "ok", "actual_ratio": float(res["ratio"])})

        return {
            **base,
            "executed": True,
            "codec": codec,
            "results": results,
            "total_original_bytes": total_orig,
            "total_compressed_bytes": total_new,
            "realised_ratio": (total_orig / total_new) if total_new else 0.0,
            "failures": failures,
            "skipped": skipped,
        }

    # ------------------------------------------------------------------
    # Pull — materialize a model back to a safetensors file
    # ------------------------------------------------------------------

    def _logical_bytes(self, dtype: str, shape: Iterable[int]) -> int:
        """Logical (decoded) byte size from dtype + shape, no blob touch."""
        tdt = _DTYPE_BY_NAME.get(str(dtype).removeprefix("torch."))
        if tdt is None:
            return 0
        n = 1
        for d in shape:
            n *= int(d)
        return n * _item_size_for_dtype(tdt)

    @staticmethod
    def _unshare_storage(sd: Dict[str, torch.Tensor]) -> None:
        """Clone tensors that alias the same storage so safetensors can save them.

        Identical tensors (e.g. tied ``rotary_emb.inv_freq`` buffers) dedup to
        one id and decode to the *same* cached object; safetensors refuses to
        save aliases. Clone only the duplicates — distinct large weights have
        distinct ids and are never copied.
        """
        seen: set[int] = set()
        for name, tensor in sd.items():
            ptr = tensor.data_ptr()
            if ptr in seen:
                sd[name] = tensor.clone()
            else:
                seen.add(ptr)

    def pull(
        self,
        model_name: str,
        out_dir: str,
        *,
        filename: str = "model.safetensors",
        verify: bool = False,
        max_shard_size: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Reconstruct every tensor in ``model_name`` and write safetensors.

        Delta-encoded blobs are transparently decompressed via ``get_tensor``.
        With ``max_shard_size`` unset, output is a single file at
        ``{out_dir}/{filename}``. With it set (bytes), the model is split into
        HF-style ``{stem}-NNNNN-of-MMMMM.safetensors`` shards plus a
        ``{filename}.index.json`` weight map; each shard is written and
        released before the next, so peak memory is bounded by the largest
        shard rather than the whole model. ``verify=True`` re-hashes every
        reconstructed tensor against its content id.
        """
        from safetensors.torch import save_file

        self._ensure_model_ready(model_name)
        mappings = self.get_model_tensors(model_name)
        if not mappings:
            raise ValueError(f"Model '{model_name}' has no tensor mappings")

        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        items = sorted(mappings.items())  # [(param_name, tid)]

        # One decode cache for the whole model: bases shared across many
        # delta targets are reconstructed once, not once per target.
        decode_cache: Dict[str, torch.Tensor] = {}

        # --- single-file path (default) ---
        if not max_shard_size:
            state_dict: Dict[str, torch.Tensor] = {}
            total_bytes = 0
            for param_name, tid in items:
                tensor = self.get_tensor(
                    tensor_id=tid, verify=verify, _cache=decode_cache
                ).contiguous()
                state_dict[param_name] = tensor
                total_bytes += tensor.element_size() * tensor.numel()
            self._unshare_storage(state_dict)
            target_file = out_path / filename
            save_file(state_dict, str(target_file))
            logger.info(
                "pull: wrote %d tensors (%d bytes logical) → %s",
                len(state_dict),
                total_bytes,
                target_file,
            )
            return {
                "model_name": model_name,
                "output_path": str(target_file),
                "num_tensors": len(state_dict),
                "total_bytes": total_bytes,
                "shards": 1,
            }

        # --- sharded path ---
        # Pass 1: plan shard boundaries from logical sizes (no decode/IO).
        size_rows = self.metadata.select_tensors_by_ids([tid for _p, tid in items])
        size_by_tid = {
            row[0]: self._logical_bytes(row[2], shape_from_json(row[1]))
            for row in size_rows
        }
        plan: List[List[Tuple[str, str]]] = []
        current: List[Tuple[str, str]] = []
        current_bytes = 0
        for param_name, tid in items:
            size = size_by_tid.get(tid, 0)
            if current and current_bytes + size > max_shard_size:
                plan.append(current)
                current, current_bytes = [], 0
            current.append((param_name, tid))
            current_bytes += size
        if current:
            plan.append(current)

        # Pass 2: decode + write each shard, releasing it before the next.
        n_shards = len(plan)
        stem = filename[:-len(".safetensors")] if filename.endswith(".safetensors") else filename
        weight_map: Dict[str, str] = {}
        total_bytes = 0
        num_tensors = 0
        for i, shard_items in enumerate(plan, start=1):
            shard_name = f"{stem}-{i:05d}-of-{n_shards:05d}.safetensors"
            shard_dict: Dict[str, torch.Tensor] = {}
            for param_name, tid in shard_items:
                tensor = self.get_tensor(
                    tensor_id=tid, verify=verify, _cache=decode_cache
                ).contiguous()
                shard_dict[param_name] = tensor
                weight_map[param_name] = shard_name
                total_bytes += tensor.element_size() * tensor.numel()
            self._unshare_storage(shard_dict)
            save_file(shard_dict, str(out_path / shard_name))
            num_tensors += len(shard_dict)
            shard_dict.clear()

        index_path = out_path / f"{filename}.index.json"
        index_path.write_text(
            json.dumps(
                {"metadata": {"total_size": total_bytes}, "weight_map": weight_map},
                indent=2,
            )
        )
        logger.info(
            "pull: wrote %d tensors across %d shards (%d bytes logical) → %s",
            num_tensors,
            n_shards,
            total_bytes,
            out_path,
        )
        return {
            "model_name": model_name,
            "output_path": str(index_path),
            "num_tensors": num_tensors,
            "total_bytes": total_bytes,
            "shards": n_shards,
        }

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_statistics(self) -> Dict[str, Any]:
        shape_hist: Dict[str, int] = {}
        for shape_json, count in self.metadata.shape_distribution().items():
            key = str(shape_from_json(shape_json))
            shape_hist[key] = shape_hist.get(key, 0) + int(count)
        return {
            "total_tensors": int(self.metadata.count_tensors()),
            "total_models": int(self.metadata.count_models()),
            "backend": self.backend_kind,
            "shape_distribution": shape_hist,
        }


__all__ = [
    "TensorDex",
    "ModelNotReadyError",
]
