"""HTTP client for pulling a model from a remote TensorDex server.

The flow mirrors HuggingFace's snapshot_download but at the
**tensor-id** granularity:

1. Fetch the model's manifest from the server.
2. Download every blob listed in the manifest — but skip any that
   already sit at the canonical path in the local hub with matching
   size (and, for compressed blobs, matching codec + base_tensor_id).
3. Register every downloaded blob + the model's mappings in the local
   hub's SQLite so subsequent local reads see the model as if it had
   been ingested locally.
4. Delegate to ``hub.pull`` for the actual safetensors assembly — the
   codec-aware loader transparently handles decompression, including
   recursively chasing base chains that we also downloaded.
"""

from __future__ import annotations

import logging
import os
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional
from urllib.parse import urlparse, urlunparse

import requests

from tensordex.core.codec import probe_codec, read_base_id
from tensordex.core.storage import LocalStorageBackend

if TYPE_CHECKING:  # pragma: no cover
    from rich.console import Console

    from tensordex.core.engine import TensorDex


logger = logging.getLogger(__name__)

_MANIFEST_SUFFIX = "/manifest"
_API_PREFIX = "/api/v1"


# ---------------------------------------------------------------------------
# URL handling
# ---------------------------------------------------------------------------


def _resolve_manifest_url(ref: str, endpoint: Optional[str]) -> tuple[str, str, str]:
    """Return ``(manifest_url, blobs_base_url, model_name)``.

    Accepts three input shapes (see ``tensordex pull`` docstring):

    1. Full manifest URL — ``http://host/api/v1/models/{model}/manifest``
    2. Full model URL without ``/manifest`` — ``http://host/api/v1/models/{model}``
    3. Bare model name + separate endpoint — ``org/model`` + ``http://host``
    """
    if ref.startswith(("http://", "https://")):
        parsed = urlparse(ref)
        path = parsed.path
        if path.endswith(_MANIFEST_SUFFIX):
            manifest_path = path
            model_path = path[: -len(_MANIFEST_SUFFIX)]
        else:
            model_path = path.rstrip("/")
            manifest_path = f"{model_path}{_MANIFEST_SUFFIX}"

        prefix_marker = f"{_API_PREFIX}/models/"
        if prefix_marker not in model_path:
            raise ValueError(
                f"URL does not look like a TensorDex model URL: {ref!r}"
            )
        model_name = model_path.split(prefix_marker, 1)[1]
        base_parts = parsed._replace(path="", params="", query="", fragment="")
        base = urlunparse(base_parts)
        manifest_url = urlunparse(parsed._replace(path=manifest_path, query="", fragment=""))
        blobs_base = f"{base}{_API_PREFIX}/blobs"
        return manifest_url, blobs_base, model_name

    if not endpoint:
        raise ValueError(
            f"Cannot resolve {ref!r} — pass a full URL or set --endpoint."
        )
    endpoint = endpoint.rstrip("/")
    manifest_url = f"{endpoint}{_API_PREFIX}/models/{ref}/manifest"
    blobs_base = f"{endpoint}{_API_PREFIX}/blobs"
    return manifest_url, blobs_base, ref


# ---------------------------------------------------------------------------
# Blob download
# ---------------------------------------------------------------------------


def _is_cached(
    backend: LocalStorageBackend, blob: Dict[str, Any]
) -> bool:
    """Return True if the local hub already has this exact blob.

    Match criterion: canonical path exists, size matches, and — for
    compressed blobs — the header's codec / base_tensor_id agrees with
    the manifest. Mismatch is conservatively treated as a miss.
    """
    tid = blob["tensor_id"]
    path = backend.blob_path_for_id(tid)
    if not path.exists():
        legacy = backend._legacy_blob_path(tid)
        if legacy.exists():
            path = legacy
        else:
            return False
    try:
        if path.stat().st_size != int(blob["size_bytes"]):
            return False
    except OSError:
        return False

    expected_compressed = bool(blob.get("is_compressed"))
    try:
        codec = probe_codec(path)
    except Exception:  # noqa: BLE001 — treat a bad header as a miss
        return False

    if expected_compressed != (codec is not None):
        return False
    if expected_compressed:
        if codec != blob.get("codec"):
            return False
        expected_base = blob.get("base_tensor_id")
        if expected_base is not None:
            try:
                actual_base = read_base_id(path)
            except Exception:  # noqa: BLE001
                return False
            if actual_base != expected_base:
                return False
    return True


def _download_blob(
    blobs_base: str,
    backend: LocalStorageBackend,
    blob: Dict[str, Any],
    chunk_size: int = 1 << 20,
) -> int:
    """Download one blob streamingly into the canonical hub path. Returns bytes read."""
    tid = blob["tensor_id"]
    dest = backend.blob_path_for_id(tid)
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = f"{blobs_base}/{tid}"

    tmp_fd, tmp_path = tempfile.mkstemp(
        dir=str(dest.parent), prefix=f"{dest.name}.", suffix=".part"
    )
    os.close(tmp_fd)
    tmp_path = Path(tmp_path)

    n_bytes = 0
    try:
        with requests.get(url, stream=True, timeout=60) as resp:
            if resp.status_code != 200:
                raise RuntimeError(
                    f"GET {url} returned {resp.status_code}: {resp.text[:200]}"
                )
            with open(tmp_path, "wb") as fh:
                for chunk in resp.iter_content(chunk_size=chunk_size):
                    if not chunk:
                        continue
                    fh.write(chunk)
                    n_bytes += len(chunk)
        expected = int(blob["size_bytes"])
        if n_bytes != expected:
            raise RuntimeError(
                f"Short read for {tid}: got {n_bytes} bytes, expected {expected}"
            )
        os.replace(tmp_path, dest)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)

    return n_bytes


# ---------------------------------------------------------------------------
# Registration — insert blobs + mappings into the local hub metadata
# ---------------------------------------------------------------------------


def _register_blobs_locally(
    hub: TensorDex,
    model_name: str,
    manifest: Dict[str, Any],
) -> None:
    """Mirror the remote manifest into the local SQLite via ingest_batch.

    Tensor rows carry no fingerprint — fingerprints are a server-side
    asset that we don't ship over the wire today. Pull (and assembly)
    don't need them; features that do (e.g. ``similar``) would need a
    separate ``/api/v1/fingerprints/{tid}`` endpoint in a follow-up.
    """
    backend = hub.storage_backend
    assert isinstance(backend, LocalStorageBackend)

    now = datetime.now(timezone.utc).isoformat()
    existing = {
        tid for tid, _uri in hub.metadata.existing_tensor_ids(
            [b["tensor_id"] for b in manifest["blobs"]]
        )
    }

    tensor_rows = []
    for blob in manifest["blobs"]:
        tid = blob["tensor_id"]
        if tid in existing:
            continue
        shape = blob.get("target_shape") or []
        dtype = blob.get("target_dtype", "")
        shape_json = __import__("json").dumps([int(x) for x in shape])
        size_bytes = int(blob["size_bytes"])
        storage_uri = backend.blob_path_for_id(tid).relative_to(backend.root_dir).as_posix()
        tensor_rows.append(
            (tid, shape_json, dtype, size_bytes, storage_uri, None, now)
        )

    mapping_rows = [
        (model_name, p["param"], p["tensor_id"], now) for p in manifest["params"]
    ]

    hub.init_model(model_name)
    hub.metadata.ingest_batch(tensor_rows, mapping_rows)
    hub.commit_model(model_name)


# ---------------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------------


def pull_remote(
    ref: str,
    *,
    endpoint: Optional[str],
    local_hub: TensorDex,
    output_dir: str,
    filename: str = "model.safetensors",
    workers: int = 8,
    verify: bool = False,
    max_shard_size: Optional[int] = None,
    console: Optional[Console] = None,
) -> Dict[str, Any]:
    """Download a model from a remote TensorDex server + assemble it locally."""
    total_start = time.perf_counter()
    manifest_url, blobs_base, model_name = _resolve_manifest_url(ref, endpoint)

    if console is not None:
        console.print(f"Manifest: [dim]{manifest_url}[/dim]")
    with requests.Session() as session:
        resp = session.get(manifest_url, timeout=30)
        if resp.status_code != 200:
            raise RuntimeError(
                f"Manifest fetch failed ({resp.status_code}): {resp.text[:300]}"
            )
        manifest = resp.json()
        blobs_base = str(manifest.get("blobs_base_url") or blobs_base).rstrip("/")

        if not isinstance(local_hub.storage_backend, LocalStorageBackend):
            raise RuntimeError("Remote pull requires a local backend as the cache.")
        backend = local_hub.storage_backend

        to_fetch: List[Dict[str, Any]] = []
        cached = 0
        for blob in manifest["blobs"]:
            if _is_cached(backend, blob):
                cached += 1
            else:
                to_fetch.append(blob)

        total_remote_bytes = sum(int(b["size_bytes"]) for b in to_fetch)
        if console is not None:
            console.print(
                f"Blobs: {len(manifest['blobs'])} total, "
                f"{cached} cached, {len(to_fetch)} to download "
                f"({_humanize_bytes(total_remote_bytes)} over wire)"
            )

        bytes_downloaded = 0
        download_start = time.perf_counter()
        if to_fetch:
            with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
                futures = {
                    ex.submit(_download_blob, blobs_base, backend, blob): blob
                    for blob in to_fetch
                }
                for fut in as_completed(futures):
                    blob = futures[fut]
                    try:
                        bytes_downloaded += fut.result()
                    except Exception as exc:
                        raise RuntimeError(
                            f"Blob {blob['tensor_id']} failed: {exc}"
                        ) from exc
        download_seconds = time.perf_counter() - download_start

    # Now that every blob is on disk at the canonical path, register the
    # model locally so hub.pull can assemble through the normal code path.
    _register_blobs_locally(local_hub, model_name, manifest)

    assemble_start = time.perf_counter()
    assemble = local_hub.pull(
        model_name,
        output_dir,
        filename=filename,
        verify=verify,
        max_shard_size=max_shard_size,
    )
    assemble_seconds = time.perf_counter() - assemble_start
    return {
        **assemble,
        "manifest_url": manifest_url,
        "blobs_total": len(manifest["blobs"]),
        "blobs_downloaded": len(to_fetch),
        "blobs_cached": cached,
        "bytes_downloaded": bytes_downloaded,
        "download_seconds": download_seconds,
        "assemble_seconds": assemble_seconds,
        "total_seconds": time.perf_counter() - total_start,
    }


def _humanize_bytes(n: int) -> str:
    size = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024 or unit == "TiB":
            return f"{int(size)} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TiB"
