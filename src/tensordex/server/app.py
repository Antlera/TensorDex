"""FastAPI app exposing a TensorDex as a read-only model repo.

Contract (client-facing):

- ``GET  /api/v1/models``
    List every ready model with lightweight summary info.

- ``GET  /api/v1/models/{model_path:path}/manifest``
    Return the download manifest for one model — everything a client
    needs to reconstruct the model, including the recursive base chain
    for any delta-encoded blob.

- ``GET  /api/v1/blobs/{tensor_id}``
    Stream the raw blob bytes. Supports ``ETag`` + ``If-None-Match``
    for client-side cache revalidation (the tensor_id **is** the ETag,
    since blobs are content-addressed).

- ``HEAD /api/v1/blobs/{tensor_id}``
    Existence check + ``Content-Length``. Lets the client decide
    whether to download before issuing the GET.

Server state is a single ``TensorDex`` instance held in FastAPI app
state. Only the read path is exposed — no ingest, no compress, no
delete is reachable over HTTP.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

try:
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import FileResponse, JSONResponse, Response
except ImportError as _exc:  # pragma: no cover — informative error for bare install
    raise ImportError(
        "tensordex.server requires fastapi. Install with `pip install tensordex[server]`."
    ) from _exc

from tensordex.core.storage import LocalStorageBackend

if TYPE_CHECKING:  # pragma: no cover
    from tensordex.core.engine import TensorDex


logger = logging.getLogger(__name__)

API_PREFIX = "/api/v1"


# ---------------------------------------------------------------------------
# Manifest construction
# ---------------------------------------------------------------------------


def _collect_needed_blobs(
    hub: TensorDex,
    backend: LocalStorageBackend,
    direct_tids: List[str],
) -> List[Dict[str, Any]]:
    """Close over the delta base chain and describe every blob to fetch.

    The closure plus each blob's codec / base / shape / dtype all come from
    SQL (the ``tensor_deltas`` graph via ``manifest_blobs``) — no safetensors
    header is parsed. Only the exact on-disk byte count, which the client
    verifies after download, still needs a cheap ``stat()``.
    """
    out: List[Dict[str, Any]] = []
    for tid, _sql_size, shape_json, dtype, codec, base_id in hub.metadata.manifest_blobs(
        direct_tids
    ):
        path = backend.blob_path_for_id(tid)
        if not path.exists():
            legacy = backend._legacy_blob_path(tid)
            if legacy.exists():
                path = legacy
            else:
                raise FileNotFoundError(f"Blob missing for {tid}")
        entry: Dict[str, Any] = {
            "tensor_id": tid,
            "size_bytes": int(path.stat().st_size),
            "target_shape": [int(x) for x in json.loads(shape_json)] if shape_json else [],
            "target_dtype": dtype,
            "is_compressed": codec is not None,
        }
        if codec is not None:
            entry["codec"] = codec
            if base_id:
                entry["base_tensor_id"] = base_id
        out.append(entry)
    return out


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def build_app(
    hub: TensorDex,
    *,
    blobs_base_url: Optional[str] = None,
    transfer_port: Optional[int] = None,
) -> FastAPI:
    """Return a FastAPI instance serving ``hub``."""
    if not isinstance(hub.storage_backend, LocalStorageBackend):
        raise RuntimeError(
            "Server currently requires a local-backend hub; S3 backend not yet wired."
        )

    backend: LocalStorageBackend = hub.storage_backend
    app = FastAPI(
        title="TensorDex Server",
        version="0.1.0",
        description="Read-only HTTP API exposing a TensorDex as a model repo.",
    )
    app.state.hub = hub

    # -- models index -----------------------------------------------------

    @app.get(f"{API_PREFIX}/models")
    def list_models() -> Dict[str, Any]:
        rows = hub.ls()
        return {
            "models": [
                {
                    "name": r["model_name"],
                    "status": r["status"],
                    "total_tensors": r["total_tensors"],
                    "updated_at": r["updated_at"],
                }
                for r in rows
                if r["status"] == "ready"
            ]
        }

    # -- per-model manifest ----------------------------------------------

    def _resolve_blobs_base_url(request: Request) -> Optional[str]:
        if blobs_base_url:
            return blobs_base_url.rstrip("/")
        if transfer_port is None:
            return None
        hostname = request.url.hostname or request.client.host
        return f"{request.url.scheme}://{hostname}:{transfer_port}{API_PREFIX}/blobs"

    @app.get(f"{API_PREFIX}/models/{{model_path:path}}/manifest")
    def get_manifest(model_path: str, request: Request) -> Dict[str, Any]:
        state = hub.get_model_state(model_path)
        if state is None:
            raise HTTPException(status_code=404, detail=f"Unknown model: {model_path}")
        if state.get("status") != "ready":
            raise HTTPException(
                status_code=409,
                detail=f"Model {model_path!r} is not ready (status={state['status']})",
            )

        mappings = hub.get_model_tensors(model_path)
        direct_tids = list(set(mappings.values()))
        try:
            blobs = _collect_needed_blobs(hub, backend, direct_tids)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=500, detail=str(exc))

        manifest = {
            "model_name": state["model_name"],
            "status": state["status"],
            "total_tensors": state["total_tensors"],
            "created_at": state["created_at"],
            "updated_at": state["updated_at"],
            "params": [
                {"param": p, "tensor_id": mappings[p]} for p in sorted(mappings)
            ],
            "blobs": blobs,
        }
        resolved_blobs_base = _resolve_blobs_base_url(request)
        if resolved_blobs_base:
            manifest["blobs_base_url"] = resolved_blobs_base
        return manifest

    # -- blob stream ------------------------------------------------------

    def _resolve_blob_path(tensor_id: str) -> Path:
        if "/" in tensor_id or ".." in tensor_id:
            raise HTTPException(status_code=400, detail="Invalid tensor_id")
        path = backend.blob_path_for_id(tensor_id)
        if not path.exists():
            legacy = backend._legacy_blob_path(tensor_id)
            if legacy.exists():
                return legacy
            raise HTTPException(status_code=404, detail=f"Blob not found: {tensor_id}")
        return path

    @app.head(f"{API_PREFIX}/blobs/{{tensor_id}}")
    def head_blob(tensor_id: str) -> Response:
        path = _resolve_blob_path(tensor_id)
        size = path.stat().st_size
        return Response(
            status_code=200,
            headers={
                "Content-Length": str(size),
                "ETag": f'"{tensor_id}"',
                "Accept-Ranges": "bytes",
                "Content-Type": "application/octet-stream",
            },
        )

    @app.get(f"{API_PREFIX}/blobs/{{tensor_id}}")
    def get_blob(tensor_id: str, request: Request) -> Response:
        path = _resolve_blob_path(tensor_id)
        etag = f'"{tensor_id}"'
        if_none_match = request.headers.get("if-none-match")
        if if_none_match and etag in {v.strip() for v in if_none_match.split(",")}:
            return Response(status_code=304, headers={"ETag": etag})

        # FileResponse already handles sendfile + Range requests via starlette.
        return FileResponse(
            path=str(path),
            media_type="application/octet-stream",
            headers={"ETag": etag, "Accept-Ranges": "bytes"},
        )

    # -- health -----------------------------------------------------------

    @app.get("/healthz")
    def healthz() -> JSONResponse:
        return JSONResponse({"status": "ok", "hub": str(hub.storage_dir)})

    return app
