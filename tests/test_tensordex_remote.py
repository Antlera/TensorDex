from __future__ import annotations

import socket
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import pytest

# Server-only deps (the optional `[server]` group). Skip this module cleanly if
# they're absent so the default `make test` isn't broken at collection time.
uvicorn = pytest.importorskip("uvicorn")
pytest.importorskip("fastapi")

import requests
import torch
from fastapi.testclient import TestClient
from safetensors.torch import load_file, save_file

from tensordex.client.remote import pull_remote
from tensordex.core.engine import TensorDex
from tensordex.server import build_app


def _build_source_hub(tmp_path: Path) -> TensorDex:
    shard = tmp_path / "source.safetensors"
    save_file(
        {
            "model.embed.weight": torch.arange(12, dtype=torch.float32).reshape(3, 4),
            "model.norm.weight": torch.ones(4, dtype=torch.float16),
        },
        str(shard),
    )

    hub = TensorDex(str(tmp_path / "source_hub"), hydrate_all=False)
    hub.init_model("org/tiny")
    hub.ingest([str(shard)], "org/tiny")
    hub.commit_model("org/tiny")
    return hub


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@contextmanager
def _live_server(hub: TensorDex) -> Iterator[str]:
    port = _free_port()
    server = uvicorn.Server(
        uvicorn.Config(
            build_app(hub),
            host="127.0.0.1",
            port=port,
            log_level="warning",
            lifespan="off",
        )
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    base_url = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        try:
            if requests.get(f"{base_url}/healthz", timeout=0.2).status_code == 200:
                break
        except requests.RequestException:
            time.sleep(0.05)
    else:
        server.should_exit = True
        thread.join(timeout=5)
        raise RuntimeError("Test server did not start")

    try:
        yield base_url
    finally:
        server.should_exit = True
        thread.join(timeout=5)


def test_server_manifest_and_blob_routes(tmp_path: Path) -> None:
    hub = _build_source_hub(tmp_path)
    client = TestClient(
        build_app(hub, blobs_base_url="http://transfer.example/api/v1/blobs/")
    )

    models = client.get("/api/v1/models")
    assert models.status_code == 200
    assert models.json()["models"] == [
        {
            "name": "org/tiny",
            "status": "ready",
            "total_tensors": 2,
            "updated_at": hub.get_model_state("org/tiny")["updated_at"],
        }
    ]

    manifest = client.get("/api/v1/models/org/tiny/manifest")
    assert manifest.status_code == 200
    body = manifest.json()
    assert body["model_name"] == "org/tiny"
    assert body["blobs_base_url"] == "http://transfer.example/api/v1/blobs"
    assert len(body["params"]) == 2
    assert len(body["blobs"]) == 2

    tensor_id = body["blobs"][0]["tensor_id"]
    head = client.head(f"/api/v1/blobs/{tensor_id}")
    assert head.status_code == 200
    assert head.headers["etag"] == f'"{tensor_id}"'
    assert int(head.headers["content-length"]) == body["blobs"][0]["size_bytes"]

    cached = client.get(
        f"/api/v1/blobs/{tensor_id}",
        headers={"If-None-Match": f'"{tensor_id}"'},
    )
    assert cached.status_code == 304


def test_remote_pull_downloads_registers_and_reuses_cache(tmp_path: Path) -> None:
    source_hub = _build_source_hub(tmp_path)
    cache_hub = TensorDex(str(tmp_path / "cache_hub"), hydrate_all=False)
    output_dir = tmp_path / "out"

    with _live_server(source_hub) as endpoint:
        first = pull_remote(
            "org/tiny",
            endpoint=endpoint,
            local_hub=cache_hub,
            output_dir=str(output_dir),
            filename="first.safetensors",
            workers=2,
        )
        second = pull_remote(
            f"{endpoint}/api/v1/models/org/tiny",
            endpoint=None,
            local_hub=cache_hub,
            output_dir=str(output_dir),
            filename="second.safetensors",
            workers=2,
        )

    assert first["blobs_downloaded"] == 2
    assert first["blobs_cached"] == 0
    assert second["blobs_downloaded"] == 0
    assert second["blobs_cached"] == 2

    pulled = load_file(str(output_dir / "second.safetensors"))
    assert sorted(pulled) == ["model.embed.weight", "model.norm.weight"]
    assert torch.equal(
        pulled["model.embed.weight"],
        torch.arange(12, dtype=torch.float32).reshape(3, 4),
    )


def test_server_rejects_path_traversal_tensor_id(tmp_path: Path) -> None:
    hub = _build_source_hub(tmp_path)
    client = TestClient(build_app(hub))

    response = client.get("/api/v1/blobs/../metadata.db")
    assert response.status_code in {400, 404}
