"""Physical blob storage backends for tensor payloads."""

from __future__ import annotations

import json
import os
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import torch
from safetensors.torch import load_file as safetensors_torch_load_file
from safetensors.torch import save_file

_SAFETENSOR_KEY = "tensor"


def _pick_tensor_from_safetensors_keys(
    keys: Sequence[str],
    get_tensor_fn,
    expected_shape: Optional[Tuple[int, ...]] = None,
) -> torch.Tensor:
    """Match Rust `select_tensor_key`: prefer ``tensor``, then shape match, then first payload key."""
    if _SAFETENSOR_KEY in keys:
        return get_tensor_fn(_SAFETENSOR_KEY)
    if expected_shape:
        exp = tuple(int(x) for x in expected_shape)
        for name in keys:
            if name.startswith("_"):
                continue
            t = get_tensor_fn(name)
            if tuple(t.shape) == exp:
                return t
    tensor_keys = [
        k
        for k in keys
        if not k.startswith("_")
        and not k.endswith("_fingerprint")
        and k != "_fingerprint"
    ]
    if tensor_keys:
        return get_tensor_fn(tensor_keys[0])
    if keys:
        return get_tensor_fn(keys[0])
    raise KeyError("Safetensors payload has no usable tensor keys")


def _load_local_safetensors_torch(
    path: Path,
    expected_shape: Optional[Tuple[int, ...]] = None,
) -> torch.Tensor:
    """Load one tensor from a local ``.safetensors`` file as ``torch.Tensor``.

    Uses ``safetensors.torch.load_file`` rather than ``safe_open(..., framework="pt")``
    because the latter raises on BF16 payloads common in LLM checkpoints.
    """
    tensors = safetensors_torch_load_file(str(path))
    keys = list(tensors.keys())
    return _pick_tensor_from_safetensors_keys(
        keys, lambda k: tensors[k], expected_shape
    ).clone()


class StorageBackend(ABC):
    """Abstract interface for physical tensor storage."""

    @abstractmethod
    def save_tensor(
        self, tensor_id: str, tensor: torch.Tensor, header: Dict[str, Any]
    ) -> str:
        """Persist tensor and return a storage URI."""

    @abstractmethod
    def load_tensor(self, storage_uri: str) -> torch.Tensor:
        """Load tensor identified by ``storage_uri``."""

    @abstractmethod
    def delete_tensor(self, storage_uri: str) -> None:
        """Remove tensor object from the backend."""


class LocalStorageBackend(StorageBackend):
    """Filesystem backed tensor blobs.

    Canonical on-disk layout is 2-level sharded: ``blobs/{id[:2]}/{id[2:4]}/{id}.safetensors``.
    All writes go to the canonical layout. Reads also accept the legacy 1-level layout
    (``blobs/{id[:2]}/{id}.safetensors``) so existing / migrated corpora keep working.
    """

    def __init__(self, root_dir: Path):
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def blob_path_for_id(self, tensor_id: str) -> Path:
        """Canonical 2-level path used for all writes. May or may not exist."""
        p1 = tensor_id[:2] or "00"
        p2 = tensor_id[2:4] or "00"
        return self.root_dir / "blobs" / p1 / p2 / f"{tensor_id}.safetensors"

    def _legacy_blob_path(self, tensor_id: str) -> Path:
        """1-level path retained for reading migrated corpora."""
        p1 = tensor_id[:2] or "00"
        return self.root_dir / "blobs" / p1 / f"{tensor_id}.safetensors"

    def _find_existing_blob(self, tensor_id: str) -> Optional[Path]:
        canonical = self.blob_path_for_id(tensor_id)
        if canonical.exists():
            return canonical
        legacy = self._legacy_blob_path(tensor_id)
        if legacy.exists():
            return legacy
        return None

    def prepare_blob_dirs(self, tensor_ids: Sequence[str]) -> None:
        """Pre-create every directory the given IDs will write into. Batch ingest helper."""
        dirs = {self.blob_path_for_id(tid).parent for tid in tensor_ids}
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)

    def save_tensor(
        self, tensor_id: str, tensor: torch.Tensor, header: Dict[str, Any]
    ) -> str:
        path = self.blob_path_for_id(tensor_id)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload = {_SAFETENSOR_KEY: tensor.detach().cpu().contiguous()}
        metadata = {_SAFETENSOR_KEY: json.dumps(header, separators=(",", ":"))}

        tmp_fd, tmp_path = tempfile.mkstemp(
            dir=str(path.parent), prefix=f"{path.name}.", suffix=".tmp"
        )
        os.close(tmp_fd)
        tmp_path = Path(tmp_path)
        try:
            save_file(payload, str(tmp_path), metadata=metadata)
            os.replace(tmp_path, path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)

        return path.relative_to(self.root_dir).as_posix()

    def load_tensor(self, storage_uri: str) -> torch.Tensor:
        path = self.root_dir / storage_uri
        if not path.exists():
            path = self._find_existing_blob(Path(storage_uri).stem)
        if path is None or not path.exists():
            raise FileNotFoundError(f"Blob not found for URI: {storage_uri}")
        return _load_local_safetensors_torch(path, expected_shape=None)

    def load_tensor_by_id(
        self,
        tensor_id: str,
        expected_shape: Optional[Tuple[int, ...]] = None,
    ) -> torch.Tensor:
        """Load using Rust ``SimpleTensorResolver`` layout (canonical 2-level, legacy 1-level)."""
        path = self._find_existing_blob(tensor_id)
        if path is None:
            raise FileNotFoundError(
                f"Blob not found for tensor_id={tensor_id} "
                f"(tried {self.blob_path_for_id(tensor_id)})"
            )
        return _load_local_safetensors_torch(path, expected_shape)

    def delete_tensor(self, storage_uri: str) -> None:
        path = self.root_dir / storage_uri
        if not path.exists():
            path = self._find_existing_blob(Path(storage_uri).stem)
        if path is not None:
            path.unlink(missing_ok=True)


class S3StorageBackend(StorageBackend):
    """S3 compatible backend storing tensors as safetensor blobs."""

    _safetensors_ops: Optional[Tuple[Any, Any]] = None

    def __init__(
        self,
        bucket: str,
        region: Optional[str] = None,
        prefix: str = "",
        **kwargs,
    ):
        try:
            import boto3
            from botocore.config import Config
            from botocore.exceptions import ClientError
        except ImportError as exc:
            raise RuntimeError(
                "S3StorageBackend requires boto3 and botocore to be installed"
            ) from exc

        self.bucket = bucket
        self.region = region
        self.prefix = prefix.strip("/") if prefix else ""
        client_kwargs = dict(kwargs)
        if "config" not in client_kwargs:
            client_kwargs["config"] = Config(
                region_name=region,
                max_pool_connections=50,
                retries={"max_attempts": 10, "mode": "standard"},
            )
        self.client = boto3.client("s3", region_name=region, **client_kwargs)
        self._client_error = ClientError

    @classmethod
    def _get_safetensors_ops(cls) -> Tuple[Any, Any]:
        if cls._safetensors_ops is None:
            from safetensors.torch import load, save
            cls._safetensors_ops = (save, load)
        return cls._safetensors_ops

    def _blob_key(self, tensor_id: str) -> str:
        shard = tensor_id[:2] or "00"
        parts = [self.prefix] if self.prefix else []
        parts.extend(["blobs", shard, f"{tensor_id}.safetensors"])
        return "/".join(parts)

    @staticmethod
    def _parse_s3_uri(storage_uri: str) -> Tuple[str, str]:
        if not storage_uri.startswith("s3://"):
            raise ValueError(f"Invalid S3 URI: {storage_uri}")
        bucket_and_key = storage_uri[5:]
        bucket, sep, key = bucket_and_key.partition("/")
        if not bucket or not sep or not key:
            raise ValueError(f"Invalid S3 URI: {storage_uri}")
        return bucket, key

    def save_tensor(
        self, tensor_id: str, tensor: torch.Tensor, header: Dict[str, Any]
    ) -> str:
        save_fn, _ = self._get_safetensors_ops()
        header = header or {}
        payload = {_SAFETENSOR_KEY: tensor.detach().cpu().contiguous()}
        metadata = {_SAFETENSOR_KEY: json.dumps(header, separators=(",", ":"))}
        body = save_fn(payload, metadata=metadata)

        key = self._blob_key(tensor_id)
        self.client.put_object(Bucket=self.bucket, Key=key, Body=body)
        return f"s3://{self.bucket}/{key}"

    def load_tensor(self, storage_uri: str) -> torch.Tensor:
        _, load_fn = self._get_safetensors_ops()
        bucket, key = self._parse_s3_uri(storage_uri)
        try:
            response = self.client.get_object(Bucket=bucket, Key=key)
        except self._client_error as exc:
            error_code = exc.response.get("Error", {}).get("Code") if hasattr(exc, "response") else None
            if error_code in {"NoSuchKey", "404"}:
                raise FileNotFoundError(f"Blob not found at {storage_uri}") from exc
            raise

        body_stream = response["Body"]
        try:
            body = body_stream.read()
        finally:
            body_stream.close()
        tensors = load_fn(body)
        keys = list(tensors.keys())
        tensor = _pick_tensor_from_safetensors_keys(
            keys, lambda k: tensors[k], expected_shape=None
        )
        return tensor.clone()

    def load_tensor_by_id(
        self,
        tensor_id: str,
        expected_shape: Optional[Tuple[int, ...]] = None,
    ) -> torch.Tensor:
        """Load using Rust ``S3TensorResolver`` key layout under this backend's prefix."""
        _, load_fn = self._get_safetensors_ops()
        object_key = self._blob_key(tensor_id)
        try:
            response = self.client.get_object(Bucket=self.bucket, Key=object_key)
        except self._client_error as exc:
            error_code = exc.response.get("Error", {}).get("Code") if hasattr(exc, "response") else None
            if error_code in {"NoSuchKey", "404"}:
                raise FileNotFoundError(
                    f"Blob not found at s3://{self.bucket}/{object_key}"
                ) from exc
            raise

        body_stream = response["Body"]
        try:
            body = body_stream.read()
        finally:
            body_stream.close()
        tensors = load_fn(body)
        keys = list(tensors.keys())
        tensor = _pick_tensor_from_safetensors_keys(
            keys, lambda k: tensors[k], expected_shape
        )
        return tensor.clone()

    def delete_tensor(self, storage_uri: str) -> None:
        bucket, key = self._parse_s3_uri(storage_uri)
        self.client.delete_object(Bucket=bucket, Key=key)
