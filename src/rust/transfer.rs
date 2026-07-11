use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use axum::body::Body;
use axum::extract::{Path as AxumPath, State};
use axum::http::header;
use axum::http::{HeaderMap, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use axum::Router;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use tokio::fs::File;
use tokio::io::{AsyncReadExt, AsyncSeekExt, SeekFrom};
use tokio::net::TcpListener;
use tokio_util::io::ReaderStream;

#[derive(Clone)]
struct TransferState {
    root_dir: Arc<PathBuf>,
}

fn canonical_blob_path(root: &Path, tensor_id: &str) -> PathBuf {
    let p1 = tensor_id.get(0..2).unwrap_or("00");
    let p2 = tensor_id.get(2..4).unwrap_or("00");
    root.join("blobs")
        .join(p1)
        .join(p2)
        .join(format!("{tensor_id}.safetensors"))
}

fn legacy_blob_path(root: &Path, tensor_id: &str) -> PathBuf {
    let p1 = tensor_id.get(0..2).unwrap_or("00");
    root.join("blobs")
        .join(p1)
        .join(format!("{tensor_id}.safetensors"))
}

fn validate_tensor_id(tensor_id: &str) -> Result<(), StatusCode> {
    if tensor_id.is_empty()
        || tensor_id.contains('/')
        || tensor_id.contains('\\')
        || tensor_id.contains("..")
    {
        return Err(StatusCode::BAD_REQUEST);
    }
    Ok(())
}

async fn resolve_blob_path(root: &Path, tensor_id: &str) -> Result<PathBuf, StatusCode> {
    validate_tensor_id(tensor_id)?;
    let canonical = canonical_blob_path(root, tensor_id);
    if tokio::fs::metadata(&canonical).await.is_ok() {
        return Ok(canonical);
    }
    let legacy = legacy_blob_path(root, tensor_id);
    if tokio::fs::metadata(&legacy).await.is_ok() {
        return Ok(legacy);
    }
    Err(StatusCode::NOT_FOUND)
}

fn common_headers(tensor_id: &str, len: u64) -> HeaderMap {
    let mut headers = HeaderMap::new();
    headers.insert(
        header::CONTENT_LENGTH,
        HeaderValue::from_str(&len.to_string()).unwrap(),
    );
    headers.insert(
        header::ETAG,
        HeaderValue::from_str(&format!("\"{tensor_id}\"")).unwrap(),
    );
    headers.insert(header::ACCEPT_RANGES, HeaderValue::from_static("bytes"));
    headers.insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("application/octet-stream"),
    );
    headers
}

fn parse_range(range: Option<&HeaderValue>, len: u64) -> Result<Option<(u64, u64)>, StatusCode> {
    let Some(raw) = range else {
        return Ok(None);
    };
    let value = raw
        .to_str()
        .map_err(|_| StatusCode::RANGE_NOT_SATISFIABLE)?;
    let Some(spec) = value.strip_prefix("bytes=") else {
        return Err(StatusCode::RANGE_NOT_SATISFIABLE);
    };
    if spec.contains(',') {
        return Err(StatusCode::RANGE_NOT_SATISFIABLE);
    }
    let Some((start_raw, end_raw)) = spec.split_once('-') else {
        return Err(StatusCode::RANGE_NOT_SATISFIABLE);
    };

    let (start, end) = if start_raw.is_empty() {
        let suffix = end_raw
            .parse::<u64>()
            .map_err(|_| StatusCode::RANGE_NOT_SATISFIABLE)?;
        if suffix == 0 {
            return Err(StatusCode::RANGE_NOT_SATISFIABLE);
        }
        let start = len.saturating_sub(suffix);
        (start, len.saturating_sub(1))
    } else {
        let start = start_raw
            .parse::<u64>()
            .map_err(|_| StatusCode::RANGE_NOT_SATISFIABLE)?;
        let end = if end_raw.is_empty() {
            len.saturating_sub(1)
        } else {
            end_raw
                .parse::<u64>()
                .map_err(|_| StatusCode::RANGE_NOT_SATISFIABLE)?
        };
        (start, end.min(len.saturating_sub(1)))
    };

    if len == 0 || start >= len || start > end {
        return Err(StatusCode::RANGE_NOT_SATISFIABLE);
    }
    Ok(Some((start, end)))
}

async fn head_blob(
    State(state): State<TransferState>,
    AxumPath(tensor_id): AxumPath<String>,
) -> Response {
    let path = match resolve_blob_path(&state.root_dir, &tensor_id).await {
        Ok(path) => path,
        Err(status) => return status.into_response(),
    };
    match tokio::fs::metadata(path).await {
        Ok(meta) => (StatusCode::OK, common_headers(&tensor_id, meta.len())).into_response(),
        Err(_) => StatusCode::NOT_FOUND.into_response(),
    }
}

async fn get_blob(
    State(state): State<TransferState>,
    AxumPath(tensor_id): AxumPath<String>,
    headers: HeaderMap,
) -> Response {
    let path = match resolve_blob_path(&state.root_dir, &tensor_id).await {
        Ok(path) => path,
        Err(status) => return status.into_response(),
    };
    let meta = match tokio::fs::metadata(&path).await {
        Ok(meta) => meta,
        Err(_) => return StatusCode::NOT_FOUND.into_response(),
    };
    let len = meta.len();
    let etag = format!("\"{tensor_id}\"");
    if let Some(if_none_match) = headers.get(header::IF_NONE_MATCH) {
        if if_none_match
            .to_str()
            .map(|v| v.split(',').any(|part| part.trim() == etag))
            .unwrap_or(false)
        {
            let mut out = HeaderMap::new();
            out.insert(header::ETAG, HeaderValue::from_str(&etag).unwrap());
            return (StatusCode::NOT_MODIFIED, out).into_response();
        }
    }

    let range = match parse_range(headers.get(header::RANGE), len) {
        Ok(range) => range,
        Err(status) => return status.into_response(),
    };

    let mut file = match File::open(path).await {
        Ok(file) => file,
        Err(_) => return StatusCode::NOT_FOUND.into_response(),
    };
    let (status, body_len, content_range) = if let Some((start, end)) = range {
        if file.seek(SeekFrom::Start(start)).await.is_err() {
            return StatusCode::INTERNAL_SERVER_ERROR.into_response();
        }
        (
            StatusCode::PARTIAL_CONTENT,
            end - start + 1,
            Some(format!("bytes {start}-{end}/{len}")),
        )
    } else {
        (StatusCode::OK, len, None)
    };

    let stream = ReaderStream::new(file.take(body_len));
    let mut response = (status, Body::from_stream(stream)).into_response();
    let headers_mut = response.headers_mut();
    for (name, value) in common_headers(&tensor_id, body_len) {
        if let Some(name) = name {
            headers_mut.insert(name, value);
        }
    }
    if let Some(value) = content_range {
        headers_mut.insert(
            header::CONTENT_RANGE,
            HeaderValue::from_str(&value).unwrap(),
        );
    }
    response
}

async fn healthz() -> &'static str {
    "ok"
}

async fn serve_transfer_async(root_dir: PathBuf, host: String, port: u16) -> anyhow::Result<()> {
    let state = TransferState {
        root_dir: Arc::new(root_dir),
    };
    let app = Router::new()
        .route("/healthz", get(healthz))
        .route("/api/v1/blobs/{tensor_id}", get(get_blob).head(head_blob))
        .with_state(state);
    let addr: SocketAddr = format!("{host}:{port}").parse()?;
    let listener = TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}

#[pyfunction]
pub fn serve_transfer(root_dir: String, host: String, port: u16) -> PyResult<()> {
    let root = PathBuf::from(root_dir);
    if !root.exists() {
        return Err(PyValueError::new_err(format!(
            "Storage directory does not exist: {}",
            root.display()
        )));
    }
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;
    runtime
        .block_on(serve_transfer_async(root, host, port))
        .map_err(|err| PyRuntimeError::new_err(err.to_string()))
}
