"""``tensordex`` command-line interface.

The CLI is intentionally a thin typer layer over the engine. Every
command maps 1:1 to a ``TensorDex`` method; the CLI only owns argument
parsing and pretty-printing.

The hub location resolves in this order:

1. ``--hub`` on the command line
2. ``TENSORDEX_HOME`` env var
3. ``~/.cache/tensordex``
"""

from __future__ import annotations

import fnmatch
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import List, Optional

import typer
from rich.console import Console
from rich.table import Table

from tensordex import __version__
from tensordex.core.codec import (
    CODEC_TENSORX,
    SUPPORTED_CODECS,
)
from tensordex.core.engine import TensorDex

app = typer.Typer(
    name="tensordex",
    help="Tensor-aware content-addressable store with base+delta compression.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()
err_console = Console(stderr=True)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _default_hub_path() -> Path:
    env = os.environ.get("TENSORDEX_HOME")
    if env:
        return Path(env).expanduser()
    return Path.home() / ".cache" / "tensordex"


def _hub_option() -> Path:
    return typer.Option(
        None,
        "--hub",
        "-H",
        help="Hub directory (default: $TENSORDEX_HOME or ~/.cache/tensordex).",
    )


def _resolve_hub_path(hub: Optional[Path]) -> Path:
    return (hub or _default_hub_path()).expanduser().resolve()


def _open_hub(hub: Optional[Path], *, must_exist: bool = True) -> TensorDex:
    path = _resolve_hub_path(hub)
    if must_exist and not path.exists():
        err_console.print(
            f"[red]Hub directory not found:[/red] {path}\n"
            "Run [bold]tensordex init[/bold] first (or set TENSORDEX_HOME)."
        )
        raise typer.Exit(code=1)
    return TensorDex(str(path), hydrate_all=False)


def _humanize_bytes(n: int) -> str:
    size = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024 or unit == "TiB":
            return f"{int(size)} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TiB"


def _parse_size(text: Optional[str]) -> Optional[int]:
    """Parse a human size like '5GB' / '500MiB' / '1073741824' into bytes."""
    if not text:
        return None
    s = text.strip().upper()
    units = {
        "TIB": 2**40, "GIB": 2**30, "MIB": 2**20, "KIB": 2**10,
        "TB": 10**12, "GB": 10**9, "MB": 10**6, "KB": 10**3, "B": 1,
    }
    for suffix, mult in units.items():
        if s.endswith(suffix):
            return int(float(s[: -len(suffix)].strip()) * mult)
    return int(s)


def _dtype_nbytes(dtype: str) -> int:
    dtype = dtype.removeprefix("torch.")
    sizes = {
        "float16": 2,
        "bfloat16": 2,
        "float32": 4,
        "float64": 8,
        "int8": 1,
        "uint8": 1,
        "bool": 1,
        "int16": 2,
        "uint16": 2,
        "int32": 4,
        "uint32": 4,
        "int64": 8,
        "uint64": 8,
    }
    try:
        return sizes[dtype]
    except KeyError as exc:
        raise ValueError(f"Unsupported dtype in manifest: {dtype!r}") from exc


def _numel(shape: List[int]) -> int:
    n = 1
    for dim in shape:
        n *= int(dim)
    return n


def _manifest_logical_bytes(manifest: dict) -> int:
    blobs = {b["tensor_id"]: b for b in manifest["blobs"]}
    total = 0
    for param in manifest["params"]:
        blob = blobs[param["tensor_id"]]
        shape = [int(x) for x in blob.get("target_shape") or []]
        total += _numel(shape) * _dtype_nbytes(str(blob["target_dtype"]))
    return total


def _split_model_param(ref: str) -> tuple[str, str]:
    if ":" not in ref:
        err_console.print(
            f"[red]Invalid reference[/red] {ref!r} — expected "
            "[bold]<model>:<param>[/bold]."
        )
        raise typer.Exit(code=2)
    model, _, param = ref.partition(":")
    if not model or not param:
        err_console.print(f"[red]Invalid reference[/red] {ref!r}.")
        raise typer.Exit(code=2)
    return model, param


def _lookup_tensor_id(hub: TensorDex, ref: str) -> str:
    model, param = _split_model_param(ref)
    try:
        return hub._lookup_tensor_id_for_param(model, param)
    except KeyError as exc:
        err_console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1)


# ---------------------------------------------------------------------------
# Lifecycle commands
# ---------------------------------------------------------------------------


@app.command()
def init(
    path: Optional[Path] = typer.Argument(
        None, help="Directory to initialize. Defaults to the resolved hub path."
    ),
    force: bool = typer.Option(
        False, "--force", help="Allow initializing a non-empty directory."
    ),
) -> None:
    """Create a new TensorDex storage directory."""
    target = _resolve_hub_path(path)
    if target.exists() and any(target.iterdir()) and not force:
        err_console.print(
            f"[yellow]{target}[/yellow] already exists and is not empty.\n"
            "Use [bold]--force[/bold] to reuse it."
        )
        raise typer.Exit(code=1)
    TensorDex(str(target), hydrate_all=False)
    console.print(f"[green]Initialized TensorDex at[/green] {target}")


@app.command()
def download(
    hf_id: str = typer.Argument(..., help="HuggingFace repo id."),
    only: List[str] = typer.Option(
        None, "--only", help="Restrict ingest to these parameter names. Repeatable."
    ),
    as_name: Optional[str] = typer.Option(
        None, "--as", help="Logical name to store under (defaults to the HF id)."
    ),
    revision: Optional[str] = typer.Option(
        None,
        "--revision",
        help="Branch/tag/commit, e.g. a 'stepN' training checkpoint.",
    ),
    hub: Optional[Path] = _hub_option(),
) -> None:
    """Download a HuggingFace model and ingest its safetensors shards."""
    h = _open_hub(hub, must_exist=False)
    logical = as_name or (f"{hf_id}@{revision}" if revision else hf_id)
    console.print(f"Downloading [cyan]{hf_id}[/cyan] → [bold]{logical}[/bold]")
    result = h.download(
        hf_id, stored_model_name=logical, only=only or None, revision=revision
    )
    console.print(
        f"[green]Ingested[/green] {len(result)} tensor mapping(s) → [bold]{logical}[/bold]."
    )


@app.command()
def whoami() -> None:
    """Show the currently logged-in HuggingFace user."""
    try:
        from huggingface_hub import whoami as hf_whoami
    except ImportError:
        err_console.print("[red]huggingface_hub is not installed.[/red]")
        raise typer.Exit(code=1)
    try:
        info = hf_whoami()
    except Exception as exc:  # noqa: BLE001 — forward auth errors verbatim
        err_console.print(f"[red]Not logged in:[/red] {exc}")
        raise typer.Exit(code=1)
    name = info.get("name") or info.get("email") or "(unknown)"
    console.print(f"Logged in as [bold]{name}[/bold]")


@app.command("env")
def env_cmd(hub: Optional[Path] = _hub_option()) -> None:
    """Print environment + hub configuration."""
    hub_path = _resolve_hub_path(hub)
    table = Table(show_header=False, box=None, pad_edge=False)
    table.add_column(style="dim")
    table.add_column()
    table.add_row("tensordex version", __version__)
    table.add_row("python", platform.python_version())
    table.add_row("platform", f"{platform.system()} {platform.release()}")
    table.add_row("hub path", str(hub_path))
    table.add_row("hub exists", "yes" if hub_path.exists() else "no")
    table.add_row("TENSORDEX_HOME", os.environ.get("TENSORDEX_HOME", "(unset)"))
    table.add_row("HF_HOME", os.environ.get("HF_HOME", "(unset)"))
    console.print(table)


# ---------------------------------------------------------------------------
# Inspection commands
# ---------------------------------------------------------------------------


@app.command("ls")
def ls_cmd(
    model: Optional[str] = typer.Option(
        None, "--model", "-m", help="Glob filter on model name (fnmatch)."
    ),
    hub: Optional[Path] = _hub_option(),
) -> None:
    """List models stored in the hub."""
    h = _open_hub(hub)
    rows = h.ls()
    if model:
        rows = [r for r in rows if fnmatch.fnmatchcase(r["model_name"], model)]

    if not rows:
        console.print("[dim]No models.[/dim]")
        return

    table = Table(show_header=True, header_style="bold")
    table.add_column("model")
    table.add_column("status")
    table.add_column("tensors", justify="right")
    table.add_column("updated_at", style="dim")
    for r in rows:
        table.add_row(
            r["model_name"],
            r["status"],
            str(r["total_tensors"]),
            r["updated_at"],
        )
    console.print(table)


@app.command()
def info(
    model: str = typer.Argument(..., help="Model name."),
    show_params: bool = typer.Option(
        False, "--params", help="List every param→tensor_id mapping."
    ),
    hub: Optional[Path] = _hub_option(),
) -> None:
    """Show lifecycle + sizing info for one model."""
    h = _open_hub(hub)
    data = h.info(model)
    if data is None:
        err_console.print(f"[red]Model not found:[/red] {model}")
        raise typer.Exit(code=1)

    table = Table(show_header=False, box=None, pad_edge=False)
    table.add_column(style="dim")
    table.add_column()
    table.add_row("name", data["model_name"])
    table.add_row("status", data["status"])
    table.add_row("tensors (mapped)", str(len(data["mappings"])))
    table.add_row("tensors (unique)", str(data["unique_tensors"]))
    table.add_row("total_bytes", _humanize_bytes(data["total_bytes"]))
    table.add_row("created_at", data["created_at"])
    table.add_row("updated_at", data["updated_at"])
    console.print(table)

    if show_params:
        params_table = Table(show_header=True, header_style="bold")
        params_table.add_column("param")
        params_table.add_column("tensor_id", style="dim")
        for param, tid in sorted(data["mappings"].items()):
            params_table.add_row(param, tid)
        console.print(params_table)


@app.command()
def stats(hub: Optional[Path] = _hub_option()) -> None:
    """Show hub-wide counters and on-disk usage."""
    h = _open_hub(hub)
    s = h.get_statistics()
    path = _resolve_hub_path(hub)
    disk = shutil.disk_usage(path) if path.exists() else None

    table = Table(show_header=False, box=None, pad_edge=False)
    table.add_column(style="dim")
    table.add_column()
    table.add_row("hub path", str(path))
    table.add_row("backend", s["backend"])
    table.add_row("total_models", str(s["total_models"]))
    table.add_row("total_tensors", str(s["total_tensors"]))
    table.add_row("distinct_shapes", str(len(s["shape_distribution"])))
    if disk is not None:
        table.add_row("disk_free", _humanize_bytes(disk.free))
    console.print(table)


# ---------------------------------------------------------------------------
# Retrieval commands
# ---------------------------------------------------------------------------


@app.command()
def get(
    ref: str = typer.Argument(..., help="Tensor reference, <model>:<param>."),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Write to this .safetensors file."
    ),
    hub: Optional[Path] = _hub_option(),
) -> None:
    """Fetch a single tensor by ``<model>:<param>`` (auto-decompresses)."""
    model, param = _split_model_param(ref)
    h = _open_hub(hub)
    try:
        tensor = h.get_tensor(model_name=model, param_name=param)
    except KeyError as exc:
        err_console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1)

    if output is None:
        console.print(
            f"[bold]{ref}[/bold]  shape={tuple(tensor.shape)}  dtype={tensor.dtype}  "
            f"bytes={_humanize_bytes(tensor.element_size() * tensor.numel())}"
        )
        return

    from safetensors.torch import save_file

    output.parent.mkdir(parents=True, exist_ok=True)
    save_file({"tensor": tensor.contiguous()}, str(output))
    console.print(f"[green]Wrote[/green] {output}")


@app.command()
def pull(
    ref: str = typer.Argument(
        ...,
        help=(
            "Model name (local hub) or a full URL to a remote TensorDex server, "
            "e.g. http://host:8000/api/v1/models/unsloth/Llama-3.2-3B"
        ),
    ),
    output: Path = typer.Option(
        ..., "--output", "-o", help="Destination directory."
    ),
    filename: str = typer.Option(
        "model.safetensors",
        "--filename",
        help="Output safetensors filename inside --output.",
    ),
    endpoint: Optional[str] = typer.Option(
        None,
        "--endpoint",
        help=(
            "HF-style: combine with a bare model name ref to target a remote "
            "server, e.g. --endpoint http://host:8000."
        ),
    ),
    workers: int = typer.Option(
        8, "--workers", help="Concurrent blob downloads for remote pulls."
    ),
    verify: bool = typer.Option(
        False,
        "--verify",
        help="Re-hash each reconstructed tensor against its content id (XXH3 hubs).",
    ),
    max_shard_size: Optional[str] = typer.Option(
        None,
        "--max-shard-size",
        help="Shard the output (e.g. 5GB, 500MB) with an index.json. Default: single file.",
    ),
    hub: Optional[Path] = _hub_option(),
) -> None:
    """Materialize a model as safetensors, locally or from a remote server.

    Three input forms are accepted:

    - local model name: ``tensordex pull org/model -o out/``
    - full URL:         ``tensordex pull http://host:8000/api/v1/models/org/model -o out/``
    - endpoint + name:  ``tensordex pull org/model --endpoint http://host:8000 -o out/``

    Remote pulls download blobs into the local hub cache (at ``--hub``),
    so subsequent pulls that share base tensors skip the network.
    """
    shard_bytes = _parse_size(max_shard_size)
    is_remote = ref.startswith(("http://", "https://")) or endpoint is not None
    if is_remote:
        from tensordex.client.remote import pull_remote

        local_hub = _open_hub(hub, must_exist=False)
        try:
            result = pull_remote(
                ref,
                endpoint=endpoint,
                local_hub=local_hub,
                output_dir=str(output),
                filename=filename,
                workers=workers,
                verify=verify,
                max_shard_size=shard_bytes,
                console=console,
            )
        except Exception as exc:  # noqa: BLE001 — display and exit
            err_console.print(f"[red]{exc}[/red]")
            raise typer.Exit(code=1)
        console.print(
            f"[green]Pulled[/green] {result['num_tensors']} tensors "
            f"({_humanize_bytes(result['total_bytes'])} logical) → "
            f"[bold]{result['output_path']}[/bold]"
        )
        console.print(
            f"  network: {_humanize_bytes(result['bytes_downloaded'])} "
            f"across {result['blobs_downloaded']} blob(s), "
            f"{result['blobs_cached']} cache hit(s)"
        )
        return

    h = _open_hub(hub)
    try:
        result = h.pull(
            ref, str(output), filename=filename, verify=verify, max_shard_size=shard_bytes
        )
    except (KeyError, ValueError) as exc:
        err_console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1)
    console.print(
        f"[green]Pulled[/green] {result['num_tensors']} tensors "
        f"({_humanize_bytes(result['total_bytes'])} logical) → "
        f"[bold]{result['output_path']}[/bold]"
    )


@app.command("demo-transfer")
def demo_transfer(
    ref: str = typer.Argument(
        ...,
        help="Remote model name or full TensorDex model URL.",
    ),
    endpoint: Optional[str] = typer.Option(
        None,
        "--endpoint",
        help="Remote TensorDex control-plane endpoint for bare model names.",
    ),
    workers: int = typer.Option(
        4,
        "--workers",
        help="Concurrent blob downloads for the compressed path.",
    ),
    hub: Optional[Path] = typer.Option(
        None,
        "--hub",
        "-H",
        help="Local cache hub. Defaults to a temporary cold cache.",
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output directory. Defaults to a temporary directory.",
    ),
    temp_root: Optional[Path] = typer.Option(
        None,
        "--temp-root",
        help="Directory for temporary cache/output when --hub or --output are omitted.",
    ),
    filename: str = typer.Option(
        "model.safetensors",
        "--filename",
        help="Output safetensors filename for the compressed reconstruction.",
    ),
    keep: bool = typer.Option(
        False,
        "--keep",
        help="Keep temporary cache/output directories after the run.",
    ),
) -> None:
    """Demo full-model raw transfer vs compressed blob transfer + reconstruction."""
    import requests

    from tensordex.client.remote import _resolve_manifest_url, pull_remote

    manifest_url, _blobs_base, model_name = _resolve_manifest_url(ref, endpoint)
    console.print(f"Manifest: [dim]{manifest_url}[/dim]")

    manifest_start = time.perf_counter()
    resp = requests.get(manifest_url, timeout=30)
    if resp.status_code != 200:
        err_console.print(f"[red]Manifest fetch failed ({resp.status_code}):[/red] {resp.text[:300]}")
        raise typer.Exit(code=1)
    manifest = resp.json()
    manifest_seconds = time.perf_counter() - manifest_start

    raw_logical_bytes = _manifest_logical_bytes(manifest)
    compressed_wire_bytes = sum(int(b["size_bytes"]) for b in manifest["blobs"])
    direct_ids = {p["tensor_id"] for p in manifest["params"]}
    direct_physical_bytes = sum(
        int(b["size_bytes"]) for b in manifest["blobs"] if b["tensor_id"] in direct_ids
    )
    compressed_blobs = sum(1 for b in manifest["blobs"] if b.get("is_compressed"))
    base_blobs = len(manifest["blobs"]) - len(direct_ids)
    bandwidth_saved = (
        1.0 - (compressed_wire_bytes / raw_logical_bytes)
        if raw_logical_bytes
        else 0.0
    )

    temp_base = (temp_root or Path.cwd()).expanduser().resolve()
    temp_base.mkdir(parents=True, exist_ok=True)

    cache_tmp: Optional[Path] = None
    out_tmp: Optional[Path] = None
    if hub is None:
        cache_tmp = Path(tempfile.mkdtemp(prefix="tensordex-demo-cache.", dir=str(temp_base)))
        hub_path = cache_tmp
    else:
        hub_path = hub.expanduser().resolve()
    if output is None:
        out_tmp = Path(tempfile.mkdtemp(prefix="tensordex-demo-out.", dir=str(temp_base)))
        output_path = out_tmp
    else:
        output_path = output.expanduser().resolve()

    local_hub = TensorDex(str(hub_path), hydrate_all=False)
    try:
        result = pull_remote(
            ref,
            endpoint=endpoint,
            local_hub=local_hub,
            output_dir=str(output_path),
            filename=filename,
            workers=workers,
            console=None,
        )
    except Exception as exc:  # noqa: BLE001
        err_console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1)

    measured_wire = int(result["bytes_downloaded"])
    download_seconds = float(result["download_seconds"])
    assemble_seconds = float(result["assemble_seconds"])
    total_seconds = float(result["total_seconds"])
    measured_mib_s = (
        measured_wire / (1024 * 1024) / download_seconds
        if download_seconds > 0 and measured_wire
        else 0.0
    )
    projected_raw_seconds = (
        raw_logical_bytes / (measured_mib_s * 1024 * 1024)
        if measured_mib_s > 0
        else 0.0
    )
    projected_speedup = (
        projected_raw_seconds / total_seconds if total_seconds > 0 else 0.0
    )

    summary = Table(title=f"TensorDex Transfer Demo: {model_name}", show_header=True)
    summary.add_column("metric")
    summary.add_column("value", justify="right")
    summary.add_row("params", str(len(manifest["params"])))
    summary.add_row("manifest blobs", str(len(manifest["blobs"])))
    summary.add_row("compressed blobs", str(compressed_blobs))
    summary.add_row("base-chain blobs", str(base_blobs))
    summary.add_row("raw full-model bytes", _humanize_bytes(raw_logical_bytes))
    summary.add_row("direct physical bytes", _humanize_bytes(direct_physical_bytes))
    summary.add_row("compressed transfer bytes", _humanize_bytes(compressed_wire_bytes))
    summary.add_row("bandwidth saved", f"{bandwidth_saved * 100:.1f}%")
    console.print(summary)

    timing = Table(title="Measured Compressed Path", show_header=True)
    timing.add_column("stage")
    timing.add_column("seconds", justify="right")
    timing.add_column("detail", justify="right")
    timing.add_row("manifest", f"{manifest_seconds:.3f}", manifest_url)
    timing.add_row(
        "download",
        f"{download_seconds:.3f}",
        f"{_humanize_bytes(measured_wire)} @ {measured_mib_s:.1f} MiB/s",
    )
    timing.add_row("register + reconstruct", f"{assemble_seconds:.3f}", str(result["output_path"]))
    timing.add_row("total", f"{total_seconds:.3f}", f"{workers} worker(s)")
    timing.add_row(
        "raw projected at same throughput",
        f"{projected_raw_seconds:.3f}",
        f"{projected_speedup:.2f}x vs compressed total",
    )
    console.print(timing)

    if cache_tmp is not None and keep:
        console.print(f"Cache hub: [bold]{hub_path}[/bold]")
    if out_tmp is not None and keep:
        console.print(f"Output: [bold]{output_path}[/bold]")

    if cache_tmp is not None and not keep:
        shutil.rmtree(cache_tmp, ignore_errors=True)
    if out_tmp is not None and not keep:
        shutil.rmtree(out_tmp, ignore_errors=True)


@app.command()
def serve(
    hub: Optional[Path] = _hub_option(),
    host: str = typer.Option("127.0.0.1", "--host", help="Bind address."),
    port: int = typer.Option(8000, "--port", help="Bind port."),
    log_level: str = typer.Option(
        "info", "--log-level", help="uvicorn log level."
    ),
    transfer_backend: str = typer.Option(
        "python",
        "--transfer-backend",
        help="Blob transfer backend: python or rust.",
    ),
    transfer_port: int = typer.Option(
        8001,
        "--transfer-port",
        help="Rust blob transfer port when --transfer-backend=rust.",
    ),
    transfer_host: Optional[str] = typer.Option(
        None,
        "--transfer-host",
        help="Rust blob bind address (defaults to --host).",
    ),
    transfer_url: Optional[str] = typer.Option(
        None,
        "--transfer-url",
        help="External blobs base URL advertised in manifests.",
    ),
) -> None:
    """Serve a hub as a read-only model repo over HTTP."""
    try:
        import uvicorn  # noqa: F401

        from tensordex.server import build_app
    except ImportError:
        err_console.print(
            "[red]Missing server dependencies.[/red]\n"
            "Install with: [bold]pip install tensordex\\[server][/bold]"
        )
        raise typer.Exit(code=1)

    if transfer_backend not in {"python", "rust"}:
        err_console.print("[red]--transfer-backend must be 'python' or 'rust'.[/red]")
        raise typer.Exit(code=2)

    h = _open_hub(hub)
    hub_path = _resolve_hub_path(hub)
    transfer_proc: Optional[subprocess.Popen] = None
    if transfer_backend == "rust":
        bind_host = transfer_host or host
        transfer_proc = subprocess.Popen(
            [
                sys.executable,
                "-c",
                (
                    "from tensordex import _ops; "
                    "import sys; "
                    "_ops.serve_transfer(sys.argv[1], sys.argv[2], int(sys.argv[3]))"
                ),
                str(hub_path),
                bind_host,
                str(transfer_port),
            ]
        )
        advertised = transfer_url
        if advertised is None and host not in {"0.0.0.0", "::"}:
            advertised = f"http://{host}:{transfer_port}/api/v1/blobs"
        fastapi_app = build_app(
            h,
            blobs_base_url=advertised,
            transfer_port=None if advertised else transfer_port,
        )
    else:
        fastapi_app = build_app(h)

    console.print(
        f"Serving hub [cyan]{hub_path}[/cyan] on "
        f"[bold]http://{host}:{port}[/bold]"
    )
    if transfer_backend == "rust":
        console.print(
            f"Rust blob transfer on [bold]http://{transfer_host or host}:{transfer_port}[/bold]"
        )
    import uvicorn as _uvicorn

    try:
        _uvicorn.run(fastapi_app, host=host, port=port, log_level=log_level)
    finally:
        if transfer_proc is not None and transfer_proc.poll() is None:
            transfer_proc.terminate()
            try:
                transfer_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                transfer_proc.kill()


# ---------------------------------------------------------------------------
# Mutation commands
# ---------------------------------------------------------------------------


@app.command()
def compress(
    target: Optional[str] = typer.Argument(
        None,
        help="Target tensor <model>:<param>. Omit when using --auto.",
    ),
    base: Optional[str] = typer.Option(
        None, "--base", help="Base tensor <model>:<param> (required without --auto)."
    ),
    auto: Optional[str] = typer.Option(
        None,
        "--auto",
        help=(
            "Run FlexSplit attach planner on this model and compress every "
            "recommended (target, base) pair."
        ),
    ),
    auto_all: bool = typer.Option(
        False,
        "--auto-all",
        help="Run --auto across every ready model in the hub (idempotent).",
    ),
    bundle: Optional[str] = typer.Option(
        None,
        "--bundle",
        help=(
            "Glob over model names (fnmatch); compress them as one group "
            "(e.g. training checkpoints) — plans once, star topology, no chains."
        ),
    ),
    cr_threshold: float = typer.Option(
        0.70,
        "--cr-threshold",
        help="Attach threshold for --auto mode (max predicted CR).",
    ),
    include_existing_bases: bool = typer.Option(
        False,
        "--include-existing-bases",
        help=(
            "When planning --auto, let the new model attach against tensors "
            "already in the hub (cross-model delta)."
        ),
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Print the --auto plan without compressing."
    ),
    codec: str = typer.Option(
        CODEC_TENSORX,
        "--codec",
        help=f"Codec to use. One of {sorted(SUPPORTED_CODECS)}.",
    ),
    level: int = typer.Option(3, "--level", help="zstd level passed to the codec."),
    hub: Optional[Path] = _hub_option(),
) -> None:
    """Compress a tensor against a base, or run the attach planner over a model."""
    if codec not in SUPPORTED_CODECS:
        err_console.print(
            f"[red]Unknown codec[/red] {codec!r} — pick one of "
            f"{sorted(SUPPORTED_CODECS)}."
        )
        raise typer.Exit(code=2)

    if bundle:
        if auto or auto_all or target or base:
            err_console.print(
                "[red]--bundle is incompatible with --auto / --auto-all / target / --base.[/red]"
            )
            raise typer.Exit(code=2)
        h = _open_hub(hub)
        models = [
            r["model_name"] for r in h.ls() if fnmatch.fnmatchcase(r["model_name"], bundle)
        ]
        if not models:
            err_console.print(f"[yellow]No models match[/yellow] {bundle!r}.")
            raise typer.Exit(code=1)
        console.print(f"bundle: {len(models)} model(s) matching [cyan]{bundle}[/cyan]")
        try:
            result = h.compress_bundle(
                models, cr_threshold=cr_threshold, codec=codec, level=level, dry_run=dry_run
            )
        except (ValueError, NotImplementedError, KeyError) as exc:
            err_console.print(f"[red]{exc}[/red]")
            raise typer.Exit(code=1)
        n_ok = sum(1 for r in result.get("results", []) if r.get("status") == "ok")
        console.print(
            f"  {result['n_tensors']} tensors → {result['n_bases']} base(s) + "
            f"{result['n_pairs']} delta pair(s)"
            + ("  [dim](dry-run)[/dim]" if dry_run else "")
        )
        if result.get("executed") and result["total_compressed_bytes"] > 0:
            console.print(
                f"  [green]compressed[/green] {n_ok} pair(s): "
                f"{_humanize_bytes(result['total_original_bytes'])} → "
                f"{_humanize_bytes(result['total_compressed_bytes'])} "
                f"([bold]{result['realised_ratio']:.2f}x[/bold] realised)"
            )
        return

    if auto_all:
        if auto or target or base:
            err_console.print(
                "[red]--auto-all is incompatible with --auto / positional target / --base.[/red]"
            )
            raise typer.Exit(code=2)
        h = _open_hub(hub)
        _run_auto_compress_all(
            h,
            cr_threshold=cr_threshold,
            codec=codec,
            level=level,
            include_existing_bases=include_existing_bases,
            dry_run=dry_run,
        )
        return

    if auto:
        if target or base:
            err_console.print(
                "[red]--auto is incompatible with positional target / --base.[/red]"
            )
            raise typer.Exit(code=2)
        h = _open_hub(hub)
        try:
            result = h.auto_compress(
                auto,
                cr_threshold=cr_threshold,
                codec=codec,
                level=level,
                include_existing_bases=include_existing_bases,
                dry_run=dry_run,
            )
        except (ValueError, NotImplementedError, KeyError) as exc:
            err_console.print(f"[red]{exc}[/red]")
            raise typer.Exit(code=1)
        _render_auto_compress(result, dry_run=dry_run)
        return

    if not target or not base:
        err_console.print(
            "[red]Provide <target> and --base, or use --auto <model>, or --auto-all.[/red]"
        )
        raise typer.Exit(code=2)

    h = _open_hub(hub)
    target_id = _lookup_tensor_id(h, target)
    base_id = _lookup_tensor_id(h, base)

    try:
        result = h.compress_pair(target_id, base_id, codec=codec, level=level)
    except (ValueError, NotImplementedError, KeyError) as exc:
        err_console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1)

    console.print(
        f"[green]Compressed[/green] {target} ← {base}: "
        f"{_humanize_bytes(result['original_bytes'])} → "
        f"{_humanize_bytes(result['compressed_bytes'])} "
        f"([bold]{result['ratio']:.2f}x[/bold], {result['codec']})"
    )


def _run_auto_compress_all(
    hub,
    *,
    cr_threshold: float,
    codec: str,
    level: int,
    include_existing_bases: bool,
    dry_run: bool,
) -> None:
    """Drive ``auto_compress_all`` with a rich progress line per model."""
    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        Progress,
        TextColumn,
        TimeElapsedColumn,
    )

    models = hub.ls()
    ready = [m for m in models if m["status"] == "ready"]
    if not ready:
        console.print("[dim]No ready models to compress.[/dim]")
        return

    console.print(
        f"auto-compress-all: {len(ready)} ready model(s), "
        f"threshold={cr_threshold:.2f}, codec={codec}, "
        f"{'dry-run' if dry_run else 'execute'}"
    )

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("compressing", total=len(ready))

        def on_model(name: str, result: dict) -> None:
            ratio = result.get("realised_ratio", 0.0) if isinstance(result, dict) else 0.0
            n_ok = sum(
                1
                for r in (result.get("results", []) if isinstance(result, dict) else [])
                if r.get("status") == "ok"
            )
            progress.update(
                task,
                advance=1,
                description=f"{name[:40]:40s}  ok={n_ok:3d}  ratio={ratio:.2f}x",
            )

        summary = hub.auto_compress_all(
            cr_threshold=cr_threshold,
            codec=codec,
            level=level,
            include_existing_bases=include_existing_bases,
            dry_run=dry_run,
            progress=on_model,
        )

    console.print(
        f"[green]done[/green]: processed={summary['models_processed']} "
        f"models, pairs ok={summary['total_ok']} / "
        f"skipped={summary['total_skipped_pairs']} / "
        f"failed={summary['total_failed_pairs']}"
    )
    if summary["total_compressed_bytes"] > 0:
        console.print(
            f"  overall: {_humanize_bytes(summary['total_original_bytes'])} → "
            f"{_humanize_bytes(summary['total_compressed_bytes'])} "
            f"([bold]{summary['overall_realised_ratio']:.2f}x[/bold] realised)"
        )


def _render_auto_compress(result: dict, *, dry_run: bool) -> None:
    """Pretty-print plan output, covering both dry-run and executed modes."""
    model = result["model_name"]
    bases = result["bases"]
    pairs = result["pairs"]
    console.print(
        f"[bold]auto-compress {model}[/bold]: "
        f"{len(bases)} base(s), {len(pairs)} attach pair(s), "
        f"threshold={result['cr_threshold']:.2f}, shapes={result['n_shapes']}"
    )
    if result.get("skipped_no_fp"):
        console.print(
            f"[yellow]skipped (no fingerprint): {len(result['skipped_no_fp'])}[/yellow]"
        )
    if not pairs:
        console.print("[dim]no attach pairs recommended.[/dim]")
        return

    table = Table(show_header=True, header_style="bold")
    table.add_column("target (tid)", style="dim")
    table.add_column("base (tid)", style="dim")
    table.add_column("distance", justify="right")
    table.add_column("pred_cr", justify="right")
    if not dry_run:
        table.add_column("actual_ratio", justify="right")
        table.add_column("status")

    per_pair = {r["target_tensor_id"]: r for r in result.get("results", [])}
    for p in pairs:
        row = [
            p["target_tensor_id"][:12],
            p["base_tensor_id"][:12],
            f"{p['distance']:.4f}",
            f"{p['predicted_cr']:.4f}",
        ]
        if not dry_run:
            r = per_pair.get(p["target_tensor_id"], {})
            if r.get("status") == "ok":
                row.extend([f"{r['actual_ratio']:.2f}x", "[green]ok[/green]"])
            elif r.get("status") == "failed":
                row.extend(["—", f"[red]fail[/red]: {r.get('error', '')[:40]}"])
            else:
                row.extend(["—", "—"])
        table.add_row(*row)
    console.print(table)

    if not dry_run and result.get("executed"):
        console.print(
            f"[green]executed[/green]: "
            f"{_humanize_bytes(result['total_original_bytes'])} → "
            f"{_humanize_bytes(result['total_compressed_bytes'])} "
            f"([bold]{result['realised_ratio']:.2f}x[/bold] realised), "
            f"{result['failures']} failure(s)"
        )


@app.command()
def rm(
    model: str = typer.Argument(..., help="Model name to remove."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation."),
    hub: Optional[Path] = _hub_option(),
) -> None:
    """Remove a model's mappings (blobs survive until ``gc``)."""
    h = _open_hub(hub)
    if not yes:
        confirm = typer.confirm(
            f"Remove model '{model}'? Blobs remain until `tensordex gc`."
        )
        if not confirm:
            raise typer.Exit(code=1)
    try:
        result = h.rm(model)
    except KeyError as exc:
        err_console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1)
    console.print(
        f"[green]Removed[/green] {model}: "
        f"{result['mappings_deleted']} mapping(s), {result['meta_deleted']} meta row."
    )


@app.command()
def gc(hub: Optional[Path] = _hub_option()) -> None:
    """Reclaim tensors + blobs no model (or delta base) references."""
    h = _open_hub(hub)
    result = h.gc()
    msg = (
        f"[green]gc[/green]: {result['tensors_deleted']} tensor(s) deleted, "
        f"{result['blobs_deleted']} blob(s) unlinked, "
        f"{result['bases_protected']} base(s) protected"
    )
    if result["blob_errors"]:
        msg += f" — [yellow]{result['blob_errors']} blob error(s)[/yellow]"
    console.print(msg)


# ---------------------------------------------------------------------------


def main() -> None:
    """Console-script entry point."""
    try:
        app()
    except typer.Exit:
        raise
    except KeyboardInterrupt:
        err_console.print("\n[yellow]Aborted.[/yellow]")
        sys.exit(130)


if __name__ == "__main__":
    main()
