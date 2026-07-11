"""HTTP server exposing a TensorDex as a read-only model repo.

Import ``build_app(hub)`` to get a configured FastAPI app and mount it
yourself, or use the ``tensordex serve`` CLI which wraps uvicorn.
"""

from tensordex.server.app import build_app

__all__ = ["build_app"]
