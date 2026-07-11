"""Internal JSON codecs for SQLite text columns."""

from __future__ import annotations

import json
from typing import Any, Dict, Optional, Tuple


def serialize_metadata(metadata: Optional[Dict[str, Any]]) -> Optional[str]:
    if metadata is None:
        return None
    try:
        return json.dumps(metadata, separators=(",", ":"))
    except (TypeError, ValueError):
        return None


def deserialize_metadata(value: Optional[str]) -> Optional[Dict[str, Any]]:
    if not value:
        return None
    try:
        decoded = json.loads(value)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    return decoded if isinstance(decoded, dict) else None


def shape_from_json(value: Optional[str]) -> Tuple[int, ...]:
    if not value:
        return ()
    try:
        data = json.loads(value)
        return tuple(int(dim) for dim in data)
    except (TypeError, ValueError, json.JSONDecodeError):
        return ()
