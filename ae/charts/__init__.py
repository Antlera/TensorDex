"""
TensorDex Chart Registry — auto-discovers chart modules with hot-reload.

Any Python module in this package that defines a top-level ``CHARTS``
dict will be auto-registered.  Each entry in ``CHARTS`` must have::

    {
        "chart_id": {
            "name": "...",
            "category": "...",
            "desc": "...",
            "fn":  callable(rc) -> matplotlib.figure.Figure,
        }
    }

Usage from serve.py::

    from charts import CHARTS, reload_if_changed
"""

import importlib
import os
import sys
import pkgutil

# Re-export db init for convenience
from ._db import init as init_db  # noqa: F401

# ── State ─────────────────────────────────────────────────────────────

# Merged registry
CHARTS: dict = {}

# Track file modification times for hot-reload
_mtimes: dict = {}     # filepath → last mtime
_modules: dict = {}    # module_name → module object
_pkg_dir = os.path.dirname(__file__)


_COLORS_PATH = os.path.join(_pkg_dir, "_colors.py")


def _scan_chart_files():
    """Return list of (module_name, filepath) for all chart modules."""
    result = []
    for finder, name, is_pkg in pkgutil.iter_modules([_pkg_dir]):
        if name.startswith("_"):
            continue
        filepath = os.path.join(_pkg_dir, f"{name}.py")
        if os.path.exists(filepath):
            result.append((name, filepath))
    return result


def _load_all():
    """(Re)load all chart modules and rebuild the CHARTS registry."""
    CHARTS.clear()
    _mtimes.clear()

    # Reload _colors first so chart modules pick up fresh COLORS
    colors_fqn = f"{__package__}._colors"
    if colors_fqn in sys.modules:
        importlib.reload(sys.modules[colors_fqn])
    if os.path.exists(_COLORS_PATH):
        _mtimes[_COLORS_PATH] = os.path.getmtime(_COLORS_PATH)

    for name, filepath in _scan_chart_files():
        fqn = f"{__package__}.{name}"

        # If already imported, reload; otherwise import fresh. A module may fail
        # to import when its (optional) staged data / research-monorepo deps are
        # absent — skip it and keep the rest of the registry usable.
        try:
            if fqn in sys.modules:
                mod = importlib.reload(sys.modules[fqn])
            else:
                mod = importlib.import_module(f".{name}", __package__)
        except Exception as e:  # noqa: BLE001
            print(f"note: chart module '{name}' skipped (needs the research "
                  f"repo — Fig 6, out of scope for the AE cache): "
                  f"{str(e).splitlines()[0][:80]}")
            continue

        _modules[name] = mod
        _mtimes[filepath] = os.path.getmtime(filepath)

        mod_charts = getattr(mod, "CHARTS", None)
        if isinstance(mod_charts, dict):
            for cid, info in mod_charts.items():
                if cid in CHARTS:
                    print(f"⚠️  Duplicate chart ID '{cid}' from '{name}', overwriting")
                CHARTS[cid] = info


def _check_changed():
    """Return True if any chart or _colors module file was modified since last load."""
    current_files = _scan_chart_files()

    # Also track _colors.py
    all_paths = {fp for _, fp in current_files}
    if os.path.exists(_COLORS_PATH):
        all_paths.add(_COLORS_PATH)

    # New or removed files
    tracked_paths = set(_mtimes.keys())
    if all_paths != tracked_paths:
        return True

    # Modified files
    for filepath in all_paths:
        try:
            mtime = os.path.getmtime(filepath)
        except OSError:
            return True
        if filepath not in _mtimes or mtime != _mtimes[filepath]:
            return True

    return False


def reload_if_changed():
    """Check for file changes and reload if needed.  Returns True if reloaded."""
    if _check_changed():
        n_before = len(CHARTS)
        _load_all()
        n_after = len(CHARTS)
        print(f"🔄 Hot-reload: {n_after} charts "
              f"({'unchanged' if n_before == n_after else f'{n_before}→{n_after}'})")
        return True
    return False


# ── Initial load on import ────────────────────────────────────────────
_load_all()
