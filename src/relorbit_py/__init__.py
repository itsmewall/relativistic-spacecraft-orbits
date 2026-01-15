# src/relorbit_py/__init__.py
from __future__ import annotations

from . import cases  # noqa: F401

try:
    from . import _engine as engine  # type: ignore
except Exception as e:  # pragma: no cover
    engine = None  # type: ignore
    _engine_import_error = e  # type: ignore


def engine_hello() -> str:
    if engine is None:
        raise RuntimeError(f"C++ engine not available: {_engine_import_error!r}")
    return engine.hello()
