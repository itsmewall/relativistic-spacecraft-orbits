# src/relorbit_py/__init__.py
from __future__ import annotations

import importlib
from typing import Optional, Any

_engine: Optional[Any] = None
_engine_import_error: Optional[BaseException] = None
_engine_tried: bool = False


def _load_engine() -> None:
    global _engine, _engine_import_error, _engine_tried
    if _engine_tried:
        return
    _engine_tried = True
    try:
        # Import robusto (não depende de import relativo)
        _engine = importlib.import_module("relorbit_py._engine")
        _engine_import_error = None
    except BaseException as e:
        _engine = None
        _engine_import_error = e


def engine_hello() -> str:
    _load_engine()
    if _engine is None:
        raise RuntimeError(
            "C++ engine not available (falha ao importar relorbit_py._engine).\n"
            f"Detalhe do import: {_engine_import_error!r}"
        )
    return _engine.hello()


def get_engine():
    _load_engine()
    if _engine is None:
        raise RuntimeError(
            "C++ engine not available (falha ao importar relorbit_py._engine).\n"
            f"Detalhe do import: {_engine_import_error!r}"
        )
    return _engine
