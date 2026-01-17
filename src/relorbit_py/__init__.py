from __future__ import annotations

import importlib
import importlib.util as iu
import sys
from typing import Optional

_engine = None
_engine_import_error: Optional[BaseException] = None


def _load_engine() -> None:
    """Carrega relorbit_py._engine de forma robusta (editable + Windows)."""
    global _engine, _engine_import_error

    if _engine is not None:
        return

    # 1) tentativa normal
    try:
        _engine = importlib.import_module("relorbit_py._engine")
        _engine_import_error = None
        return
    except BaseException as e:
        _engine = None
        _engine_import_error = e

    # 2) fallback: encontra o .pyd e força load pelo caminho
    try:
        spec = iu.find_spec("relorbit_py._engine")
        if spec is None or spec.origin is None:
            raise ImportError("find_spec('relorbit_py._engine') returned None")

        # Se já está em sys.modules mas quebrou, remove e tenta reload limpo
        sys.modules.pop("relorbit_py._engine", None)

        # Força o import usando o origin (path do .pyd)
        spec2 = iu.spec_from_file_location("relorbit_py._engine", spec.origin)
        if spec2 is None or spec2.loader is None:
            raise ImportError(f"spec_from_file_location failed for origin={spec.origin!r}")

        mod = iu.module_from_spec(spec2)
        spec2.loader.exec_module(mod)  # type: ignore[attr-defined]
        sys.modules["relorbit_py._engine"] = mod

        _engine = mod
        _engine_import_error = None
        return
    except BaseException as e:
        _engine = None
        _engine_import_error = e


def engine_hello() -> str:
    _load_engine()
    if _engine is None:
        spec = iu.find_spec("relorbit_py._engine")
        raise RuntimeError(
            "C++ engine not available. Provável causa: falha ao carregar DLL/ABI ou confusão de paths no editable.\n"
            f"find_spec('relorbit_py._engine') = {spec}\n"
            f"Detalhe do import: {_engine_import_error!r}"
        )
    return _engine.hello()


def get_engine():
    _load_engine()
    if _engine is None:
        spec = iu.find_spec("relorbit_py._engine")
        raise RuntimeError(
            "C++ engine not available. Provável causa: falha ao carregar DLL/ABI ou confusão de paths no editable.\n"
            f"find_spec('relorbit_py._engine') = {spec}\n"
            f"Detalhe do import: {_engine_import_error!r}"
        )
    return _engine
