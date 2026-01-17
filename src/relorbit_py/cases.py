# src/relorbit_py/cases.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import yaml


@dataclass(frozen=True)
class SolverSpec:
    method: str = "RK4"
    dt: float = 1.0e-3
    n_steps: int = 0


@dataclass(frozen=True)
class Case:
    name: str
    model: str
    params: Dict[str, Any]
    state0: List[float]
    span: Tuple[float, float]
    solver: SolverSpec
    expected: Dict[str, Any]


def _as_float_list(x: Any, n: Optional[int] = None) -> List[float]:
    if not isinstance(x, (list, tuple)):
        raise ValueError(f"Expected list/tuple, got {type(x)}")
    out = [float(v) for v in x]
    if n is not None and len(out) != n:
        raise ValueError(f"Expected {n} elements, got {len(out)}")
    return out


def _as_span(x: Any) -> Tuple[float, float]:
    v = _as_float_list(x, n=2)
    return (v[0], v[1])


def _load_yaml(path: Union[str, Path]) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"YAML not found: {p.resolve()}")
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("YAML root must be a mapping/dict")
    return data


def load_cases(path: Union[str, Path] = "validation_cases.yaml") -> List[Case]:
    data = _load_yaml(path)
    raw_cases = data.get("cases", [])
    if not isinstance(raw_cases, list):
        raise ValueError("YAML must contain key 'cases' as a list")

    cases: List[Case] = []
    for i, rc in enumerate(raw_cases):
        if not isinstance(rc, dict):
            raise ValueError(f"Case #{i} must be a dict")

        name = str(rc["name"])
        model = str(rc["model"]).strip().lower()

        params = rc.get("params", {})
        if not isinstance(params, dict):
            raise ValueError(f"{name}: params must be a dict")

        state0 = _as_float_list(rc["state0"])
        span = _as_span(rc["span"])

        solver_raw = rc.get("solver", {}) or {}
        if not isinstance(solver_raw, dict):
            raise ValueError(f"{name}: solver must be a dict")
        solver = SolverSpec(
            method=str(solver_raw.get("method", "RK4")),
            dt=float(solver_raw.get("dt", 1.0e-3)),
            n_steps=int(solver_raw.get("n_steps", 0)),
        )

        expected = rc.get("expected", {}) or {}
        if not isinstance(expected, dict):
            raise ValueError(f"{name}: expected must be a dict")

        cases.append(
            Case(
                name=name,
                model=model,
                params=params,
                state0=state0,
                span=span,
                solver=solver,
                expected=expected,
            )
        )

    return cases
