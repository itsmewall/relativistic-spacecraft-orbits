# src/relorbit_py/cases.py
# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import yaml


@dataclass(frozen=True)
class SolverCfg:
    # Para Newton (por enquanto): RK4 fixo no C++
    method: str = "RK4"
    dt: float = 1.0e-3
    n_steps: int = 0  # se 0, calcula via (tf-t0)/dt

    # Reservado para o futuro (RK45/solve_ivp-like)
    rtol: float = 1.0e-10
    atol: float = 1.0e-12
    max_step: Optional[float] = None


@dataclass(frozen=True)
class ValidationCase:
    name: str
    model: str  # "newton" | "schwarzschild" (por enquanto só newton)
    params: Mapping[str, Any]
    state0: Tuple[float, ...]
    span: Tuple[float, float]
    solver: SolverCfg
    expected: Mapping[str, Any]


_ALLOWED_MODELS = {"newton", "schwarzschild"}


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(msg)


def _as_float_list(x: Any, field: str) -> List[float]:
    _require(isinstance(x, list), f"Campo '{field}' deve ser uma lista.")
    _require(len(x) > 0, f"Campo '{field}' não pode ser vazio.")
    out: List[float] = []
    for i, v in enumerate(x):
        _require(isinstance(v, (int, float)), f"Campo '{field}[{i}]' deve ser numérico.")
        out.append(float(v))
    return out


def _parse_solver(d: Mapping[str, Any]) -> SolverCfg:
    _require(isinstance(d, dict), "Campo 'solver' deve ser um dicionário.")
    method = str(d.get("method", "RK4"))

    dt = float(d.get("dt", 1.0e-3))
    _require(dt > 0.0, "solver.dt deve ser > 0.")

    n_steps = int(d.get("n_steps", 0))
    _require(n_steps >= 0, "solver.n_steps deve ser >= 0.")

    rtol = float(d.get("rtol", 1.0e-10))
    atol = float(d.get("atol", 1.0e-12))
    _require(rtol > 0.0 and atol > 0.0, "solver.rtol e solver.atol devem ser > 0.")

    max_step = d.get("max_step", None)
    if max_step is not None:
        _require(isinstance(max_step, (int, float)), "solver.max_step deve ser numérico ou null.")
        max_step = float(max_step)
        _require(max_step > 0.0, "solver.max_step deve ser > 0.")

    return SolverCfg(method=method, dt=dt, n_steps=n_steps, rtol=rtol, atol=atol, max_step=max_step)


def _validate_case_raw(c: Mapping[str, Any]) -> None:
    _require(isinstance(c, dict), "Cada caso deve ser um dicionário.")
    _require("name" in c, "Caso sem 'name'.")
    _require("model" in c, f"Caso '{c.get('name','?')}' sem 'model'.")
    _require("params" in c, f"Caso '{c.get('name','?')}' sem 'params'.")
    _require("state0" in c, f"Caso '{c.get('name','?')}' sem 'state0'.")
    _require("span" in c, f"Caso '{c.get('name','?')}' sem 'span'.")
    _require("solver" in c, f"Caso '{c.get('name','?')}' sem 'solver'.")
    _require("expected" in c, f"Caso '{c.get('name','?')}' sem 'expected'.")

    name = c["name"]
    _require(isinstance(name, str) and name.strip(), "Campo 'name' deve ser string não vazia.")

    model = c["model"]
    _require(isinstance(model, str), f"Campo 'model' do caso '{name}' deve ser string.")
    _require(model in _ALLOWED_MODELS, f"model inválido no caso '{name}': {model}")

    _require(isinstance(c["params"], dict), f"params do caso '{name}' deve ser dicionário.")
    _require(isinstance(c["expected"], dict), f"expected do caso '{name}' deve ser dicionário.")

    span = _as_float_list(c["span"], "span")
    _require(len(span) == 2, f"span do caso '{name}' deve ter 2 valores [t0, tf].")
    _require(span[1] > span[0], f"span do caso '{name}' deve ter tf>t0.")

    state0 = _as_float_list(c["state0"], "state0")

    # valida solver
    _ = _parse_solver(c["solver"])

    if model == "newton":
        _require(len(state0) == 4, f"state0 do caso '{name}' (newton) deve ter 4 valores [x,y,vx,vy].")
        _require("mu" in c["params"], f"params do caso '{name}' (newton) deve conter 'mu'.")
        _require(float(c["params"]["mu"]) > 0.0, f"mu do caso '{name}' deve ser > 0.")

    if model == "schwarzschild":
        # reservado para depois
        _require(len(state0) == 4, f"state0 do caso '{name}' (schwarzschild) deve ter 4 valores [r,pr,phi,t].")
        for k in ("M", "E", "L"):
            _require(k in c["params"], f"params do caso '{name}' (schwarzschild) deve conter '{k}'.")
        M = float(c["params"]["M"])
        _require(M > 0.0, f"M do caso '{name}' deve ser > 0.")
        r0 = float(state0[0])
        _require(r0 > 2.0 * M, f"r0 do caso '{name}' deve ser > 2M (fora do horizonte).")


def _to_case(c: Mapping[str, Any]) -> ValidationCase:
    _validate_case_raw(c)

    name = str(c["name"])
    model = str(c["model"])
    params = dict(c["params"])
    state0 = tuple(float(v) for v in c["state0"])
    t0, tf = (float(c["span"][0]), float(c["span"][1]))
    solver = _parse_solver(c["solver"])
    expected = dict(c["expected"])

    return ValidationCase(
        name=name,
        model=model,
        params=params,
        state0=state0,
        span=(t0, tf),
        solver=solver,
        expected=expected,
    )


def load_cases(path: str | Path = "validation_cases.yaml") -> List[ValidationCase]:
    path = Path(path)
    _require(path.exists(), f"Arquivo não encontrado: {path}")

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    _require(isinstance(data, dict), "YAML inválido: raiz deve ser um dicionário.")
    _require("cases" in data, "YAML inválido: faltou chave 'cases'.")
    _require(isinstance(data["cases"], list), "YAML inválido: 'cases' deve ser lista.")
    _require(len(data["cases"]) > 0, "YAML inválido: lista 'cases' vazia.")

    seen = set()
    out: List[ValidationCase] = []
    for raw in data["cases"]:
        case = _to_case(raw)
        _require(case.name not in seen, f"Nome de caso duplicado: {case.name}")
        seen.add(case.name)
        out.append(case)

    return out


def split_cases(cases: List[ValidationCase]) -> Dict[str, List[ValidationCase]]:
    groups: Dict[str, List[ValidationCase]] = {m: [] for m in _ALLOWED_MODELS}
    for c in cases:
        groups[c.model].append(c)
    return groups
