# src/relorbit_py/simulate.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import yaml
import relorbit_py as rp


def load_cases_yaml(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"YAML inválido: raiz deve ser dict. path={path}")
    return cfg


def _get_solver_field(case: Dict[str, Any], key: str, default: Any = None) -> Any:
    if isinstance(case.get("solver"), dict) and key in case["solver"]:
        return case["solver"][key]
    if key in case:
        return case[key]
    return default


def _get_span(case: Dict[str, Any]) -> Tuple[float, float]:
    if "span" in case:
        a, b = case["span"]
        return float(a), float(b)
    if "t0" in case and "tf" in case:
        return float(case["t0"]), float(case["tf"])
    if "tau0" in case and "tauf" in case:
        return float(case["tau0"]), float(case["tauf"])
    raise KeyError(
        f"span ausente no caso '{case.get('name','<sem-nome>')}'. "
        "Esperado case.span=[a,b] (novo) ou (t0,tf)/(tau0,tauf) (legado)."
    )


def _make_solver_cfg(case: Dict[str, Any]) -> Any:
    eng = rp.get_engine()
    cfg = eng.SolverCfg()

    dt = _get_solver_field(case, "dt", None)
    if dt is None:
        raise KeyError(
            f"dt ausente no caso '{case.get('name','<sem-nome>')}'. "
            "Esperado em case.solver.dt (novo) ou case.dt (legado)."
        )
    cfg.dt = float(dt)

    n_steps = _get_solver_field(case, "n_steps", 0)
    cfg.n_steps = int(n_steps) if n_steps is not None else 0
    return cfg


def _parse_pr0(case: Dict[str, Any]) -> float:
    """
    Regras:
      - Se existir case["pr0"] -> usa.
      - Senão, se existir case["params"]["pr0"] -> usa.
      - Senão, se existir radial_dir -> aplica sinal em um "pr0_mag" se existir, senão 0.0.
      - Se nada existir -> 0.0 (default legado).
    """
    params = case.get("params", {}) or {}

    pr0_raw = case.get("pr0", None)
    if pr0_raw is None:
        pr0_raw = params.get("pr0", None)

    if pr0_raw is not None:
        return float(pr0_raw)

    radial_dir = case.get("radial_dir", None)
    if radial_dir is None:
        radial_dir = params.get("radial_dir", None)

    pr0_mag_raw = case.get("pr0_mag", None)
    if pr0_mag_raw is None:
        pr0_mag_raw = params.get("pr0_mag", None)

    pr0_mag = float(pr0_mag_raw) if pr0_mag_raw is not None else 0.0

    if radial_dir is None:
        return 0.0

    rd = radial_dir
    if isinstance(rd, (int, float)):
        return -abs(pr0_mag) if float(rd) < 0 else abs(pr0_mag)

    if isinstance(rd, str):
        s = rd.strip().lower()
        if s in ("in", "inbound", "inward", "plunge", "capture", "neg", "negative", "-"):
            return -abs(pr0_mag)
        if s in ("out", "outbound", "outward", "pos", "positive", "+"):
            return abs(pr0_mag)

    # Não reconhecido -> default legado
    return 0.0


def simulate_case(case: Dict[str, Any], suite_name: str) -> Any:
    eng = rp.get_engine()

    model = case.get("model", suite_name)
    cfg = _make_solver_cfg(case)
    a0, af = _get_span(case)

    if model == "newton":
        params = case.get("params", {}) or {}
        mu = float(params.get("mu", case.get("mu", 1.0)))
        state0 = case["state0"]  # [x,y,vx,vy]
        t0, tf = a0, af
        return eng.simulate_newton_rk4(mu, state0, t0, tf, cfg)

    if model == "schwarzschild":
        if not hasattr(eng, "simulate_schwarzschild_equatorial_rk4"):
            avail = [n for n in dir(eng) if "schw" in n.lower() or "schwarz" in n.lower()]
            raise AttributeError(
                "Engine não expõe simulate_schwarzschild_equatorial_rk4. "
                f"Encontradas parecidas: {avail}"
            )

        params = case.get("params", {}) or {}
        M = float(params.get("M", case.get("M", 1.0)))
        E = float(params.get("E", case.get("E")))
        L = float(params.get("L", case.get("L")))

        state0 = case.get("state0", None)
        if not isinstance(state0, list) or len(state0) < 2:
            raise ValueError(
                "Para Schwarzschild, state0 deve ser lista com pelo menos [r0, phi0]. "
                f"Recebido: {state0}"
            )

        r0 = float(state0[0])
        phi0 = float(state0[1])

        pr0 = _parse_pr0(case)

        tau0, tauf = a0, af

        capture_r = float(params.get("capture_r", 2.0))
        capture_eps = float(params.get("capture_eps", 1e-12))

        # Nova assinatura: ... r0, phi0, pr0, tau0, tauf, cfg, ...
        return eng.simulate_schwarzschild_equatorial_rk4(
            M, E, L, r0, phi0, pr0, tau0, tauf, cfg, capture_r, capture_eps
        )

    raise ValueError(f"Modelo desconhecido no caso '{case.get('name','<sem-nome>')}': {model}")
