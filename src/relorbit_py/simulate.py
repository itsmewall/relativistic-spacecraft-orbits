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


def _pick_pr0(case: Dict[str, Any], params: Dict[str, Any]) -> float:
    # prioridade: case.pr0 > params.pr0 > radial_dir > 0.0
    if "pr0" in case:
        return float(case["pr0"])
    if "pr0" in params:
        return float(params["pr0"])

    radial_dir = case.get("radial_dir", params.get("radial_dir", None))
    if radial_dir is None:
        return 0.0

    rd = str(radial_dir).strip().lower()
    if rd in ("in", "inbound", "fall", "plunge", "-1", "neg", "negative"):
        return -0.02
    if rd in ("out", "outbound", "+1", "pos", "positive"):
        return +0.02

    raise ValueError(f"radial_dir inválido no caso '{case.get('name','<sem-nome>')}': {radial_dir}")


def simulate_case(case: Dict[str, Any], suite_name: str) -> Any:
    eng = rp.get_engine()

    model = case.get("model", suite_name)
    cfg = _make_solver_cfg(case)
    a0, af = _get_span(case)

    if model == "newton":
        params = case.get("params", {}) or {}
        mu = float(params.get("mu", case.get("mu", 1.0)))
        state0 = case["state0"]
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

        pr0 = _pick_pr0(case, params)
        tau0, tauf = a0, af

        # Lê do YAML (preferência: params)
        capture_r = float(params.get("capture_r", case.get("capture_r", 2.0)))
        capture_eps = float(params.get("capture_eps", case.get("capture_eps", 1e-12)))

        # CHAMADA POR KEYWORD: evita erro de ordem e evita cair em defaults sem querer
        # Retorno do engine já inclui tcoord (novo) e demais campos.
        return eng.simulate_schwarzschild_equatorial_rk4(
            M=M,
            E=E,
            L=L,
            r0=r0,
            phi0=phi0,
            pr0=pr0,
            tau0=tau0,
            tauf=tauf,
            cfg=cfg,
            capture_r=capture_r,
            capture_eps=capture_eps,
        )

    raise ValueError(f"Modelo desconhecido no caso '{case.get('name','<sem-nome>')}': {model}")
