# src/relorbit_py/simulate.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import yaml
import relorbit_py as rp


def load_cases_yaml(path: str | Path) -> Dict[str, Any]:
    """Carrega o arquivo de configuração (cases.yaml ou mission.yaml)."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"YAML inválido: raiz deve ser dict. path={path}")
    return cfg


def _get_solver_field(case: Dict[str, Any], key: str, default: Any = None) -> Any:
    """Busca campos no bloco 'solver' ou na raiz do caso para retrocompatibilidade."""
    if isinstance(case.get("solver"), dict) and key in case["solver"]:
        return case["solver"][key]
    if key in case:
        return case[key]
    return default


def _get_span(case: Dict[str, Any]) -> Tuple[float, float]:
    """Extrai o intervalo de tempo (span) suportando formatos novos e legados."""
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


def _setup_maneuvers(case: Dict[str, Any], cfg: Any) -> None:
    """
    Traduz a lista de manobras do YAML para objetos C++ Maneuver.
    Essencial para o planejamento de missões (Item 4 do Plano de Ação).
    """
    eng = rp.get_engine()
    # Busca manobras no bloco solver ou na raiz do caso
    maneuver_list = _get_solver_field(case, "maneuvers", [])
    
    if not isinstance(maneuver_list, list):
        return

    for m_data in maneuver_list:
        m = eng.Maneuver()
        m.tau = float(m_data.get("tau", 0.0))
        m.dv_r = float(m_data.get("dv_r", 0.0))
        m.dv_phi = float(m_data.get("dv_phi", 0.0))
        cfg.maneuvers.append(m)


def _make_solver_cfg(case: Dict[str, Any]) -> Any:
    """Cria e configura o objeto SolverCfg para a engine C++."""
    eng = rp.get_engine()
    cfg = eng.SolverCfg()

    dt = _get_solver_field(case, "dt", None)
    if dt is None:
        raise KeyError(
            f"dt ausente no caso '{case.get('name','<sem-nome>')}'. "
            "Esperado em case.solver.dt ou case.dt."
        )
    cfg.dt = float(dt)

    n_steps = _get_solver_field(case, "n_steps", 0)
    cfg.n_steps = int(n_steps) if n_steps is not None else 0
    
    # Risco 1: Stride para economia de RAM em missões longas
    re = _get_solver_field(case, "record_every", 1)
    cfg.record_every = int(re) if re is not None else 1
    
    # Item 4: Configuração de manobras impulsivas
    _setup_maneuvers(case, cfg)
    
    return cfg


def _pick_pr0(case: Dict[str, Any], params: Dict[str, Any]) -> float:
    """Define o momento radial inicial (pr0) com base em prioridades ou direção."""
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
    """Ponto de entrada principal para rodar uma simulação (Newton, Schwarzschild ou Kerr)."""
    eng = rp.get_engine()

    model = case.get("model", suite_name)
    cfg = _make_solver_cfg(case)
    a0, af = _get_span(case)

    # 1. Modelo Newtoniano
    if model == "newton":
        params = case.get("params", {}) or {}
        mu = float(params.get("mu", case.get("mu", 1.0)))
        state0 = case["state0"]
        t0, tf = a0, af
        return eng.simulate_newton_rk4(mu, state0, t0, tf, cfg)

    # 2. Modelo Schwarzschild
    if model in ("schwarzschild", "schwarzschild_equatorial"):
        params = case.get("params", {}) or {}
        M = float(params.get("M", case.get("M", 1.0)))
        E = float(params.get("E", case.get("E")))
        L = float(params.get("L", case.get("L")))

        state0 = case.get("state0", None)
        if not isinstance(state0, list) or len(state0) < 2:
            raise ValueError(f"state0 inválido para Schwarzschild em '{case.get('name')}'.")

        r0, phi0 = float(state0[0]), float(state0[1])
        pr0 = _pick_pr0(case, params)
        capture_r = float(params.get("capture_r", 2.0))
        capture_eps = float(params.get("capture_eps", 1e-12))

        return eng.simulate_schwarzschild_equatorial_rk4(
            M=M, E=E, L=L, r0=r0, phi0=phi0, pr0=pr0,
            tau0=a0, tauf=af, cfg=cfg,
            capture_r=capture_r, capture_eps=capture_eps
        )

    # 3. Modelo Kerr
    if model in ("kerr", "kerr_equatorial"):
        params = case.get("params", {}) or {}
        M = float(params.get("M", case.get("M", 1.0)))
        a = float(params.get("a", case.get("a", 0.0)))
        E = float(params.get("E", case.get("E")))
        L = float(params.get("L", case.get("L")))

        state0 = case.get("state0", None)
        if not isinstance(state0, list) or len(state0) < 2:
            raise ValueError(f"state0 inválido para Kerr em '{case.get('name')}'.")

        r0, phi0 = float(state0[0]), float(state0[1])
        pr0 = _pick_pr0(case, params)
        capture_r = float(params.get("capture_r", 2.0))
        capture_eps = float(params.get("capture_eps", 1e-12))

        return eng.simulate_kerr_equatorial_rk4(
            M=M, a=a, E=E, L=L, r0=r0, phi0=phi0, pr0=pr0,
            tau0=a0, tauf=af, cfg=cfg,
            capture_r=capture_r, capture_eps=capture_eps
        )

    raise ValueError(f"Modelo desconhecido: {model}")