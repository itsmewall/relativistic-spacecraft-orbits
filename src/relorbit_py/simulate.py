# src/relorbit_py/simulate.py
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Tuple
import yaml
import relorbit_py as rp


def load_cases_yaml(path):
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"YAML inválido: raiz deve ser dict. path={path}")
    return cfg


def _get_solver_field(case, key, default=None):
    if isinstance(case.get("solver"), dict) and key in case["solver"]:
        return case["solver"][key]
    if key in case:
        return case[key]
    return default


def _get_span(case):
    if "span" in case:
        a, b = case["span"]; return float(a), float(b)
    if "t0" in case and "tf" in case:
        return float(case["t0"]), float(case["tf"])
    if "tau0" in case and "tauf" in case:
        return float(case["tau0"]), float(case["tauf"])
    raise KeyError(f"span ausente no caso '{case.get('name','?')}'.")


def _setup_maneuvers(case, cfg):
    eng = rp.get_engine()
    maneuver_list = _get_solver_field(case, "maneuvers", [])
    if not isinstance(maneuver_list, list):
        return
    for m_data in maneuver_list:
        m = eng.Maneuver()
        m.tau    = float(m_data.get("tau", 0.0))
        m.dv_r   = float(m_data.get("dv_r", 0.0))
        m.dv_phi = float(m_data.get("dv_phi", 0.0))
        cfg.maneuvers.append(m)


def _make_solver_cfg(case):
    eng = rp.get_engine()
    cfg = eng.SolverCfg()
    dt = _get_solver_field(case, "dt", None)
    if dt is None:
        raise KeyError(f"dt ausente no caso '{case.get('name','?')}'.")
    cfg.dt = float(dt)
    n_steps = _get_solver_field(case, "n_steps", 0)
    cfg.n_steps = int(n_steps) if n_steps else 0
    re = _get_solver_field(case, "record_every", 1)
    cfg.record_every = int(re) if re else 1
    _setup_maneuvers(case, cfg)
    return cfg


def _pick_pr0(case, params):
    if "pr0" in case:   return float(case["pr0"])
    if "pr0" in params: return float(params["pr0"])
    rd = str(case.get("radial_dir", params.get("radial_dir", "")) or "").strip().lower()
    if rd in ("in","inbound","fall","plunge","-1","neg","negative"): return -0.02
    if rd in ("out","outbound","+1","pos","positive"):               return +0.02
    return 0.0


def _make_thrust_cfg(thrust_raw: Dict[str, Any]) -> Any:
    """Traduz o bloco 'thrust:' do YAML para um objeto ThrustCfg C++."""
    eng = rp.get_engine()
    thr = eng.ThrustCfg()
    thr.F_r         = float(thrust_raw.get("F_r",         0.0))
    thr.F_phi       = float(thrust_raw.get("F_phi",       0.0))
    thr.isp_s       = float(thrust_raw.get("isp_s",       3000.0))
    thr.mass0_kg    = float(thrust_raw.get("mass0_kg",    1000.0))
    thr.dry_mass_kg = float(thrust_raw.get("dry_mass_kg", 300.0))
    thr.tau_on      = float(thrust_raw.get("tau_on",      0.0))
    thr.tau_off     = float(thrust_raw.get("tau_off",     1e18))
    mode_str = str(thrust_raw.get("mode", "CONSTANT")).upper().replace(" ", "_")
    thr.mode = getattr(eng.ThrustMode, mode_str, eng.ThrustMode.CONSTANT)
    return thr


def simulate_case(case: Dict[str, Any], suite_name: str) -> Any:
    eng = rp.get_engine()
    model  = case.get("model", suite_name)
    cfg    = _make_solver_cfg(case)
    a0, af = _get_span(case)

    # ── Newton ────────────────────────────────────────────────
    if model == "newton":
        params = case.get("params", {}) or {}
        mu     = float(params.get("mu", case.get("mu", 1.0)))
        return eng.simulate_newton_rk4(mu, case["state0"], a0, af, cfg)

    # ── Schwarzschild geodésica ───────────────────────────────
    if model in ("schwarzschild", "schwarzschild_equatorial"):
        params = case.get("params", {}) or {}
        M = float(params.get("M", case.get("M", 1.0)))
        E = float(params.get("E", case.get("E")))
        L = float(params.get("L", case.get("L")))
        state0 = case.get("state0", None)
        if not isinstance(state0, list) or len(state0) < 2:
            raise ValueError(f"state0 inválido em '{case.get('name')}'.")
        r0, phi0 = float(state0[0]), float(state0[1])
        pr0 = _pick_pr0(case, params)
        capture_r   = float(params.get("capture_r",   2.0))
        capture_eps = float(params.get("capture_eps", 1e-12))
        return eng.simulate_schwarzschild_equatorial_rk4(
            M=M, E=E, L=L, r0=r0, phi0=phi0, pr0=pr0,
            tau0=a0, tauf=af, cfg=cfg,
            capture_r=capture_r, capture_eps=capture_eps)

    # ── Kerr geodésica ────────────────────────────────────────
    if model in ("kerr", "kerr_equatorial"):
        params = case.get("params", {}) or {}
        M   = float(params.get("M", case.get("M", 1.0)))
        a   = float(params.get("a", case.get("a", 0.0)))
        E   = float(params.get("E", case.get("E")))
        L   = float(params.get("L", case.get("L")))
        state0 = case.get("state0", None)
        if not isinstance(state0, list) or len(state0) < 2:
            raise ValueError(f"state0 inválido em '{case.get('name')}'.")
        r0, phi0 = float(state0[0]), float(state0[1])
        pr0 = _pick_pr0(case, params)
        capture_r   = float(params.get("capture_r",   2.0))
        capture_eps = float(params.get("capture_eps", 1e-12))
        return eng.simulate_kerr_equatorial_rk4(
            M=M, a=a, E=E, L=L, r0=r0, phi0=phi0, pr0=pr0,
            tau0=a0, tauf=af, cfg=cfg,
            capture_r=capture_r, capture_eps=capture_eps)

    # ── Schwarzschild Low-Thrust ──────────────────────────────
    if model in ("schwarzschild_lowthrust", "schwarzschild_lt"):
        params = case.get("params", {}) or {}
        M   = float(params.get("M", 1.0))
        E   = float(params.get("E", case.get("E")))
        L   = float(params.get("L", case.get("L")))
        state0 = case.get("state0", None)
        if not isinstance(state0, list) or len(state0) < 2:
            raise ValueError(f"state0 inválido em '{case.get('name')}'.")
        r0, phi0 = float(state0[0]), float(state0[1])
        pr0 = _pick_pr0(case, params)
        capture_r   = float(params.get("capture_r",   2.0))
        capture_eps = float(params.get("capture_eps", 1e-12))
        thrust_raw  = case.get("thrust", {}) or {}
        thr = _make_thrust_cfg(thrust_raw)
        return eng.simulate_schwarzschild_lowthrust_rk4(
            M=M, E0=E, L0=L, r0=r0, phi0=phi0, pr0=pr0,
            tau0=a0, tauf=af, thrust=thr, cfg=cfg,
            capture_r=capture_r, capture_eps=capture_eps)

    # ── Kerr Low-Thrust ───────────────────────────────────────
    if model in ("kerr_lowthrust", "kerr_lt"):
        params = case.get("params", {}) or {}
        M   = float(params.get("M", 1.0))
        a   = float(params.get("a", 0.0))
        E   = float(params.get("E", case.get("E")))
        L   = float(params.get("L", case.get("L")))
        state0 = case.get("state0", None)
        if not isinstance(state0, list) or len(state0) < 2:
            raise ValueError(f"state0 inválido em '{case.get('name')}'.")
        r0, phi0 = float(state0[0]), float(state0[1])
        pr0 = _pick_pr0(case, params)
        capture_r   = float(params.get("capture_r",   2.0))
        capture_eps = float(params.get("capture_eps", 1e-12))
        thrust_raw  = case.get("thrust", {}) or {}
        thr = _make_thrust_cfg(thrust_raw)
        return eng.simulate_kerr_lowthrust_rk4(
            M=M, a=a, E0=E, L0=L, r0=r0, phi0=phi0, pr0=pr0,
            tau0=a0, tauf=af, thrust=thr, cfg=cfg,
            capture_r=capture_r, capture_eps=capture_eps)

    raise ValueError(f"Modelo desconhecido: {model}")