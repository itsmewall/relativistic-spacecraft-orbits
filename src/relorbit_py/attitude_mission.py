# relorbit_py/attitude_mission.py
#
# Módulo de missão para dinâmica de atitude (Item 7).
# Constrói, executa e valida simulações de atitude a partir do motor C++.
#
# Mantido separado de run_mission.py para não sobrecarregar esse ficheiro.
#
# Uso típico:
#   from relorbit_py.attitude_mission import run_attitude_mission, validate_attitude
#   result = run_attitude_mission(cfg_dict)
#   report = validate_attitude(result)

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

def _engine():
    """Lazy loader — devolve o módulo C++ ou levanta RuntimeError."""
    try:
        from relorbit_py import get_engine
        return get_engine()
    except Exception as e:
        raise RuntimeError(f"motor C++ (_engine) não disponível: {e}") from e


# ─────────────────────────────────────────────────────────────────
# AttitudeMissionCfg — descrição completa de uma missão de atitude
# ─────────────────────────────────────────────────────────────────
@dataclass
class AttitudeMissionCfg:
    name: str = "attitude_mission"
    description: str = ""

    # Estado inicial
    q0: float = 1.0    # parte escalar do quaternion (identidade por defeito)
    q1: float = 0.0
    q2: float = 0.0
    q3: float = 0.0
    wx: float = 0.0    # velocidade angular inicial [rad/s ou adim.]
    wy: float = 0.0
    wz: float = 0.0

    # Tensor de inércia (body frame)
    Ixx: float = 100.0
    Iyy: float = 200.0
    Izz: float = 150.0
    Ixy: float = 0.0
    Ixz: float = 0.0
    Iyz: float = 0.0

    # Torque externo
    tx: float = 0.0
    ty: float = 0.0
    tz: float = 0.0
    torque_t_on:  float = 0.0
    torque_t_off: float = 0.0   # 0 => sem torque

    # Intervalo de integração
    t0: float = 0.0
    tf: float = 100.0

    # Configuração do integrador
    dt:           float = 0.01
    record_every: int   = 1
    renorm_every: int   = 1
    renorm_tol:   float = 1e-9


# ─────────────────────────────────────────────────────────────────
# AttitudeMissionResult
# ─────────────────────────────────────────────────────────────────
@dataclass
class AttitudeMissionResult:
    name: str
    traj: Any                   # TrajectoryAttitude do motor C++
    cfg: AttitudeMissionCfg
    metrics: Dict[str, float] = field(default_factory=dict)
    passed: bool = False
    notes: List[str] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────
# Builders — constroem os objectos do motor C++ a partir da config
# ─────────────────────────────────────────────────────────────────
def _build_state(cfg: AttitudeMissionCfg) -> Any:
    s = _engine().AttitudeState()
    s.q0, s.q1, s.q2, s.q3 = cfg.q0, cfg.q1, cfg.q2, cfg.q3
    s.wx, s.wy, s.wz        = cfg.wx, cfg.wy, cfg.wz
    return s


def _build_inertia(cfg: AttitudeMissionCfg) -> Any:
    if cfg.Ixy == 0.0 and cfg.Ixz == 0.0 and cfg.Iyz == 0.0:
        return _engine().InertiaTensor.diagonal(cfg.Ixx, cfg.Iyy, cfg.Izz)
    return _engine().InertiaTensor.full(
        cfg.Ixx, cfg.Iyy, cfg.Izz,
        cfg.Ixy, cfg.Ixz, cfg.Iyz,
    )


def _build_torque(cfg: AttitudeMissionCfg) -> Any:
    tc = _engine().TorqueCfg()
    tc.tx, tc.ty, tc.tz = cfg.tx, cfg.ty, cfg.tz
    tc.t_on  = cfg.torque_t_on
    # se t_off não foi explicitamente configurado (==0), sem torque
    tc.t_off = cfg.torque_t_off if cfg.torque_t_off > cfg.torque_t_on else cfg.torque_t_on
    return tc


def _build_attitude_cfg(cfg: AttitudeMissionCfg) -> Any:
    ac = _engine().AttitudeCfg()
    ac.dt           = cfg.dt
    ac.n_steps      = 0            # calculado internamente
    ac.record_every = cfg.record_every
    ac.renorm_every = cfg.renorm_every
    ac.renorm_tol   = cfg.renorm_tol
    return ac


# ─────────────────────────────────────────────────────────────────
# run_attitude_mission
# ─────────────────────────────────────────────────────────────────
def run_attitude_mission(cfg: AttitudeMissionCfg) -> AttitudeMissionResult:
    """
    Executa uma missão de atitude com o motor C++.

    Retorna AttitudeMissionResult com a trajectória e métricas de validação.
    """
    traj = _engine().simulate_attitude_rk4(
        _build_state(cfg),
        _build_inertia(cfg),
        _build_torque(cfg),
        cfg.t0, cfg.tf,
        _build_attitude_cfg(cfg),
    )

    result = AttitudeMissionResult(name=cfg.name, traj=traj, cfg=cfg)
    result.metrics, result.passed, result.notes = _compute_metrics(traj, cfg)
    return result


# ─────────────────────────────────────────────────────────────────
# _compute_metrics — calcula e valida os critérios do Item 7
# ─────────────────────────────────────────────────────────────────
def _compute_metrics(
    traj: Any,
    cfg: AttitudeMissionCfg,
) -> Tuple[Dict[str, float], bool, List[str]]:
    """
    Critério (a): ‖q‖ = 1 com renormalização controlada.
    Critério (b): T_rot constante quando τ = 0.

    Retorna (metrics_dict, passed, notes).
    """
    metrics: Dict[str, float] = {}
    notes:   List[str] = []

    if traj.status != "OK" or not traj.t:
        notes.append(f"integração falhou: {traj.message}")
        return metrics, False, notes

    qnorm = traj.qnorm
    T_rot = traj.T_rot

    # ── Critério (a): norma do quaternion ─────────────────────
    max_qnorm_err = max(abs(n - 1.0) for n in qnorm)
    metrics["max_qnorm_error"]  = max_qnorm_err
    metrics["final_qnorm"]      = qnorm[-1]

    QNORM_TOL = 1e-6
    qnorm_ok  = max_qnorm_err < QNORM_TOL
    if qnorm_ok:
        notes.append(f"(a) PASS  ‖q‖ erro máx = {max_qnorm_err:.2e} < {QNORM_TOL:.0e}")
    else:
        notes.append(f"(a) FAIL  ‖q‖ erro máx = {max_qnorm_err:.2e} ≥ {QNORM_TOL:.0e}")

    # ── Critério (b): conservação de energia (só sem torque) ───
    has_torque = (
        abs(cfg.tx) + abs(cfg.ty) + abs(cfg.tz) > 0.0
        and cfg.torque_t_off > cfg.torque_t_on
    )
    if not has_torque:
        T0    = T_rot[0]
        T_max = max(T_rot)
        T_min = min(T_rot)
        T_rel_drift = (T_max - T_min) / abs(T0) if abs(T0) > 1e-30 else float("inf")
        metrics["T_rot_0"]         = T0
        metrics["T_rot_final"]     = T_rot[-1]
        metrics["T_rot_rel_drift"] = T_rel_drift

        T_TOL = 1e-6
        T_ok  = T_rel_drift < T_TOL
        if T_ok:
            notes.append(f"(b) PASS  T_rot drift relativo = {T_rel_drift:.2e} < {T_TOL:.0e}")
        else:
            notes.append(f"(b) FAIL  T_rot drift relativo = {T_rel_drift:.2e} ≥ {T_TOL:.0e}")
        passed = qnorm_ok and T_ok
    else:
        notes.append("(b) SKIP  torque activo — conservação de energia não verificada")
        passed = qnorm_ok

    metrics["n_steps_integrated"] = float(len(traj.t))
    return metrics, passed, notes


# ─────────────────────────────────────────────────────────────────
# validate_attitude — interface de alto nível (para run_mission.py)
# ─────────────────────────────────────────────────────────────────
def validate_attitude(result: AttitudeMissionResult) -> Dict[str, Any]:
    """
    Formata o resultado de uma missão de atitude como dict de relatório,
    compatível com o formato de report.json do projecto.
    """
    return {
        "name":    result.name,
        "status":  "PASS" if result.passed else "FAIL",
        "metrics": result.metrics,
        "notes":   result.notes,
        "traj_status":  result.traj.status  if result.traj else "N/A",
        "traj_message": result.traj.message if result.traj else "N/A",
    }


# ─────────────────────────────────────────────────────────────────
# from_yaml_dict — constrói AttitudeMissionCfg a partir de um dict YAML
# ─────────────────────────────────────────────────────────────────
def from_yaml_dict(d: Dict[str, Any]) -> AttitudeMissionCfg:
    """
    Exemplo de estrutura YAML esperada:

      - name: "Atitude_Torque_Nulo"
        model: "attitude"
        inertia: {Ixx: 100, Iyy: 200, Izz: 150}
        state0: {q0: 1, q1: 0, q2: 0, q3: 0, wx: 0.1, wy: 0.05, wz: 0.0}
        torque: {tx: 0, ty: 0, tz: 0}
        span: [0.0, 200.0]
        solver: {dt: 0.01, record_every: 5, renorm_every: 1}
    """
    cfg = AttitudeMissionCfg()
    cfg.name        = d.get("name", "attitude_mission")
    cfg.description = d.get("description", "")

    span = d.get("span", [0.0, 100.0])
    cfg.t0, cfg.tf = float(span[0]), float(span[1])

    s = d.get("state0", {})
    cfg.q0 = float(s.get("q0", 1.0))
    cfg.q1 = float(s.get("q1", 0.0))
    cfg.q2 = float(s.get("q2", 0.0))
    cfg.q3 = float(s.get("q3", 0.0))
    cfg.wx = float(s.get("wx", 0.0))
    cfg.wy = float(s.get("wy", 0.0))
    cfg.wz = float(s.get("wz", 0.0))

    I = d.get("inertia", {})
    cfg.Ixx = float(I.get("Ixx", 100.0))
    cfg.Iyy = float(I.get("Iyy", 200.0))
    cfg.Izz = float(I.get("Izz", 150.0))
    cfg.Ixy = float(I.get("Ixy", 0.0))
    cfg.Ixz = float(I.get("Ixz", 0.0))
    cfg.Iyz = float(I.get("Iyz", 0.0))

    T = d.get("torque", {})
    cfg.tx = float(T.get("tx", 0.0))
    cfg.ty = float(T.get("ty", 0.0))
    cfg.tz = float(T.get("tz", 0.0))
    cfg.torque_t_on  = float(T.get("t_on",  0.0))
    cfg.torque_t_off = float(T.get("t_off", 0.0))

    sol = d.get("solver", {})
    cfg.dt           = float(sol.get("dt",           0.01))
    cfg.record_every = int  (sol.get("record_every", 1))
    cfg.renorm_every = int  (sol.get("renorm_every", 1))
    cfg.renorm_tol   = float(sol.get("renorm_tol",   1e-9))

    return cfg