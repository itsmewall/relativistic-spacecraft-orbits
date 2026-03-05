# src/relorbit_py/simulate_6dof.py
"""
Módulo de missão 6-DOF acoplada: Órbita Schwarzschild + Atitude + Thrust Vectoring.

Arquitetura:
  Ao contrário das missões impulsivas (run_mission.py), aqui a atitude e a órbita
  são integradas no mesmo loop RK4 em C++. A direcção do empuxo muda a cada passo
  consoante o quaternion actual — acoplamento real.

Uso típico (YAML, model: "schwarzschild_6dof"):
  - name: "Missao_ThrustVectoring"
    model: "schwarzschild_6dof"
    params:   {M: 1.0, E: 0.95, L: 3.8}
    state0:   [10.0, 0.0]          # [r, phi]
    pr0:      0.0
    attitude0:
      q0: 1.0; q1: 0.0; q2: 0.0; q3: 0.0
      wx: 0.0; wy: 0.0; wz: 0.05   # spin lento em z
    inertia:  {Ixx: 100, Iyy: 200, Izz: 150}
    engine:
      F_newton: 50.0
      isp_s:    3000.0
      mass0_kg: 1000.0
      dry_mass_kg: 300.0
      nozzle_body: [0, 0, 1]
      tau_on:   0.0
      tau_off:  400.0
    ext_torque:
      tx: 0.0; ty: 0.0; tz: 0.0
    span:     [0.0, 600.0]
    solver:   {dt: 0.005, record_every: 20}
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np


# ── Lazy engine loader ───────────────────────────────────────────

def _engine():
    try:
        from relorbit_py import get_engine
        return get_engine()
    except Exception as e:
        raise RuntimeError(f"motor C++ (_engine) nao disponivel: {e}") from e


# ── Resultado ────────────────────────────────────────────────────

@dataclass
class Result6DOF:
    name:  str
    traj:  Any          # TrajectoryCoupled do motor C++
    cfg:   Dict[str, Any]

    def ok(self) -> bool:
        if self.traj is None:
            return False
        return str(getattr(self.traj, "status", "ERROR")) not in ("ERROR", "CAPTURE")

    def summary(self) -> Dict[str, Any]:
        t = self.traj
        if t is None or not t.tau:
            return {"status": "NO_DATA"}

        tau  = np.array(t.tau)
        mass = np.array(t.mass)
        qn   = np.array(t.qnorm)
        Tr   = np.array(t.T_rot)
        eps  = np.array(t.epsilon)
        tr   = np.array(t.thrust_r)
        tphi = np.array(t.thrust_phi)

        max_qerr = float(np.max(np.abs(qn - 1.0))) if len(qn) else float("nan")

        # FIX-A: drift robusto — evita nan quando T_rot(0)=0 (sem spin inicial)
        T0          = float(Tr[0]) if len(Tr) else 0.0
        T_abs_drift = float(np.max(Tr) - np.min(Tr)) if len(Tr) else float("nan")
        eps_T       = max(1e-30, 1e-12 * abs(T0))
        Tr_drift    = T_abs_drift / abs(T0) if abs(T0) > eps_T else float("nan")

        eps_rms  = float(np.sqrt(np.mean(eps**2))) if len(eps) else float("nan")
        dv_total = float(np.trapz(np.sqrt(tr**2 + tphi**2), tau)) if len(tr) > 1 else 0.0

        return {
            "status":           str(t.status),
            "n_points":         len(tau),
            "tau_span":         [float(tau[0]), float(tau[-1])],
            "r_range":          [float(np.min(t.r)), float(np.max(t.r))],
            "mass_consumed_kg": float(mass[0] - mass[-1]) if len(mass) else 0.0,
            "qnorm_max_err":    max_qerr,
            "T_rot0":           T0,
            "T_rot_abs_drift":  T_abs_drift,
            "T_rot_rel_drift":  Tr_drift,
            "epsilon_rms":      eps_rms,
            "dv_geom_integral": dv_total,
        }


# ── Builder: converte YAML dict → objectos C++ ──────────────────

def _build_engine(d: Dict[str, Any]) -> Any:
    eng = d.get("engine", {})
    e = _engine().EngineCfg()
    e.F_newton    = float(eng.get("F_newton",    0.0))
    e.isp_s       = float(eng.get("isp_s",       3000.0))
    e.tau_on      = float(eng.get("tau_on",       0.0))
    e.tau_off     = float(eng.get("tau_off",      1e18))
    e.mass0_kg    = float(eng.get("mass0_kg",     1000.0))
    e.dry_mass_kg = float(eng.get("dry_mass_kg",  300.0))
    nb = eng.get("nozzle_body", [0.0, 0.0, 1.0])
    e.nozzle_body = [float(nb[0]), float(nb[1]), float(nb[2])]
    tr = eng.get("torque_reaction", [0.0, 0.0, 0.0])
    e.torque_reaction = [float(tr[0]), float(tr[1]), float(tr[2])]
    return e


def _build_att0(d: Dict[str, Any]) -> Any:
    a = d.get("attitude0", {})
    s = _engine().AttitudeState()
    s.q0 = float(a.get("q0", 1.0))
    s.q1 = float(a.get("q1", 0.0))
    s.q2 = float(a.get("q2", 0.0))
    s.q3 = float(a.get("q3", 0.0))
    s.wx = float(a.get("wx", 0.0))
    s.wy = float(a.get("wy", 0.0))
    s.wz = float(a.get("wz", 0.0))
    return s


def _build_att_cfg(d: Dict[str, Any]) -> Any:
    ac = _engine().AttitudeCfg6DOF()

    I = d.get("inertia", {})
    if I.get("Ixy", 0) == 0 and I.get("Ixz", 0) == 0 and I.get("Iyz", 0) == 0:
        ac.inertia = _engine().InertiaTensor.diagonal(
            float(I.get("Ixx", 100.0)),
            float(I.get("Iyy", 200.0)),
            float(I.get("Izz", 150.0)),
        )
    else:
        ac.inertia = _engine().InertiaTensor.full(
            float(I.get("Ixx", 100.0)), float(I.get("Iyy", 200.0)), float(I.get("Izz", 150.0)),
            float(I.get("Ixy", 0.0)),  float(I.get("Ixz", 0.0)),   float(I.get("Iyz", 0.0)),
        )

    T = d.get("ext_torque", {})
    # Schwarzschild 6DOF usa set_tx/set_ty/set_tz (setter pybind11 do AttitudeCfg6DOF)
    ac.ext_torque.set_tx(float(T.get("tx", 0.0)))
    ac.ext_torque.set_ty(float(T.get("ty", 0.0)))
    ac.ext_torque.set_tz(float(T.get("tz", 0.0)))
    ac.ext_torque.t_on  = float(T.get("t_on",  0.0))
    ac.ext_torque.t_off = float(T.get("t_off", 1e18))

    sol = d.get("solver", {})
    ac.renorm_every = int(sol.get("renorm_every", 1))
    ac.renorm_tol   = float(sol.get("renorm_tol",   1e-9))
    return ac


def _build_solver(d: Dict[str, Any]) -> Any:
    sol = d.get("solver", {})
    cfg = _engine().SolverCfg6DOF()
    cfg.dt           = float(sol.get("dt",           0.005))
    cfg.n_steps      = int  (sol.get("n_steps",       0))
    cfg.record_every = int  (sol.get("record_every",  20))
    cfg.renorm_every = int  (sol.get("renorm_every",  1))
    cfg.renorm_tol   = float(sol.get("renorm_tol",    1e-9))
    cfg.capture_r    = float(d.get("params", {}).get("capture_r",   2.0))
    cfg.capture_eps  = float(d.get("params", {}).get("capture_eps", 1e-12))
    return cfg


# ── Ponto de entrada principal ───────────────────────────────────

def run_6dof_mission(m_cfg: Dict[str, Any]) -> Result6DOF:
    """Executa uma missão 6-DOF acoplada a partir de um dict de configuração YAML."""
    params = m_cfg.get("params", {})
    M  = float(params.get("M",  1.0))
    E0 = float(params.get("E",  0.95))
    L0 = float(params.get("L",  3.8))

    state0 = m_cfg.get("state0", [10.0, 0.0])
    r0   = float(state0[0])
    phi0 = float(state0[1])
    pr0  = float(m_cfg.get("pr0", 0.0))

    span = m_cfg.get("span", [0.0, 500.0])
    tau0 = float(span[0])
    tauf = float(span[1])

    eng     = _build_engine(m_cfg)
    att0    = _build_att0(m_cfg)
    att_cfg = _build_att_cfg(m_cfg)
    solver  = _build_solver(m_cfg)

    traj = _engine().simulate_schwarzschild_6dof_rk4(
        M, E0, L0, r0, phi0, pr0,
        att0, tau0, tauf,
        eng, att_cfg, solver,
    )

    return Result6DOF(name=m_cfg.get("name", "6dof"), traj=traj, cfg=m_cfg)


# ── Validação para run_mission.py ────────────────────────────────

def validate_6dof(result: Result6DOF) -> Dict[str, Any]:
    s = result.summary()
    passed = (
        s.get("status") not in ("ERROR", "CAPTURE")
        and s.get("qnorm_max_err", 1.0) < 1e-6
    )
    T_rel = s.get("T_rot_rel_drift", float("nan"))
    T_abs = s.get("T_rot_abs_drift", float("nan"))
    T0    = s.get("T_rot0", 0.0)
    trot_note = (
        f"T_rot drift = {T_rel:.2e}  [T0={T0:.3g}]"
        if not math.isnan(T_rel) else
        f"T_rot abs drift = {T_abs:.2e}  [T0≈0, sem spin inicial]"
    )
    rr = s.get("r_range", ["?", "?"])
    notes = [
        f"Orbita: r in [{rr[0]:.2f}, {rr[1]:.2f}] M",
        f"Massa consumida: {s.get('mass_consumed_kg', 0.0):.2f} kg",
        f"||q|| max err = {s.get('qnorm_max_err', float('nan')):.2e}",
        f"epsilon RMS   = {s.get('epsilon_rms', float('nan')):.2e}",
        trot_note,
    ]
    return {
        "status":  "PASS" if passed else "FAIL",
        "notes":   notes,
        "metrics": {k: v for k, v in s.items() if isinstance(v, float)},
    }