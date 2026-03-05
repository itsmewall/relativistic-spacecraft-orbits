# src/relorbit_py/simulate_kerr_6dof.py
"""
Módulo de missão Kerr 6-DOF acoplada: Órbita Kerr + Atitude + Torque de Maré.

Uso típico (YAML, model: "kerr_6dof"):
  - name: "Kerr_6DOF_TidalTest"
    model: kerr_6dof
    params:   {M: 1.0, a: 0.5, E: 0.95, L: 3.8}
    state0:   [10.0, 0.0]          # [r, phi]
    pr0:      0.0
    attitude0:
      q0: 1.0; q1: 0.0; q2: 0.0; q3: 0.0
      wx: 0.0; wy: 0.0; wz: 0.05
    inertia:  {Ixx: 100, Iyy: 200, Izz: 150}
    engine:
      F_newton: 0.0
      isp_s: 3000.0
      mass0_kg: 1000.0
      dry_mass_kg: 300.0
      nozzle_body: [0, 0, 1]
      tau_on: 0.0
      tau_off: 0.0
    ext_torque:
      tx: 0.0; ty: 0.0; tz: 0.0
    tidal:
      enabled: true
      model: RIEMANN_FD      # WEAK_N | DIAG_EIJ | RIEMANN_FD
      fd_eps_r: 1.0e-5
      Q_from_inertia: true
      spin_correction: false
    span:     [0.0, 600.0]
    solver:   {dt: 0.005, record_every: 20}
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np


def _engine():
    try:
        from relorbit_py import get_engine
        return get_engine()
    except Exception as e:
        raise RuntimeError(f"motor C++ não disponível: {e}") from e


# ── Resultado ────────────────────────────────────────────────────

@dataclass
class ResultKerr6DOF:
    name: str
    traj: Any           # TrajectoryCoupledKerr
    cfg:  Dict[str, Any]

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
        tn   = np.array(t.tidal_norm)
        al   = np.array(t.align_angle_rad)

        max_qerr = float(np.max(np.abs(qn - 1.0))) if len(qn) else float("nan")

        # ── FIX-A: T_rot drift robusto ────────────────────────────────────────
        # Bug anterior: divisão por Tr[0] causava nan/inf quando T_rot(0)=0
        # (nave sem spin inicial: wx=wy=wz=0 → T_rot(0) = ½·w·I·w = 0).
        # Correcção: drift RELATIVO só faz sentido quando |T0| >> 0;
        # caso contrário reportar drift ABSOLUTO (unidades de energia).
        T0 = float(Tr[0]) if len(Tr) else 0.0
        T_abs_drift = float(np.max(Tr) - np.min(Tr)) if len(Tr) else float("nan")
        # eps_T: threshold abaixo do qual T0 é considerado "zero"
        eps_T = max(1e-30, 1e-12 * abs(T0))
        T_rel_drift = T_abs_drift / abs(T0) if abs(T0) > eps_T else float("nan")

        # ── FIX-B: balanço trabalho-energia com torque tidal ─────────────────
        # T_rot NÃO é conservada quando há torque de maré (é esperado).
        # O invariante correcto é: dT_rot/dτ = ω·τ_tidal
        # Logo: ΔT_rot − ∫ω·τ_tidal dτ ≈ 0  (se só maré e sem outros torques)
        # "work_energy_err" próximo de 0 confirma integração consistente;
        # valor grande indica bug ou outros torques activos.
        work_tidal = float("nan")
        work_energy_err = float("nan")
        delta_Trot = float("nan")
        if len(tau) > 1 and len(list(t.tidal_tau_x)) == len(tau):
            wx_ = np.array(t.wx);  wy_ = np.array(t.wy);  wz_ = np.array(t.wz)
            tx_ = np.array(t.tidal_tau_x)
            ty_ = np.array(t.tidal_tau_y)
            tz_ = np.array(t.tidal_tau_z)
            omega_dot_tau = wx_*tx_ + wy_*ty_ + wz_*tz_
            work_tidal = float(np.trapz(omega_dot_tau, tau))
            delta_Trot = float(Tr[-1] - Tr[0]) if len(Tr) > 1 else 0.0
            denom = max(abs(delta_Trot), abs(work_tidal), 1e-30)
            work_energy_err = abs(delta_Trot - work_tidal) / denom

        # ── FIX-B2: balanço momento angular (corpo rígido no frame do corpo) ──
        # Equação de Euler no frame do corpo:
        #   dL/dτ + ω×L = τ_total
        # Para "apenas maré" (sem τ_ext), deveríamos ter:
        #   ΔL ≈ ∫(τ_tidal − ω×L) dτ
        #
        # O teu bloco antigo fazia ΔL ≈ ∫τ dτ, o que só vale quando ω×L é desprezível.
        angular_balance_err = float("nan")
        if len(tau) > 1 and len(list(t.tidal_tau_x)) == len(tau):
            wx_ = np.array(t.wx);  wy_ = np.array(t.wy);  wz_ = np.array(t.wz)
            tx_ = np.array(t.tidal_tau_x)
            ty_ = np.array(t.tidal_tau_y)
            tz_ = np.array(t.tidal_tau_z)

            # Inércia (assumindo diagonal no frame do corpo)
            Ixx = float(self.cfg.get("inertia", {}).get("Ixx", 100.0))
            Iyy = float(self.cfg.get("inertia", {}).get("Iyy", 200.0))
            Izz = float(self.cfg.get("inertia", {}).get("Izz", 150.0))

            # L(t) = I ω
            Lx = Ixx * wx_
            Ly = Iyy * wy_
            Lz = Izz * wz_
            L  = np.vstack([Lx, Ly, Lz]).T

            omega = np.vstack([wx_, wy_, wz_]).T
            omega_x_L = np.cross(omega, L)

            tau_tidal = np.vstack([tx_, ty_, tz_]).T

            rhs = tau_tidal - omega_x_L
            int_rhs = np.array([
                float(np.trapz(rhs[:, 0], tau)),
                float(np.trapz(rhs[:, 1], tau)),
                float(np.trapz(rhs[:, 2], tau)),
            ])

            dL = L[-1] - L[0]
            denom_ang = max(np.linalg.norm(dL), np.linalg.norm(int_rhs), 1e-30)
            angular_balance_err = float(np.linalg.norm(dL - int_rhs) / denom_ang)

        return {
            "status":               str(t.status),
            "n_points":             len(tau),
            "tau_span":             [float(tau[0]), float(tau[-1])],
            "r_range":              [float(np.min(t.r)), float(np.max(t.r))],
            "mass_consumed_kg":     float(mass[0] - mass[-1]) if len(mass) else 0.0,
            "qnorm_max_err":        max_qerr,
            # FIX-A: T0 exposto para diagnóstico; rel só válido quando |T0|>>0
            "T_rot0":               T0,
            "T_rot_abs_drift":      T_abs_drift,
            "T_rot_rel_drift":      T_rel_drift,
            # FIX-B: balanço energético e angular (≈0 se física OK)
            "work_tidal":           float(work_tidal) if not math.isnan(work_tidal) else 0.0,
            "delta_T_rot":          float(delta_Trot) if not math.isnan(delta_Trot) else 0.0,
            "work_energy_err":      work_energy_err,
            "angular_balance_err":  angular_balance_err,
            "epsilon_rms":          float(np.sqrt(np.mean(eps**2))) if len(eps) else float("nan"),
            "tidal_norm_max":       float(np.max(tn)) if len(tn) else 0.0,
            "align_angle_final":    float(np.degrees(al[-1])) if len(al) else float("nan"),
        }


# ── Builders ─────────────────────────────────────────────────────

def _tidal_model_from_str(s: str):
    eng = _engine()
    mapping = {
        "NONE":       eng.TidalModel.NONE,
        "WEAK_N":     eng.TidalModel.WEAK_N,
        "DIAG_EIJ":   eng.TidalModel.DIAG_EIJ,
        "RIEMANN_FD": eng.TidalModel.RIEMANN_FD,
    }
    s_upper = str(s).upper()
    if s_upper not in mapping:
        raise ValueError(f"TidalModel desconhecido: {s!r}. Opções: {list(mapping)}")
    return mapping[s_upper]


def _build_tidal_cfg(d: Dict[str, Any]):
    eng = _engine()
    td  = d.get("tidal", {})
    cfg = eng.TidalCfg()
    cfg.enabled         = bool(td.get("enabled", False))
    cfg.model           = _tidal_model_from_str(td.get("model", "WEAK_N"))
    cfg.fd_eps_r        = float(td.get("fd_eps_r", 1e-5))
    cfg.Q_from_inertia  = bool(td.get("Q_from_inertia", True))
    cfg.spin_correction = bool(td.get("spin_correction", False))
    return cfg


def _build_att_cfg_kerr(d: Dict[str, Any]):
    eng = _engine()
    ac  = eng.AttitudeCfgKerr()

    I = d.get("inertia", {})
    if I.get("Ixy", 0) == 0 and I.get("Ixz", 0) == 0 and I.get("Iyz", 0) == 0:
        ac.inertia = eng.InertiaTensor.diagonal(
            float(I.get("Ixx", 100.0)),
            float(I.get("Iyy", 200.0)),
            float(I.get("Izz", 150.0)),
        )
    else:
        ac.inertia = eng.InertiaTensor.full(
            float(I.get("Ixx", 100.0)), float(I.get("Iyy", 200.0)), float(I.get("Izz", 150.0)),
            float(I.get("Ixy", 0.0)),  float(I.get("Ixz", 0.0)),   float(I.get("Iyz", 0.0)),
        )

    T = d.get("ext_torque", {})
    ac.ext_torque.tx = float(T.get("tx", 0.0))
    ac.ext_torque.ty = float(T.get("ty", 0.0))
    ac.ext_torque.tz = float(T.get("tz", 0.0))
    ac.ext_torque.t_on  = float(T.get("t_on",  0.0))
    ac.ext_torque.t_off = float(T.get("t_off", 1e18))

    sol = d.get("solver", {})
    ac.renorm_every = int(sol.get("renorm_every", 1))
    ac.renorm_tol   = float(sol.get("renorm_tol",   1e-9))

    ac.tidal = _build_tidal_cfg(d)
    return ac


def _build_engine(d: Dict[str, Any]):
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


def _build_att0(d: Dict[str, Any]):
    att0_d = d.get("attitude0", {})
    s = _engine().AttitudeState()
    s.q0 = float(att0_d.get("q0", 1.0))
    s.q1 = float(att0_d.get("q1", 0.0))
    s.q2 = float(att0_d.get("q2", 0.0))
    s.q3 = float(att0_d.get("q3", 0.0))
    s.wx = float(att0_d.get("wx", 0.0))
    s.wy = float(att0_d.get("wy", 0.0))
    s.wz = float(att0_d.get("wz", 0.0))
    return s


def _build_solver(d: Dict[str, Any]):
    sol = d.get("solver", {})
    cfg = _engine().SolverCfg6DOF()
    cfg.dt           = float(sol.get("dt",          0.005))
    cfg.n_steps      = int  (sol.get("n_steps",      0))
    cfg.record_every = int  (sol.get("record_every", 20))
    cfg.renorm_every = int  (sol.get("renorm_every", 1))
    cfg.renorm_tol   = float(sol.get("renorm_tol",   1e-9))
    cfg.capture_r    = float(d.get("params", {}).get("capture_r", 2.0))
    cfg.capture_eps  = float(d.get("params", {}).get("capture_eps", 1e-12))
    return cfg


# ── Ponto de entrada principal ────────────────────────────────────

def run_kerr_6dof_mission(m_cfg: Dict[str, Any]) -> ResultKerr6DOF:
    """Executa missão Kerr 6-DOF acoplada a partir de dict YAML."""
    params = m_cfg.get("params", {})
    M   = float(params.get("M",   1.0))
    a   = float(params.get("a",   0.0))
    E0  = float(params.get("E",   0.95))
    L0  = float(params.get("L",   3.8))

    state0 = m_cfg.get("state0", [10.0, 0.0])
    r0     = float(state0[0])
    phi0   = float(state0[1])
    pr0    = float(m_cfg.get("pr0", 0.0))

    span   = m_cfg.get("span", [0.0, 500.0])
    tau0   = float(span[0])
    tauf   = float(span[1])

    eng     = _build_engine(m_cfg)
    att0    = _build_att0(m_cfg)
    att_cfg = _build_att_cfg_kerr(m_cfg)
    solver  = _build_solver(m_cfg)

    traj = _engine().simulate_kerr_6dof_rk4(
        M, a, E0, L0, r0, phi0, pr0,
        att0, tau0, tauf,
        eng, att_cfg, solver,
    )

    return ResultKerr6DOF(name=m_cfg.get("name", "kerr_6dof"), traj=traj, cfg=m_cfg)


# ── Validação ────────────────────────────────────────────────────

def validate_kerr_6dof(result: ResultKerr6DOF) -> Dict[str, Any]:
    s = result.summary()
    passed = (
        s.get("status") not in ("ERROR", "CAPTURE")
        and s.get("qnorm_max_err", 1.0) < 1e-6
    )

    # ── FIX-A: formatar T_rot drift de forma não-enganosa ─────────────────
    T0      = s.get("T_rot0", 0.0)
    T_abs   = s.get("T_rot_abs_drift", float("nan"))
    T_rel   = s.get("T_rot_rel_drift", float("nan"))
    if not math.isnan(T_rel):
        trot_line = (f"T_rot rel drift = {T_rel:.2e}"
                     f"  [T0={T0:.3g} J]")
    else:
        trot_line = (f"T_rot abs drift = {T_abs:.2e} J"
                     f"  [T0≈0: sem spin inicial, drift relativo indefinido]")

    # ── FIX-B: interpretar balanços ───────────────────────────────────────
    we  = s.get("work_energy_err",     float("nan"))
    ang = s.get("angular_balance_err", float("nan"))
    dT  = s.get("delta_T_rot",  0.0)
    wt  = s.get("work_tidal",   0.0)

    if math.isnan(we):
        we_line = "work-energy balance = N/A (sem dados de torque)"
    elif we < 0.02:
        we_line  = (f"work-energy balance = {we:.2e} ✓"
                    f"  ΔT={dT:.3g} J  W_tidal={wt:.3g} J")
    else:
        we_line  = (f"work-energy balance = {we:.2e} ⚠"
                    f"  ΔT={dT:.3g} J  W_tidal={wt:.3g} J"
                    f"  → verificar outros torques ou dt")

    if math.isnan(ang):
        ang_line = "angular balance = N/A"
    elif ang < 0.02:
        ang_line = f"angular balance     = {ang:.2e} ✓"
    else:
        ang_line = f"angular balance     = {ang:.2e} ⚠  → verificar inércia ou τ_ext"

    # ── FIX-2: aviso de tau=0 só quando maré está activa ────────────────────
    tn            = s.get("tidal_norm_max", 0.0)
    tidal_enabled = result.cfg.get("tidal", {}).get("enabled", False)
    tidal_model   = str(result.cfg.get("tidal", {}).get("model", "NONE")).upper()

    if not tidal_enabled or tidal_model == "NONE":
        tidal_line = "Tidal: desactivado (correcto)"
    elif tn == 0.0:
        # Torque zero com maré activada → verdadeiro aviso
        if tidal_model == "WEAK_N":
            hint = ("Ex: q=(0.924,0,0.383,0) [45° em y] garante n_body≠eixo principal.")
        else:
            hint = "Verificar Q_from_inertia e inércia assimétrica."
        tidal_line = (f"Tidal max |tau| = 0  ⚠  [{tidal_model}]"
                      f" → n_body pode estar alinhado com eixo principal de I. {hint}")
    else:
        tidal_line = f"Tidal max |tau| = {tn:.2e}  [{tidal_model}] ✓"

    notes = [
        f"Orbita Kerr: r in [{s.get('r_range',['?','?'])[0]:.2f}, {s.get('r_range',['?','?'])[1]:.2f}] M",
        f"Massa consumida   = {s.get('mass_consumed_kg', 0.0):.2f} kg",
        f"||q|| max err     = {s.get('qnorm_max_err', float('nan')):.2e}",
        f"epsilon RMS       = {s.get('epsilon_rms', float('nan')):.2e}",
        trot_line,
        we_line,
        ang_line,
        tidal_line,
        f"Align final       = {s.get('align_angle_final', float('nan')):.2f} deg",
    ]
    return {
        "status":  "PASS" if passed else "FAIL",
        "notes":   notes,
        "metrics": {k: v for k, v in s.items() if isinstance(v, float)},
    }


# ── Plots ────────────────────────────────────────────────────────

def plot_kerr_6dof(result: ResultKerr6DOF, outdir: str) -> List[str]:
    """Gera plots para missão Kerr 6-DOF: órbita, atitude, torque tidal, alinhamento."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    traj = result.traj
    if traj is None or not traj.tau:
        return []

    os.makedirs(outdir, exist_ok=True)
    tau   = list(traj.tau)
    paths: List[str] = []
    name  = result.name

    # ── Plot 1: Órbita + Massa ──────────────────────────────────
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    fig.suptitle(f"{name} - Orbita Kerr + Massa", fontsize=12)
    ax0.plot(tau, traj.r, linewidth=1.4)
    ax0.set_ylabel("r [M]"); ax0.grid(True, alpha=0.3)
    ax1.plot(tau, traj.mass, color="darkorange", linewidth=1.4)
    ax1.set_ylabel("massa [kg]"); ax1.set_xlabel("τ [M]"); ax1.grid(True, alpha=0.3)
    fig.tight_layout()
    p = os.path.join(outdir, f"{name}_orbit_mass.png")
    fig.savefig(p, dpi=130); plt.close(fig); paths.append(p)

    # ── Plot 2: ω e T_rot ──────────────────────────────────────
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    fig.suptitle(f"{name} - Velocidade Angular e Energia Rotacional", fontsize=12)
    for comp, lbl, col in zip([traj.wx, traj.wy, traj.wz], ["ωx","ωy","ωz"], ["C0","C1","C2"]):
        ax0.plot(tau, comp, label=lbl, color=col)
    ax0.set_ylabel("ω [rad/s]"); ax0.legend(fontsize=8); ax0.grid(True, alpha=0.3)
    Tr  = np.array(traj.T_rot)
    T0p = Tr[0] if abs(Tr[0]) > 1e-30 else 1.0
    ax1.plot(tau, Tr/T0p - 1.0, color="darkorange")
    ax1.axhline(0.0, color="k", linewidth=0.5, linestyle="--")
    ax1.set_ylabel("T_rot/T₀ − 1"); ax1.set_xlabel("τ [M]"); ax1.grid(True, alpha=0.3)
    fig.tight_layout()
    p = os.path.join(outdir, f"{name}_omega_Trot.png")
    fig.savefig(p, dpi=130); plt.close(fig); paths.append(p)

    # ── Plot 3: Quaternion e ||q|| − 1 ──────────────────────────
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    fig.suptitle(f"{name} - Quaternion", fontsize=12)
    for comp, lbl in zip([traj.q0, traj.q1, traj.q2, traj.q3], ["q0","q1","q2","q3"]):
        ax0.plot(tau, comp, label=lbl)
    ax0.set_ylabel("componentes"); ax0.legend(fontsize=8); ax0.grid(True, alpha=0.3)
    ax1.plot(tau, [n - 1.0 for n in traj.qnorm], color="red", linewidth=0.8)
    ax1.axhline(0.0, color="k", linewidth=0.5, linestyle="--")
    ax1.set_ylabel("‖q‖ − 1"); ax1.set_xlabel("τ [M]"); ax1.grid(True, alpha=0.3)
    fig.tight_layout()
    p = os.path.join(outdir, f"{name}_quaternion.png")
    fig.savefig(p, dpi=130); plt.close(fig); paths.append(p)

    # ── Plot 4: Torque de Maré ──────────────────────────────────
    if any(abs(v) > 0 for v in traj.tidal_norm):
        fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
        fig.suptitle(f"{name} - Torque de Maré", fontsize=12)
        for comp, lbl, col in zip([traj.tidal_tau_x, traj.tidal_tau_y, traj.tidal_tau_z],
                                   ["τx","τy","τz"], ["C0","C1","C2"]):
            ax0.plot(tau, comp, label=lbl, color=col)
        ax0.set_ylabel("τ_tidal [N·m]"); ax0.legend(fontsize=8); ax0.grid(True, alpha=0.3)
        ax1.plot(tau, traj.tidal_norm, color="purple", linewidth=0.8)
        ax1.set_ylabel("|τ_tidal|"); ax1.set_xlabel("τ [M]"); ax1.grid(True, alpha=0.3)
        fig.tight_layout()
        p = os.path.join(outdir, f"{name}_tidal_torque.png")
        fig.savefig(p, dpi=130); plt.close(fig); paths.append(p)

    # ── Plot 5: Ângulo de Alinhamento ────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 4))
    fig.suptitle(f"{name} - Alinhamento Eixo Corpo / Direcção Radial", fontsize=12)
    align_deg = [math.degrees(a_) for a_ in traj.align_angle_rad]
    ax.plot(tau, align_deg, color="teal", linewidth=1.2)
    ax.axhline(0.0, color="k", linewidth=0.5, linestyle="--")
    ax.set_ylabel("ângulo [graus]"); ax.set_xlabel("τ [M]"); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p = os.path.join(outdir, f"{name}_align_angle.png")
    fig.savefig(p, dpi=130); plt.close(fig); paths.append(p)

    # ── Plot 6: E_ij norma ────────────────────────────────────
    if any(v > 0 for v in traj.tidal_E_norm):
        fig, ax = plt.subplots(figsize=(11, 4))
        ax.set_title(f"{name} - Norma Frobenius de E_ij (tensor de maré ZAMO)")
        ax.plot(tau, traj.tidal_E_norm, color="crimson", linewidth=1.0)
        # Referência campo fraco: 3*M/r^3 * sqrt(6) ≈ ‖diag(−2,1,1)‖ * M/r^3
        r_arr = np.array(traj.r)
        M_val = result.cfg.get("params", {}).get("M", 1.0)
        E_weak = np.sqrt(6.0) * M_val / r_arr**3
        ax.plot(tau, E_weak, "k--", linewidth=0.8, label="campo fraco 3M/r³·√6")
        ax.set_yscale("log")
        ax.set_ylabel("‖E_ij‖_F"); ax.set_xlabel("τ [M]"); ax.grid(True, alpha=0.3); ax.legend()
        fig.tight_layout()
        p = os.path.join(outdir, f"{name}_Eij_norm.png")
        fig.savefig(p, dpi=130); plt.close(fig); paths.append(p)

    return paths