# src/relorbit_py/run_mission.py
"""
Ponto de entrada para simulações de missão.

Uso:
    py -3 -m relorbit_py.mission.run_mission
    py -3 -m relorbit_py.mission.run_mission --yaml src/relorbit_py/mission/kerr_6dof_cases.yaml --out out/missions
"""
from __future__ import annotations

import argparse
import copy
import math
import os
import sys
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from relorbit_py.core.simulate        import load_cases_yaml, simulate_case
from relorbit_py.mission.mission      import run_mission, MissionResult

from relorbit_py.plots.plots_orbit import (
    plot_orbit,
    plot_mass,
    print_budget_tables,
)
from relorbit_py.plots.plots_invariant_telemetry import (
    plot_invariant,
    plot_telemetry,
)
from relorbit_py.plots.plots_visibility_redshift import (
    plot_visibility_map,
    plot_redshift_asymptotic,
)
from relorbit_py.core.simulate_6dof import (
    run_6dof_mission,
    validate_6dof,
)
from relorbit_py.mission.attitude_mission import (
    from_yaml_dict       as attitude_cfg_from_yaml,
    run_attitude_mission,
    validate_attitude,
)
from relorbit_py.core.simulate_kerr_6dof import (
    run_kerr_6dof_mission,
    validate_kerr_6dof,
    plot_kerr_6dof,
)

# ── Item 8: importação opcional de helpers de validate_coupling ───────────────
try:
    from relorbit_py.validate.validate_coupling import (
        CFG_GEO, CFG_R_OUT, CFG_PHI, CFG_R_IN,
        extract as _coupling_extract,
    )
    _COUPLING_AVAILABLE = True
except ImportError:
    _COUPLING_AVAILABLE = False

from relorbit_py.telemetry.null_geodesic_kerr import (
    circular_orbit_omega,
)
from relorbit_py.telemetry.telemetry_raytracer import (
    TelemetryRayTracer,
    TelemetryResult,
    RayTracerConfig,
)
from relorbit_py.plots.plot_raytracer import (
    plot_raytracer_results,
    print_raytracer_report,
)


# ─────────────────────────────────────────────────────────────────────────────
# Constantes Item 8 (usadas em _run_coupling_mission)
# ─────────────────────────────────────────────────────────────────────────────
# τ_span = T_orbit/4 para E=0.95, L=3.8, M=1, a=0.5, r=10 M
# dphi/dtau ≈ 0.0391 rad/M  →  T ≈ 160.8 M  →  T/4 ≈ 40.2 M
_ITEM8_ORBIT_PERIOD_M = 160.8
_ITEM8_TAU_SPAN       = _ITEM8_ORBIT_PERIOD_M / 4.0   # ≈ 40.2 M


# ── Atitude ───────────────────────────────────────────────────

def _plot_attitude(result: Any, outdir: str) -> List[str]:
    """Quaternion + norma  e  omega + T_rot."""
    traj = result.traj
    if not traj or traj.status != "OK" or not traj.t:
        return []
    os.makedirs(outdir, exist_ok=True)
    t = traj.t
    paths: List[str] = []

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    fig.suptitle(f"{result.name} - Quaternion", fontsize=12)
    for comp, label in zip([traj.q0, traj.q1, traj.q2, traj.q3], ["q0", "q1", "q2", "q3"]):
        ax0.plot(t, comp, label=label)
    ax0.set_ylabel("componentes"); ax0.legend(fontsize=8); ax0.grid(True, alpha=0.3)
    ax1.plot(t, [n - 1.0 for n in traj.qnorm], color="red", linewidth=0.8)
    ax1.axhline(0.0, color="black", linewidth=0.5, linestyle="--")
    ax1.set_ylabel("||q|| - 1"); ax1.set_xlabel("tempo"); ax1.grid(True, alpha=0.3)
    fig.tight_layout()
    p1 = os.path.join(outdir, f"{result.name}_quaternion.png")
    fig.savefig(p1, dpi=120); plt.close(fig); paths.append(p1)

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    fig.suptitle(f"{result.name} - Velocidade angular e energia", fontsize=12)
    for comp, label in zip([traj.wx, traj.wy, traj.wz], ["wx", "wy", "wz"]):
        ax0.plot(t, comp, label=label)
    ax0.set_ylabel("omega [rad/s]"); ax0.legend(fontsize=8); ax0.grid(True, alpha=0.3)
    T0 = traj.T_rot[0] if traj.T_rot else 1.0
    if abs(T0) > 1e-30:
        ax1.plot(t, [T / T0 - 1.0 for T in traj.T_rot], color="darkorange")
        ax1.set_ylabel("T_rot/T0 - 1")
    else:
        ax1.plot(t, traj.T_rot, color="darkorange")
        ax1.set_ylabel("T_rot")
    ax1.axhline(0.0, color="black", linewidth=0.5, linestyle="--")
    ax1.set_xlabel("tempo"); ax1.grid(True, alpha=0.3)
    fig.tight_layout()
    p2 = os.path.join(outdir, f"{result.name}_omega_Trot.png")
    fig.savefig(p2, dpi=120); plt.close(fig); paths.append(p2)

    return paths


def _run_attitude_mission(m_cfg: Dict[str, Any], outdir: str) -> bool:
    name = m_cfg.get("name", "<sem-nome>")
    try:
        result = run_attitude_mission(attitude_cfg_from_yaml(m_cfg))
        report = validate_attitude(result)
        print(f"   Status : {report['status']}")
        for note in report["notes"]:
            print(f"   {note}")
        for k, v in report["metrics"].items():
            print(f"   {k}: {v:.6g}")
        for p in _plot_attitude(result, outdir):
            print(f"   Plot: {p}")
        return report["status"] == "PASS"
    except Exception as ex:
        print(f"   [ERRO] Falha na missao de atitude {name}: {ex}")
        return False


# ── 6-DOF Schwarzschild ───────────────────────────────────────

def _plot_6dof(result: Any, outdir: str) -> List[str]:
    traj = result.traj
    if traj is None or not traj.tau:
        return []
    os.makedirs(outdir, exist_ok=True)
    tau = list(traj.tau)
    paths: List[str] = []

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    fig.suptitle(f"{result.name} - Orbita + Massa", fontsize=12)
    ax0.plot(tau, traj.r, linewidth=1.4)
    ax0.set_ylabel("r [M]"); ax0.grid(True, alpha=0.3)
    ax1.plot(tau, traj.mass, color="darkorange", linewidth=1.4)
    ax1.set_ylabel("massa [kg]"); ax1.set_xlabel("tau"); ax1.grid(True, alpha=0.3)
    fig.tight_layout()
    p = os.path.join(outdir, f"{result.name}_orbit_mass.png")
    fig.savefig(p, dpi=130); plt.close(fig); paths.append(p)

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    fig.suptitle(f"{result.name} - Quaternion", fontsize=12)
    for comp, lbl in zip([traj.q0, traj.q1, traj.q2, traj.q3], ["q0", "q1", "q2", "q3"]):
        ax0.plot(tau, comp, label=lbl)
    ax0.set_ylabel("componentes"); ax0.legend(fontsize=8); ax0.grid(True, alpha=0.3)
    ax1.plot(tau, [n - 1.0 for n in traj.qnorm], color="red", linewidth=0.8)
    ax1.axhline(0.0, color="black", linewidth=0.5, linestyle="--")
    ax1.set_ylabel("||q|| - 1"); ax1.set_xlabel("tau"); ax1.grid(True, alpha=0.3)
    fig.tight_layout()
    p = os.path.join(outdir, f"{result.name}_quaternion.png")
    fig.savefig(p, dpi=130); plt.close(fig); paths.append(p)

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    fig.suptitle(f"{result.name} - Thrust Vectoring", fontsize=12)
    ax0.plot(tau, traj.thrust_r,   label="f_r [geom]",   linewidth=1.2)
    ax0.plot(tau, traj.thrust_phi, label="f_phi [geom]", linewidth=1.2, linestyle="--")
    ax0.set_ylabel("acel. geometrica"); ax0.legend(fontsize=8); ax0.grid(True, alpha=0.3)
    ax1.plot(tau, list(traj.epsilon), color="purple", linewidth=0.8)
    ax1.axhline(0.0, color="black", linewidth=0.5, linestyle="--")
    ax1.set_ylabel("epsilon = pr2 + Veff - E2"); ax1.set_xlabel("tau"); ax1.grid(True, alpha=0.3)
    fig.tight_layout()
    p = os.path.join(outdir, f"{result.name}_thrust_epsilon.png")
    fig.savefig(p, dpi=130); plt.close(fig); paths.append(p)

    return paths


def _run_6dof_mission(m_cfg: Dict[str, Any], outdir: str) -> bool:
    name = m_cfg.get("name", "<sem-nome>")
    try:
        result = run_6dof_mission(m_cfg)
        report = validate_6dof(result)
        print(f"   Status : {report['status']}")
        for note in report["notes"]:
            print(f"   {note}")
        for p in _plot_6dof(result, outdir):
            print(f"   Plot: {p}")
        return report["status"] == "PASS"
    except Exception as ex:
        print(f"   [ERRO] Falha na missao 6-DOF {name}: {ex}")
        return False


# ── Kerr 6-DOF ────────────────────────────────────────────────

def _run_kerr_6dof_mission(m_cfg: Dict[str, Any], outdir: str) -> bool:
    name = m_cfg.get("name", "<sem-nome>")
    try:
        result = run_kerr_6dof_mission(m_cfg)
        report = validate_kerr_6dof(result)
        print(f"   Status : {report['status']}")
        for note in report["notes"]:
            print(f"   {note}")
        for k, v in report["metrics"].items():
            if isinstance(v, float):
                print(f"   {k}: {v:.6g}")
        tidal_outdir = os.path.join(outdir, "kerr_6dof_plots")
        for p in plot_kerr_6dof(result, tidal_outdir):
            print(f"   Plot: {p}")
        return report["status"] == "PASS"
    except Exception as ex:
        import traceback
        print(f"   [ERRO] Falha na missao Kerr 6-DOF {name}: {ex}")
        traceback.print_exc()
        return False


# ── Kerr Ray Tracer ───────────────────────────────────────────

def _run_raytracer_mission(m_cfg: Dict[str, Any], outdir: str) -> bool:
    """
    Handler para model: kerr_raytracer.

    Fluxo:
      1. Integra órbita Kerr 6-DOF (mesmo solver que kerr_6dof)
      2. Constrói TelemetryRayTracer via from_kerr_trajectory()
      3. rt.run() → TelemetryResult (b*(τ), z(τ), Δt(τ) por busca binária)
      4. Plots em out/missions/raytracer_plots/ + relatório no terminal
    """
    name = m_cfg.get("name", "<sem-nome>")
    try:
        result_6dof = run_kerr_6dof_mission(m_cfg)
        report_6dof = validate_kerr_6dof(result_6dof)
        print(f"   [6DOF] Status : {report_6dof['status']}")
        for note in report_6dof["notes"]:
            print(f"   [6DOF] {note}")

        traj = result_6dof.traj
        if traj is None or not list(traj.tau):
            print(f"   [ERRO] Trajectória vazia — ray tracing impossível.")
            return False

        rt_cfg_d = m_cfg.get("raytracer", {})
        cfg = RayTracerConfig(
            receiver_r   = float(rt_cfg_d.get("receiver_r",   1000.0)),
            receiver_phi = float(rt_cfg_d.get("receiver_phi",    0.0)),
            n_lut        = int(rt_cfg_d.get("n_lut",           1000)),
            n_steps_lut  = int(rt_cfg_d.get("n_steps_lut",  12_000)),
            n_images_max = int(rt_cfg_d.get("n_images_max",      2)),
            dl_coarse    = float(rt_cfg_d.get("dl_coarse",     0.5)),
            dl_fine      = float(rt_cfg_d.get("dl_fine",       0.05)),
            n_bisect     = int(rt_cfg_d.get("n_bisect",         50)),
        )
        do_fan = bool(rt_cfg_d.get("do_fan", True))

        tracer    = TelemetryRayTracer.from_kerr_trajectory(traj, cfg=cfg)
        rt_result : TelemetryResult = tracer.run()

        print_raytracer_report(rt_result, name=name)
        rt_outdir = os.path.join(outdir, "raytracer_plots")
        for p in plot_raytracer_results(rt_result, rt_outdir, name=name, do_fan=do_fan):
            print(f"   Plot: {p}")

        return report_6dof["status"] == "PASS"

    except Exception as ex:
        import traceback
        print(f"   [ERRO] Falha na missao ray tracer {name}: {ex}")
        traceback.print_exc()
        return False


# ── Item 8: Acoplamento órbita–atitude ───────────────────────────
#
# YAML mínimo:
#   - name: coupling_test
#     model: coupling_test
#     params: {M: 1.0, a: 0.5, E: 0.95, L: 3.8}   # opcional
#     coupling:
#       F_newton:  30.0     # default 30
#       tau_final: 40.2     # default T/4 ≈ 40.2 M  (NÃO usar 300!)
#       dt:        0.005
#
# NOTA: O YAML preferido para Item 8 são as 4 missões explícitas
# kerr_item8_* com model: kerr_6dof (ver kerr_6dof_cases.yaml).
# O handler coupling_test é mantido para compatibilidade, mas agora:
#   1. Usa copy.deepcopy em cada sub-caso (sem partilha de dicts mutáveis)
#   2. Usa a_geom_override = F_n / mass0  (correcção de escala, Item 8-A)
#   3. Usa tau_span = T/4 ≈ 40.2 M por default  (correcção de averaging, Item 8-B)

def _run_coupling_mission(m_cfg: Dict[str, Any], outdir: str) -> bool:
    import numpy as np
    name = m_cfg.get("name", "coupling_test")

    try:
        cp    = m_cfg.get("coupling", {})
        base  = m_cfg.get("params",   {})
        M_bh  = float(base.get("M",        1.0))
        a_bh  = float(base.get("a",        0.5))
        E0    = float(base.get("E",        0.95))
        L0    = float(base.get("L",        3.8))
        r0    = float(base.get("r0",      10.0))
        F_n   = float(cp.get("F_newton",  30.0))
        # Item 8-B: default T/4 em vez de 300
        tau_span = float(cp.get("tau_final", _ITEM8_TAU_SPAN))
        dt    = float(cp.get("dt",        0.005))
        isp   = float(cp.get("isp_s",   3000.0))
        mass0 = float(cp.get("mass0_kg", 1000.0))
        dry   = float(cp.get("dry_mass_kg", 300.0))
        # Item 8-A: a_geom_override = F/m para unidades geométricas c=1
        a_geom_override = float(cp.get("a_geom_override", F_n / mass0 if mass0 > 0 else 0.0))

        def _q_rotz(deg: float):
            h = math.radians(deg) / 2.0
            return (math.cos(h), 0.0, 0.0, math.sin(h))

        # Item 8-E: deepcopy — sem partilha de dicts mutáveis entre sub-casos
        def _make(label: str, q_deg: float, F: float = F_n) -> Dict[str, Any]:
            q0_, q1_, q2_, q3_ = _q_rotz(q_deg)
            ov = a_geom_override if F > 0 else 0.0
            cfg_base = {
                "name": f"{name}_{label}",
                "model": "kerr_6dof",
                "params": {
                    "M": M_bh, "a": a_bh, "E": E0, "L": L0,
                    "capture_r": 2.0, "capture_eps": 1e-12,
                },
                "state0": [r0, 0.0], "pr0": 0.0,
                "attitude0": {
                    "q0": q0_, "q1": q1_, "q2": q2_, "q3": q3_,
                    "wx": 0.0, "wy": 0.0, "wz": 0.0,
                },
                "inertia": {"Ixx": 100.0, "Iyy": 200.0, "Izz": 150.0},
                "engine": {
                    "F_newton":        F,
                    "a_geom_override": ov,      # ← Item 8-A
                    "isp_s":           isp,
                    "mass0_kg":        mass0,
                    "dry_mass_kg":     dry,
                    "nozzle_body":     [1.0, 0.0, 0.0],
                    "tau_on":          0.0,
                    "tau_off":         tau_span,  # ← Item 8-B
                },
                "ext_torque": {"tx": 0.0, "ty": 0.0, "tz": 0.0},
                "tidal": {
                    "enabled": False, "model": "NONE",
                    "fd_eps_r": 1e-5, "Q_from_inertia": True,
                    "spin_correction": False,
                },
                "span":   [0.0, tau_span],
                "solver": {"dt": dt, "record_every": 100},
            }
            return copy.deepcopy(cfg_base)   # ← deepcopy garantido

        cases = [
            ("geodesic",       _make("geodesic",       0.0, F=0.0)),
            ("radial_outward", _make("radial_outward",   0.0)),
            ("tangential",     _make("tangential",      90.0)),
            ("radial_inward",  _make("radial_inward",  180.0)),
        ]

        metrics: Dict[str, Any] = {}
        for lbl, cfg_c in cases:
            print(f"   → integrando: {lbl} ...", end="", flush=True)
            res   = run_kerr_6dof_mission(cfg_c)
            t     = res.traj
            r_a   = np.asarray(t.r,       dtype=float)
            L_a   = np.asarray(t.L,       dtype=float)
            pr_a  = np.asarray(t.pr,      dtype=float)
            eps_a = np.asarray(t.epsilon, dtype=float)
            qn_a  = np.asarray(t.qnorm,   dtype=float)
            tr_a  = np.asarray(t.thrust_r,   dtype=float)
            tph_a = np.asarray(t.thrust_phi, dtype=float)
            metrics[lbl] = {
                "status":          str(t.status),
                "r_final":         float(r_a[-1])       if len(r_a) else float("nan"),
                "r_min":           float(np.min(r_a))   if len(r_a) else float("nan"),
                "r_max":           float(np.max(r_a))   if len(r_a) else float("nan"),
                "delta_L":         float(L_a[-1] - L_a[0]) if len(L_a) else float("nan"),
                "delta_pr":        float(pr_a[-1] - pr_a[0]) if len(pr_a) else float("nan"),
                "eps_rms":         float(np.sqrt(np.mean(eps_a**2))) if len(eps_a) else float("nan"),
                "qnorm_err":       float(np.max(np.abs(qn_a - 1.))) if len(qn_a) else float("nan"),
                "thrust_r_mean":   float(np.mean(np.abs(tr_a[tr_a != 0])))   if np.any(tr_a  != 0) else 0.0,
                "thrust_phi_mean": float(np.mean(np.abs(tph_a[tph_a != 0]))) if np.any(tph_a != 0) else 0.0,
            }
            print(f" r_f={metrics[lbl]['r_final']:.4f}M"
                  f"  r∈[{metrics[lbl]['r_min']:.3f},{metrics[lbl]['r_max']:.3f}]"
                  f"  [{metrics[lbl]['status']}]")

        geo = metrics["geodesic"]
        ro  = metrics["radial_outward"]
        tan = metrics["tangential"]
        ri  = metrics["radial_inward"]

        checks: List[bool] = []

        def _chk(desc: str, ok: bool, detail: str) -> None:
            checks.append(ok)
            sym = "✓" if ok else "✗"
            print(f"   [{sym}] {desc}")
            print(f"        {detail}")

        # ── Asserts de direcção ───────────────────────────────────────────────
        LIMIAR = 1e-6
        _chk("tangential: thrust_phi_mean > 0",
             tan["thrust_phi_mean"] > LIMIAR,
             f"thrust_phi={tan['thrust_phi_mean']:.3e}  thrust_r={tan['thrust_r_mean']:.3e}")
        _chk("radial_outward: thrust_r_mean > 0",
             ro["thrust_r_mean"] > LIMIAR,
             f"thrust_r={ro['thrust_r_mean']:.3e}")
        _chk("radial_inward: thrust_r_mean > 0",
             ri["thrust_r_mean"] > LIMIAR,
             f"thrust_r={ri['thrust_r_mean']:.3e}")
        _chk("geodesic: thrust ≡ 0",
             geo["thrust_r_mean"] < LIMIAR and geo["thrust_phi_mean"] < LIMIAR,
             f"thrust_r={geo['thrust_r_mean']:.3e}  thrust_phi={geo['thrust_phi_mean']:.3e}")

        # ── Asserts físicos ───────────────────────────────────────────────────
        # Usa r_max (apoapsis) e r_min (periapsis) — mais robustos que r_final,
        # pois r_final depende da fase orbital no instante exacto de τ_off.
        drmax_ro  = ro["r_max"]  - geo["r_max"]
        drmax_tan = tan["r_max"] - geo["r_max"]
        drmin_ri  = ri["r_min"]  - geo["r_min"]

        _chk("Thrust outward expande apoapsis vs geodésica  (r_max)",
             drmax_ro > 0.01,
             f"r_max: outward={ro['r_max']:.4f}M  geo={geo['r_max']:.4f}M  "
             f"Δr_max={drmax_ro:+.4f}M  (limiar +0.01M)")

        _chk("Thrust tangencial expande apoapsis vs geodésica  (r_max)",
             drmax_tan > 0.01,
             f"r_max: tangential={tan['r_max']:.4f}M  geo={geo['r_max']:.4f}M  "
             f"Δr_max={drmax_tan:+.4f}M  (limiar +0.01M)")

        _chk("Thrust inward contrai periapsis vs geodésica  (r_min)",
             drmin_ri < -0.001,
             f"r_min: inward={ri['r_min']:.4f}M  geo={geo['r_min']:.4f}M  "
             f"Δr_min={drmin_ri:+.4f}M  (limiar -0.001M)")

        _chk("Atitudes diferentes → r_max diferentes  (acoplamento real)",
             abs(ro["r_max"] - tan["r_max"]) > 0.01
             and abs(ro["r_max"] - ri["r_max"]) > 0.01,
             f"r_max: outward={ro['r_max']:.4f}  tangential={tan['r_max']:.4f}  "
             f"inward={ri['r_max']:.4f}")

        _chk("Thrust radial acumula pr, não L",
             abs(ro["delta_L"]) < 0.5 * max(abs(ro["delta_pr"]), 1e-9),
             f"ΔL={ro['delta_L']:+.4f}  Δpr={ro['delta_pr']:+.4f}")

        _chk("Thrust tangencial acumula L",
             abs(tan["delta_L"]) > 0.05,
             f"ΔL={tan['delta_L']:+.4f}  (esperado >0.05)")

        _chk("Outward/inward: r_max opostos ao redor da geodésica",
             drmax_ro > 0 and drmax_tan > 0 and drmin_ri < 0,
             f"Δr_max_out={drmax_ro:+.4f}  Δr_max_tan={drmax_tan:+.4f}  "
             f"Δr_min_in={drmin_ri:+.4f}")

        for lbl, m in metrics.items():
            _chk(f"ε_rms razoável [{lbl}]",
                 math.isfinite(m["eps_rms"]) and m["eps_rms"] < 1.0,
                 f"ε_rms={m['eps_rms']:.2e}")
            _chk(f"‖q‖-1 < 1e-6 [{lbl}]",
                 math.isfinite(m["qnorm_err"]) and m["qnorm_err"] < 1e-6,
                 f"‖q‖err={m['qnorm_err']:.2e}")

        # ── Plot de comparação ────────────────────────────────────────────────
        try:
            coup_outdir = os.path.join(outdir, "coupling_plots")
            os.makedirs(coup_outdir, exist_ok=True)
            all_traj: Dict[str, Any] = {}
            colors = {
                "geodesic":       "gray",
                "radial_outward": "royalblue",
                "tangential":     "forestgreen",
                "radial_inward":  "crimson",
            }
            for lbl, cfg_c in cases:
                all_traj[lbl] = run_kerr_6dof_mission(copy.deepcopy(cfg_c)).traj

            fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
            fig.suptitle(f"{name} — Acoplamento Órbita–Atitude  (Item 8)", fontsize=12)
            for lbl, traj_t in all_traj.items():
                tau_a = np.asarray(traj_t.tau)
                r_a   = np.asarray(traj_t.r)
                L_a   = np.asarray(traj_t.L)
                pr_a  = np.asarray(traj_t.pr)
                lw    = 1.8 if lbl != "geodesic" else 1.0
                ls    = "--" if lbl == "geodesic" else "-"
                axes[0].plot(tau_a, r_a,  color=colors[lbl], lw=lw, ls=ls, label=lbl)
                axes[1].plot(tau_a, L_a,  color=colors[lbl], lw=lw, ls=ls)
                axes[2].plot(tau_a, pr_a, color=colors[lbl], lw=lw, ls=ls)
            axes[0].set_ylabel("r [M]"); axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.3)
            axes[1].set_ylabel("L [mM]"); axes[1].grid(True, alpha=0.3)
            axes[2].set_ylabel("pr"); axes[2].set_xlabel("τ [M]"); axes[2].grid(True, alpha=0.3)
            fig.tight_layout()
            plot_path = os.path.join(coup_outdir, f"{name}_coupling.png")
            fig.savefig(plot_path, dpi=130); plt.close(fig)
            print(f"   Plot: {plot_path}")
        except Exception as pex:
            print(f"   [AVISO] Plot de acoplamento falhou: {pex}")

        n_pass = sum(checks)
        n_fail = len(checks) - n_pass
        status = "PASS" if n_fail == 0 else "FAIL"
        print(f"   {'='*50}")
        print(f"   ACOPLAMENTO ITEM 8: {status}  ({n_pass}/{len(checks)} critérios)")
        print(f"   τ_span={tau_span:.1f}M  a_geom_override={a_geom_override:.4f} M⁻¹")
        print(f"   {'='*50}")
        return n_fail == 0

    except Exception as ex:
        import traceback
        print(f"   [ERRO] Falha na validação de acoplamento {name}: {ex}")
        traceback.print_exc()
        return False


# ── Runner principal ──────────────────────────────────────────────────────────

def run_all_missions(yaml_path: str, outdir: str = "out/missions") -> bool:
    cfg      = load_cases_yaml(yaml_path)
    missions = cfg.get("missions", [])
    os.makedirs(outdir, exist_ok=True)

    all_ok = True
    for m_cfg in missions:
        name  = m_cfg.get("name",  "<sem-nome>")
        model = m_cfg.get("model", "schwarzschild_equatorial")
        print(f"\n{'='*60}\n==> Missao: {name}  [{model}]\n{'='*60}")

        if model == "schwarzschild_6dof":
            if not _run_6dof_mission(m_cfg, outdir):
                all_ok = False
            continue

        if model == "kerr_6dof":
            if not _run_kerr_6dof_mission(m_cfg, outdir):
                all_ok = False
            continue

        if model == "coupling_test":
            if not _run_coupling_mission(m_cfg, outdir):
                all_ok = False
            continue

        if model == "kerr_raytracer":
            if not _run_raytracer_mission(m_cfg, outdir):
                all_ok = False
            continue

        if model == "attitude":
            if not _run_attitude_mission(m_cfg, outdir):
                all_ok = False
            continue

        try:
            result = run_mission(m_cfg, simulate_fn=simulate_case)
            print_budget_tables(result)

            if result.segments:
                orbit_path = plot_orbit(result, outdir, m_cfg)
                mass_path  = plot_mass(result, outdir)
                inv_path   = plot_invariant(result, outdir, m_cfg)
                print(f"\n   Plots: {orbit_path}, {mass_path}")
                if inv_path:
                    print(f"   Invariante:  {inv_path}")

                rs_path = plot_redshift_asymptotic(result, outdir, m_cfg)
                if rs_path:
                    print(f"   Redshift:    {rs_path}")

                vis_paths = plot_visibility_map(result, outdir, m_cfg, phi_obs=0.0)
                for vp in vis_paths:
                    print(f"   Visibilidade: {vp}")

                tele_paths = plot_telemetry(result, outdir, m_cfg, observer_dir=(1.0, 0.0))
                if tele_paths:
                    print(f"   Telemetry:   {os.path.join(outdir, 'telemetry_plots')}")
            else:
                print(f"\n   [AVISO] Graficos ignorados: sem segmentos gerados.")

            if not result.ok:
                all_ok = False

        except Exception as ex:
            print(f"   [ERRO] Falha critica na missao {name}: {ex}")
            all_ok = False

    return all_ok


# ── CLI ───────────────────────────────────────────────────────

def _find_yaml() -> str:
    for c in ["mission.yaml", "kerr_6dof_cases.yaml", "src/relorbit_py/mission.yaml"]:
        if os.path.exists(c):
            return c
    raise FileNotFoundError(
        "mission.yaml nao encontrado. Use --yaml para especificar o caminho."
    )


def _configure_stdio() -> None:
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is None:
            continue

        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue

        try:
            reconfigure(errors="replace")
        except Exception:
            pass


def main() -> None:
    _configure_stdio()
    parser = argparse.ArgumentParser(description="RelOrbit - simulador de missoes")
    parser.add_argument("--yaml", default=None, help="Caminho para mission.yaml")
    parser.add_argument("--out",  default="out/missions", help="Diretorio de saida")
    args = parser.parse_args()
    yaml_path = args.yaml or _find_yaml()
    print(f"YAML: {yaml_path}\nOut:  {args.out}")
    sys.exit(0 if run_all_missions(yaml_path, outdir=args.out) else 1)


if __name__ == "__main__":
    main()
