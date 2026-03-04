# src/relorbit_py/run_mission.py
"""
Ponto de entrada para simulações de missão.

Uso:
    py -3 -m relorbit_py.run_mission
    py -3 -m relorbit_py.run_mission --yaml src/relorbit_py/mission.yaml --out out/missions
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from relorbit_py.simulate import load_cases_yaml, simulate_case
from relorbit_py.mission  import run_mission, MissionResult

from relorbit_py.plots_orbit import (
    plot_orbit,
    plot_mass,
    print_budget_tables,
)
from relorbit_py.plots_invariant_telemetry import (
    plot_invariant,
    plot_telemetry,
)
from relorbit_py.plots_visibility_redshift import (
    plot_visibility_map,
    plot_redshift_asymptotic,
)
from relorbit_py.attitude_mission import (
    from_yaml_dict       as attitude_cfg_from_yaml,
    run_attitude_mission,
    validate_attitude,
)


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
    for comp, label in zip([traj.q0, traj.q1, traj.q2, traj.q3], ["q0","q1","q2","q3"]):
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
    for comp, label in zip([traj.wx, traj.wy, traj.wz], ["wx","wy","wz"]):
        ax0.plot(t, comp, label=label)
    ax0.set_ylabel("omega [rad/s]"); ax0.legend(fontsize=8); ax0.grid(True, alpha=0.3)
    T0 = traj.T_rot[0] if traj.T_rot else 1.0
    if abs(T0) > 1e-30:
        ax1.plot(t, [T/T0 - 1.0 for T in traj.T_rot], color="darkorange")
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


# ── Runner principal ──────────────────────────────────────────

def run_all_missions(yaml_path: str, outdir: str = "out/missions") -> bool:
    cfg      = load_cases_yaml(yaml_path)
    missions = cfg.get("missions", [])
    os.makedirs(outdir, exist_ok=True)

    all_ok = True
    for m_cfg in missions:
        name  = m_cfg.get("name",  "<sem-nome>")
        model = m_cfg.get("model", "schwarzschild_equatorial")
        print(f"\n{'='*60}\n==> Missao: {name}  [{model}]\n{'='*60}")

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
    for c in ["mission.yaml", "src/relorbit_py/mission.yaml"]:
        if os.path.exists(c):
            return c
    raise FileNotFoundError(
        "mission.yaml nao encontrado. Use --yaml para especificar o caminho."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="RelOrbit - simulador de missoes")
    parser.add_argument("--yaml", default=None, help="Caminho para mission.yaml")
    parser.add_argument("--out",  default="out/missions", help="Diretorio de saida")
    args = parser.parse_args()
    yaml_path = args.yaml or _find_yaml()
    print(f"YAML: {yaml_path}\nOut:  {args.out}")
    sys.exit(0 if run_all_missions(yaml_path, outdir=args.out) else 1)


if __name__ == "__main__":
    main()