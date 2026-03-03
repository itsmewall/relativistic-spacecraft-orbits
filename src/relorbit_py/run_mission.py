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

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from relorbit_py.simulate import load_cases_yaml, simulate_case
from relorbit_py.mission import run_mission, MissionResult


# ============================================================
# Plots
# ============================================================

def _horizon_radius(params: Dict[str, Any]) -> float:
    M = float(params.get("M", 1.0))
    if "a" in params:
        a = float(params["a"])
        return M + (M * M - a * a) ** 0.5
    return 2.0 * M


def _isco_radius(params: Dict[str, Any]) -> float:
    M = float(params.get("M", 1.0))
    if "a" in params:
        # ISCO prograde Kerr (aproximação Boyer-Lindquist)
        a = float(params["a"])
        Z1 = 1.0 + (1.0 - (a/M)**2)**(1/3) * ((1 + a/M)**(1/3) + (1 - a/M)**(1/3))
        Z2 = (3*(a/M)**2 + Z1**2)**0.5
        return M * (3 + Z2 - ((3 - Z1)*(3 + Z1 + 2*Z2))**0.5)
    return 6.0 * M


def plot_orbit(result: MissionResult, outdir: str, m_cfg: Dict[str, Any]) -> str:
    """
    Plot da órbita com suporte a visualização de horizontes de Schwarzschild e Kerr,
    ergosfera equatorial e marcadores de manobra.
    """
    tau_all, r_all, phi_all, mass_all = result.get_trajectory()
    
    # Se a trajetória estiver vazia (falha imediata), não gera plot
    if len(r_all) == 0:
        return ""

    fig, ax = plt.subplots(figsize=(9, 9))
    ax.set_aspect("equal")

    # --- Configurações Físicas ---
    params = m_cfg.get("params", {})
    M = float(params.get("M", 1.0))
    r_isco = _isco_radius(params)

    if "kerr" in result.model.lower():
        a = float(params.get("a", 0.0))
        # Horizonte de Eventos Externo (r+) e Interno (r-)
        delta_disc = M**2 - a**2
        r_plus = M + np.sqrt(max(0, delta_disc))
        r_minus = M - np.sqrt(max(0, delta_disc))
        # Ergosfera no equador (theta=pi/2) ocorre em r = 2M
        r_ergo = 2.0 * M

        # Desenho das superfícies de Kerr
        ax.add_patch(mpatches.Circle((0, 0), r_ergo, color="yellow", alpha=0.1, 
                                     label=f"Ergosfera r={r_ergo:.2f}M"))
        ax.add_patch(mpatches.Circle((0, 0), r_plus, color="black", alpha=0.4, 
                                     label=f"Horizonte r+={r_plus:.2f}M"))
        if r_minus > 0.1: # Evita desenhar se for muito pequeno (a próximo de M)
            ax.add_patch(mpatches.Circle((0, 0), r_minus, color="red", alpha=0.15, 
                                         linestyle=":", label=f"Horiz. Interno r-={r_minus:.2f}M"))
    else:
        # Schwarzschild Padrão
        r_hor = 2.0 * M
        ax.add_patch(mpatches.Circle((0, 0), r_hor, color="black", alpha=0.3, 
                                     label=f"Horizonte r={r_hor:.2f}M"))

    # Desenho da ISCO (Inner Stable Circular Orbit)
    ax.add_patch(mpatches.Circle((0, 0), r_isco, color="orange", alpha=0.1, linestyle="--",
                                  fill=True, label=f"ISCO r={r_isco:.2f}M"))
    
    # Buraco Negro (Singularidade central simbólica)
    ax.scatter([0], [0], color="black", s=180, zorder=10)

    # --- Plotagem dos Segmentos ---
    cmap = plt.get_cmap("tab10")
    for seg_i, seg in enumerate(result.segments):
        r_s   = np.array(seg.r,   dtype=float)
        phi_s = np.array(seg.phi, dtype=float)
        xs = r_s * np.cos(phi_s)
        ys = r_s * np.sin(phi_s)
        
        # Define label: diferencia entre segmentos de manobra e segmento final
        if seg_i < len(result.maneuver_log):
            label = f"Segmento {seg_i+1}"
        else:
            label = "Segmento final"
            
        ax.plot(xs, ys, color=cmap(seg_i % 10), alpha=0.8, linewidth=1.5, label=label, zorder=5)

    # --- Marcadores de Manobra (Queimas) ---
    for burn in result.maneuver_log:
        xb = burn.r_burn * np.cos(burn.phi_burn)
        yb = burn.r_burn * np.sin(burn.phi_burn)
        
        # Marcador visual de erro ou sucesso
        color_b = "red" if burn.ok else "darkred"
        marker_b = "x" if burn.ok else "X"
        
        ax.scatter([xb], [yb], marker=marker_b, color=color_b, s=150, zorder=15, linewidths=2.5)
        
        # Etiqueta de telemetria da manobra
        label_txt = (
            f"Burn #{burn.index+1} τ={burn.tau_scheduled:.1f}\n"
            f"Δv={burn.dv_ms:.1f} m/s\n"
            f"Δm={burn.fuel_consumed:.2f} kg"
        )
        ax.annotate(label_txt, (xb, yb), textcoords="offset points",
                    xytext=(15, 10), color=color_b, fontsize=8, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=color_b, alpha=0.85),
                    zorder=20)

    status_str = "OK ✓" if result.ok else f"FALHA: {result.abort_reason}"
    ax.set_title(f"RelOrbit Mission Profile: {result.name}\nModelo: {result.model} | {status_str}", 
                 fontsize=12, fontweight='bold', pad=15)
    ax.set_xlabel("x [M]", fontsize=10)
    ax.set_ylabel("y [M]", fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.5)
    
    # Ajusta limites para focar na trajetória mas manter a origem
    max_r = max(r_all) if len(r_all) > 0 else 10.0
    limit = max_r * 1.1
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)

    path = os.path.join(outdir, f"{result.name}_orbit.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_mass(result: MissionResult, outdir: str) -> str:
    """Plot do consumo de propelente ao longo do tempo próprio."""
    tau_all, r_all, phi_all, mass_all = result.get_trajectory()

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=False)

    ax_mass = axes[0]
    ax_r    = axes[1]

    # Massa
    mask = np.isfinite(mass_all)
    ax_mass.plot(tau_all[mask], mass_all[mask], color="green", linewidth=1.5)
    ax_mass.axhline(result.dry_mass, color="gray", linestyle="--", linewidth=1,
                    label=f"Massa seca = {result.dry_mass:.0f} kg")
    for burn in result.maneuver_log:
        ax_mass.axvline(burn.tau_actual, color="red", linestyle=":", alpha=0.6)
        ax_mass.annotate(f"#{burn.index+1}", (burn.tau_actual, burn.mass_before),
                         fontsize=7, color="red", ha="left", va="top")
    ax_mass.set_ylabel("Massa [kg]")
    ax_mass.set_title(f"Consumo de Propelente — {result.name}")
    ax_mass.legend(fontsize=8)
    ax_mass.grid(True, linestyle="--", alpha=0.4)

    # r(τ)
    ax_r.plot(tau_all, r_all, color="steelblue", linewidth=1.2)
    params_M = result.M
    ax_r.axhline(2 * params_M, color="black", linestyle="--", linewidth=1, label="Horizonte 2M")
    ax_r.axhline(6 * params_M, color="orange", linestyle="--", linewidth=1, label="ISCO 6M (Schw)")
    for burn in result.maneuver_log:
        ax_r.axvline(burn.tau_actual, color="red", linestyle=":", alpha=0.6)
    ax_r.set_xlabel("Tempo Próprio τ")
    ax_r.set_ylabel("r [M]")
    ax_r.legend(fontsize=8)
    ax_r.grid(True, linestyle="--", alpha=0.4)

    fig.tight_layout()
    path = os.path.join(outdir, f"{result.name}_mass.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def print_budget_tables(result: MissionResult) -> None:
    """Imprime tabelas de Δv e Massa formatadas."""
    result.print_summary()


# ============================================================
# Runner principal
# ============================================================

def run_all_missions(yaml_path: str, outdir: str = "out/missions") -> bool:
    cfg = load_cases_yaml(yaml_path)
    missions = cfg.get("missions", [])
    os.makedirs(outdir, exist_ok=True)

    all_ok = True
    for m_cfg in missions:
        name = m_cfg.get("name", "<sem-nome>")
        print(f"\n{'='*60}\n==> Missão: {name}\n{'='*60}")

        try:
            result = run_mission(m_cfg, simulate_fn=simulate_case)
            # Imprime as tabelas (o get_trajectory agora é seguro)
            print_budget_tables(result)

            if result.segments:
                orbit_path = plot_orbit(result, outdir, m_cfg)
                mass_path  = plot_mass(result, outdir)
                print(f"\n   Plots: {orbit_path}, {mass_path}")
            else:
                print(f"\n   [AVISO] Gráficos ignorados: sem segmentos gerados.")

            if not result.ok: all_ok = False

        except Exception as ex:
            print(f"   [ERRO] Falha crítica na missão {name}: {ex}")
            all_ok = False
            continue

    return all_ok

# ============================================================
# CLI
# ============================================================

def _find_yaml() -> str:
    candidates = [
        "mission.yaml",
        "src/relorbit_py/mission.yaml",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    raise FileNotFoundError(
        "mission.yaml não encontrado. Use --yaml para especificar o caminho."
    )

def main() -> None:
    parser = argparse.ArgumentParser(description="RelOrbit — simulador de missões")
    parser.add_argument("--yaml",  default=None,          help="Caminho para mission.yaml")
    parser.add_argument("--out",   default="out/missions", help="Diretório de saída")
    args = parser.parse_args()

    yaml_path = args.yaml or _find_yaml()
    print(f"YAML: {yaml_path}")
    print(f"Out:  {args.out}")

    ok = run_all_missions(yaml_path, outdir=args.out)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()