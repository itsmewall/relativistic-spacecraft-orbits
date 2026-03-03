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
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from relorbit_py.simulate import load_cases_yaml, simulate_case
from relorbit_py.mission import run_mission, MissionResult


# ============================================================
# Helpers físicos
# ============================================================

def _horizon_radius(params: Dict[str, Any]) -> float:
    M = float(params.get("M", 1.0))
    if "a" in params:
        a = float(params["a"])
        disc = M * M - a * a
        if disc < 0.0:
            # a > M não-físico (ou unit mismatch). Usa fallback.
            return 2.0 * M
        return M + (disc) ** 0.5
    return 2.0 * M


def _isco_radius(params: Dict[str, Any]) -> float:
    M = float(params.get("M", 1.0))
    if "a" in params:
        # ISCO prograde Kerr (aproximação Boyer-Lindquist)
        a = float(params["a"])
        if M == 0:
            return 0.0
        am = a / M
        # evita NaNs se am>1 por erro de unidade
        am = max(-0.999999, min(0.999999, am))
        Z1 = 1.0 + (1.0 - am**2)**(1/3) * ((1 + am)**(1/3) + (1 - am)**(1/3))
        Z2 = (3*(am**2) + Z1**2)**0.5
        return M * (3 + Z2 - ((3 - Z1)*(3 + Z1 + 2*Z2))**0.5)
    return 6.0 * M


# ============================================================
# Telemetria (NOVO)
# ============================================================

def _central_diff(x: np.ndarray, t: np.ndarray) -> np.ndarray:
    n = len(x)
    out = np.full(n, np.nan, dtype=float)
    if n < 2:
        return out
    for i in range(n):
        if i == 0:
            dt = t[1] - t[0]
            dx = x[1] - x[0]
        elif i == n - 1:
            dt = t[-1] - t[-2]
            dx = x[-1] - x[-2]
        else:
            dt = t[i + 1] - t[i - 1]
            dx = x[i + 1] - x[i - 1]
        out[i] = np.nan if dt == 0 else dx / dt
    return out


def _visibility_2d_occlusion(r: float, phi: float, r_occ: float, observer_dir: Tuple[float, float]=(1.0, 0.0)) -> bool:
    # Posição polar -> cartesiana
    x = r * math.cos(phi)
    y = r * math.sin(phi)

    nx, ny = observer_dir
    nn = math.hypot(nx, ny)
    if nn == 0.0:
        nx, ny = 1.0, 0.0
        nn = 1.0
    nx, ny = nx / nn, ny / nn

    # Linha de visada: p(s)=p+s*n, s>=0
    # Se a distância mínima da semi-reta até a origem < r_occ => ocluído
    pdotn = x * nx + y * ny
    s_star = -pdotn

    # Se o ponto mais próximo cai no próprio ponto (ou “pra trás”), não atravessa o disco à frente
    if s_star <= 0.0:
        return True

    x2 = x + s_star * nx
    y2 = y + s_star * ny
    dmin = math.hypot(x2, y2)

    return dmin >= r_occ


def _extract_full_timeseries(result: MissionResult) -> Dict[str, np.ndarray]:
    """
    Tenta extrair séries completas (concatenadas) incluindo tcoord.
    Se não houver tcoord nos segmentos, retorna apenas tau/r/phi/mass.
    """
    tau_all, r_all, phi_all, mass_all = result.get_trajectory()

    out: Dict[str, np.ndarray] = {
        "tau": np.array(tau_all, dtype=float),
        "r": np.array(r_all, dtype=float),
        "phi": np.array(phi_all, dtype=float),
        "mass": np.array(mass_all, dtype=float),
    }

    # Tenta pegar tcoord por segmento (se existir no MissionResult.segments)
    t_list: List[np.ndarray] = []
    ok = True
    for seg in getattr(result, "segments", []):
        if hasattr(seg, "tcoord"):
            t_list.append(np.array(seg.tcoord, dtype=float))
        elif hasattr(seg, "t"):
            t_list.append(np.array(seg.t, dtype=float))
        else:
            ok = False
            break

    if ok and len(t_list) > 0:
        out["tcoord"] = np.concatenate(t_list)

    # Tenta pegar ut_theory/ut_fd se existir
    ut_list: List[np.ndarray] = []
    ut_ok = True
    for seg in getattr(result, "segments", []):
        if hasattr(seg, "ut_theory"):
            ut_list.append(np.array(seg.ut_theory, dtype=float))
        elif hasattr(seg, "ut_fd"):
            ut_list.append(np.array(seg.ut_fd, dtype=float))
        else:
            ut_ok = False
            break
    if ut_ok and len(ut_list) > 0:
        out["ut"] = np.concatenate(ut_list)

    return out


def plot_telemetry(result: MissionResult, outdir: str, m_cfg: Dict[str, Any], observer_dir=(1.0, 0.0)) -> List[str]:
    """
    Gera:
      - Communication Latency (tcoord - tau) vs tau e vs r
      - dt/dtau vs r
      - freq_ratio ~ 1/(dt/dtau) vs r
      - Visibility flag (0/1) vs tau
    Outputs em: outdir/telemetry_plots
    """
    params = m_cfg.get("params", {})
    r_occ = _horizon_radius(params)

    ts = _extract_full_timeseries(result)
    tau = ts["tau"]
    r = ts["r"]
    phi = ts["phi"]

    # Precisa de tcoord pra latency
    if "tcoord" not in ts:
        print("   [AVISO] Telemetria ignorada: trajetória não possui tcoord nos segmentos.")
        return []

    tcoord = ts["tcoord"]
    if len(tcoord) != len(tau):
        # Se houver diferença por concatenar segmentos diferentes, tenta truncar pelo mínimo
        n = min(len(tcoord), len(tau))
        tau = tau[:n]
        r = r[:n]
        phi = phi[:n]
        tcoord = tcoord[:n]

    # dt/dtau: preferir ut (teoria/FD), senão derivada numérica de tcoord
    if "ut" in ts and len(ts["ut"]) >= len(tau):
        dt_dtau = ts["ut"][:len(tau)]
    else:
        dt_dtau = _central_diff(tcoord, tau)

    latency = tcoord - tau

    # freq ratio proxy
    freq_ratio = np.full(len(dt_dtau), np.nan, dtype=float)
    mask = np.isfinite(dt_dtau) & (dt_dtau != 0.0)
    freq_ratio[mask] = 1.0 / dt_dtau[mask]

    # visibility
    visible = np.array([1.0 if _visibility_2d_occlusion(float(r[i]), float(phi[i]), float(r_occ), observer_dir=observer_dir) else 0.0
                        for i in range(len(r))], dtype=float)

    tele_dir = os.path.join(outdir, "telemetry_plots")
    os.makedirs(tele_dir, exist_ok=True)

    paths: List[str] = []

    # 1) latency vs tau
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(tau, latency, linewidth=1.5)
    ax.set_xlabel("Tempo próprio τ")
    ax.set_ylabel("tcoord - τ")
    ax.set_title(f"{result.name} — Communication Latency vs τ")
    ax.grid(True, linestyle="--", alpha=0.4)
    p = os.path.join(tele_dir, f"{result.name}_communication_latency_vs_tau.png")
    fig.savefig(p, dpi=160, bbox_inches="tight")
    plt.close(fig)
    paths.append(p)

    # 2) latency vs r
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(r, latency, linewidth=1.5)
    ax.set_xlabel("r [M]")
    ax.set_ylabel("tcoord - τ")
    ax.set_title(f"{result.name} — Communication Latency vs r")
    ax.grid(True, linestyle="--", alpha=0.4)
    p = os.path.join(tele_dir, f"{result.name}_communication_latency_vs_r.png")
    fig.savefig(p, dpi=160, bbox_inches="tight")
    plt.close(fig)
    paths.append(p)

    # 3) dt/dtau vs r
    fig, ax = plt.subplots(figsize=(11, 4.5))
    m2 = np.isfinite(dt_dtau)
    ax.plot(r[m2], dt_dtau[m2], linewidth=1.2)
    ax.set_xlabel("r [M]")
    ax.set_ylabel("dt/dτ")
    ax.set_title(f"{result.name} — dt/dτ vs r")
    ax.grid(True, linestyle="--", alpha=0.4)
    p = os.path.join(tele_dir, f"{result.name}_dt_dtau_vs_r.png")
    fig.savefig(p, dpi=160, bbox_inches="tight")
    plt.close(fig)
    paths.append(p)

    # 4) freq ratio vs r
    fig, ax = plt.subplots(figsize=(11, 4.5))
    m3 = np.isfinite(freq_ratio)
    ax.plot(r[m3], freq_ratio[m3], linewidth=1.2)
    ax.set_xlabel("r [M]")
    ax.set_ylabel("freq_ratio ~ 1/(dt/dτ)")
    ax.set_title(f"{result.name} — Redshift/Doppler proxy vs r")
    ax.grid(True, linestyle="--", alpha=0.4)
    p = os.path.join(tele_dir, f"{result.name}_freq_ratio_vs_r.png")
    fig.savefig(p, dpi=160, bbox_inches="tight")
    plt.close(fig)
    paths.append(p)

    # 5) visibility vs tau (0/1)
    fig, ax = plt.subplots(figsize=(11, 3.5))
    ax.plot(tau, visible, linewidth=1.2)
    ax.set_xlabel("Tempo próprio τ")
    ax.set_ylabel("visível (1) / oculto (0)")
    ax.set_yticks([0, 1])
    ax.set_title(f"{result.name} — Visibility vs τ (oclusão geométrica)")
    ax.grid(True, linestyle="--", alpha=0.4)
    p = os.path.join(tele_dir, f"{result.name}_visibility_vs_tau.png")
    fig.savefig(p, dpi=160, bbox_inches="tight")
    plt.close(fig)
    paths.append(p)

    # Print rápido de resumo (sem depender de report.json)
    lat_end = float(latency[-1]) if len(latency) else float("nan")
    dt_min = float(np.nanmin(dt_dtau)) if np.isfinite(dt_dtau).any() else float("nan")
    dt_max = float(np.nanmax(dt_dtau)) if np.isfinite(dt_dtau).any() else float("nan")
    vis_frac = float(np.mean(visible)) if len(visible) else float("nan")

    print(f"   Telemetria: latency_end={lat_end:.6g} | dt/dτ[min,max]=[{dt_min:.6g}, {dt_max:.6g}] | visible_frac={vis_frac:.3f}")

    return paths


# ============================================================
# Plots (órbita e massa) — já existentes
# ============================================================

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
        if r_minus > 0.1:  # Evita desenhar se for muito pequeno (a próximo de M)
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
        r_s = np.array(seg.r, dtype=float)
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

        color_b = "red" if burn.ok else "darkred"
        marker_b = "x" if burn.ok else "X"

        ax.scatter([xb], [yb], marker=marker_b, color=color_b, s=150, zorder=15, linewidths=2.5)

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
    ax_r = axes[1]

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
# Plot de Invariante Físico (verificação de erro numérico)
# ============================================================

# Bandas de qualidade: (limite_superior_relativo, cor, rótulo)
_QUALITY_BANDS = [
    (1e-10, "#2ecc71", "Machine ε"),   # verde escuro
    (1e-7,  "#a8e6a3", "Excellent"),    # verde claro
    (1e-4,  "#fff3a3", "Good"),         # amarelo
    (1e-1,  "#ffc880", "Warning"),      # laranja
    (1e9,   "#ff9999", "Poor"),         # vermelho
]

_QUALITY_COLOR = {
    "excellent": "#2ecc71",
    "good":      "#f39c12",
    "warning":   "#e67e22",
    "poor":      "#e74c3c",
    "no_data":   "#95a5a6",
}


def plot_invariant(result: MissionResult, outdir: str, m_cfg: Dict[str, Any]) -> str:
    """
    Gera o gráfico de conservação do invariante ε = pr² + V_eff − E².

    Layout — 3 painéis empilhados:

    ┌─────────────────────────────────────────────┐
    │ Painel 1: ε(τ) — valor absoluto             │
    │   • Geodésica: ideal = 0; desvio = erro RK4 │
    │   • Low-Thrust: ε ≠ 0 durante empuxo        │
    │   • Linhas verticais: τ das manobras         │
    │   • Anotações: salto Δε em cada manobra      │
    ├─────────────────────────────────────────────┤
    │ Painel 2: |Δε_intra(τ)| — escala log        │
    │   Deriva DENTRO de cada segmento:            │
    │     Δε_intra = |ε(τ) − ε(τ_seg_início)|     │
    │   Isso é erro numérico puro do RK4,          │
    │   independente do modelo (geo ou LT).        │
    │   Bandas coloridas de qualidade.             │
    ├─────────────────────────────────────────────┤
    │ Painel 3: |Δε_intra| / E²  (erro relativo)  │
    │   Mesmas bandas. Badge de qualidade.         │
    └─────────────────────────────────────────────┘
    """
    tau_eps, eps_all, boundaries = result.get_epsilon()
    stats = result.epsilon_stats()

    if len(eps_all) == 0 or stats.get("n_segments", 0) == 0:
        print(f"   [invariante] Sem dados de epsilon para {result.name}.")
        return ""

    is_lt = "lowthrust" in result.model.lower() or "_lt" in result.model.lower()
    E2    = stats["eps_E2_scale"]

    # ── Calcular Δε intrassegmento ──────────────────────────────
    seg_starts = list(boundaries) + [len(eps_all)]
    drift_abs  = np.full(len(eps_all), np.nan)  # |ε(τ) − ε_seg0|
    drift_rel  = np.full(len(eps_all), np.nan)  # /E²

    for k in range(len(boundaries)):
        i0 = seg_starts[k]
        i1 = seg_starts[k + 1]
        seg = eps_all[i0:i1]
        if len(seg) == 0:
            continue
        seg0 = seg[0]  # valor inicial do segmento (referência)
        d = np.abs(seg - seg0)
        drift_abs[i0:i1] = d
        drift_rel[i0:i1] = d / max(abs(E2), 1e-14)

    # ── Figura 3 painéis ────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(12, 10),
                             gridspec_kw={"height_ratios": [2, 1.8, 1.8]})
    fig.subplots_adjust(hspace=0.42)

    ax_eps, ax_drift, ax_rel = axes

    # ── Cores de segmento para fundo ────────────────────────────
    seg_bg_colors = ["#f0f4ff", "#fff8f0"]
    for k in range(len(boundaries)):
        i0 = seg_starts[k]
        i1 = seg_starts[k + 1]
        if i1 <= i0:
            continue
        t0 = tau_eps[i0]
        t1 = tau_eps[i1 - 1]
        for ax in axes:
            ax.axvspan(t0, t1, color=seg_bg_colors[k % 2], alpha=0.45, zorder=0)

    # ── Manobra: linhas verticais e anotações de salto ──────────
    for burn in result.maneuver_log:
        tau_b = burn.tau_actual
        # Salto em ε nesta manobra
        jump_idx = burn.index
        jump_val = (stats["jump_at_maneuver"][jump_idx]
                    if jump_idx < len(stats["jump_at_maneuver"]) else None)
        for ax in axes:
            ax.axvline(tau_b, color="#c0392b", linestyle="--", linewidth=1.1,
                       alpha=0.7, zorder=3)
        if jump_val is not None:
            ax_eps.annotate(
                f"Burn #{burn.index+1}\nΔε={jump_val:.1e}",
                xy=(tau_b, 0), xycoords=("data", "axes fraction"),
                xytext=(6, 0.62), textcoords=("offset points", "axes fraction"),
                fontsize=7, color="#c0392b",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#c0392b", alpha=0.85),
                zorder=5,
            )

    # ── Painel 1: ε(τ) ──────────────────────────────────────────
    finite_m = np.isfinite(eps_all)
    ax_eps.plot(tau_eps[finite_m], eps_all[finite_m],
                color="#2c3e50", linewidth=1.3, zorder=4, label="ε(τ)")
    ax_eps.axhline(0.0, color="gray", linewidth=0.9, linestyle=":", zorder=2)

    if is_lt:
        subtitle = "ε ≠ 0 durante empuxo (física LT)"
    else:
        subtitle = "ε → 0 ideal (geodésica); desvio = erro RK4"

    ax_eps.set_ylabel("ε = pr² + V_eff − E²", fontsize=9)
    ax_eps.set_title(
        f"Conservação do Invariante de Massa — {result.name}\n"
        f"Modelo: {result.model}  |  {subtitle}",
        fontsize=10, fontweight="bold",
    )
    ax_eps.legend(fontsize=8, loc="upper left")
    ax_eps.grid(True, linestyle=":", alpha=0.4, zorder=1)
    ax_eps.tick_params(labelbottom=False)

    # ── Painel 2: |Δε_intra| log ─────────────────────────────────
    valid = np.isfinite(drift_abs) & (drift_abs > 0)
    if valid.any():
        ax_drift.semilogy(tau_eps[valid], drift_abs[valid],
                          color="#8e44ad", linewidth=1.2, zorder=4,
                          label="|Δε_intra(τ)|")
    # Bandas de qualidade (valores absolutos escalados por E²)
    y_lo, y_hi = 1e-15, max(float(np.nanmax(drift_abs[valid])) * 5 if valid.any() else 1e-3, 1e-3)
    for (lim, color, name) in _QUALITY_BANDS:
        band_abs = lim * E2
        if band_abs > y_lo:
            ax_drift.axhline(band_abs, color=color, linestyle="--",
                             linewidth=0.9, alpha=0.8, zorder=2)
            ax_drift.text(tau_eps[-1], band_abs, f" {name}",
                          va="bottom", fontsize=7, color="gray", zorder=3)

    ax_drift.set_ylim(y_lo, y_hi * 2)
    ax_drift.set_ylabel("|Δε_intra| [abs]", fontsize=9)
    ax_drift.set_title("Deriva intrassegmento — erro numérico puro do RK4", fontsize=9)
    ax_drift.legend(fontsize=8, loc="upper left")
    ax_drift.grid(True, which="both", linestyle=":", alpha=0.3, zorder=1)
    ax_drift.tick_params(labelbottom=False)

    # ── Painel 3: |Δε_intra| / E² ────────────────────────────────
    valid_r = np.isfinite(drift_rel) & (drift_rel > 0)
    if valid_r.any():
        ax_rel.semilogy(tau_eps[valid_r], drift_rel[valid_r],
                        color="#2980b9", linewidth=1.2, zorder=4,
                        label="|Δε_intra| / E²")
    y_lo_r = 1e-16
    y_hi_r = max(float(np.nanmax(drift_rel[valid_r])) * 5 if valid_r.any() else 1e-3, 1e-3)
    for (lim, color, name) in _QUALITY_BANDS:
        if lim > y_lo_r:
            ax_rel.axhline(lim, color=color, linestyle="--",
                           linewidth=0.9, alpha=0.8, zorder=2)
            ax_rel.text(tau_eps[-1], lim, f" {name}",
                        va="bottom", fontsize=7, color="gray", zorder=3)
    ax_rel.set_ylim(y_lo_r, y_hi_r * 2)
    ax_rel.set_xlabel("Tempo próprio τ [M]", fontsize=9)
    ax_rel.set_ylabel("|Δε_intra| / E²", fontsize=9)
    ax_rel.set_title("Erro relativo (adimensional)", fontsize=9)
    ax_rel.legend(fontsize=8, loc="upper left")
    ax_rel.grid(True, which="both", linestyle=":", alpha=0.3, zorder=1)

    # ── Badge de qualidade ────────────────────────────────────────
    qlabel = stats.get("quality_label", "no_data").upper()
    qcolor = _QUALITY_COLOR.get(stats.get("quality_label", "no_data"), "gray")
    dr_max = stats.get("drift_rel_max", float("nan"))
    badge_txt = (
        f"Qualidade: {qlabel}\n"
        f"max|Δε|/E² = {dr_max:.2e}\n"
        f"RMS ε = {stats.get('eps_rms', float('nan')):.2e}"
    )
    ax_rel.text(
        0.98, 0.97, badge_txt,
        transform=ax_rel.transAxes,
        ha="right", va="top", fontsize=8,
        bbox=dict(boxstyle="round,pad=0.4", fc=qcolor, alpha=0.25, ec=qcolor),
        zorder=6,
    )

    path = os.path.join(outdir, f"{result.name}_invariant.png")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


# ============================================================
# Mapa de Visibilidade Relativística — Cone de Escape de Fótons
# ============================================================
#
# FÍSICA CENTRAL — Schwarzschild:
#   O fóton emitido de r com ângulo α (da radial de saída) tem parâmetro de impacto
#     b(α, r) = (r / √A) · sin(α)    A = 1 − 2M/r
#
#   Para r > 3M (fora da esfera de fótons):
#     • α ∈ [0, π/2): fóton SAINDO — sempre escapa  ✓
#     • α ∈ (π/2, π]: fóton ENTRANDO — escapa se  b = (r/√A)·sin(π−α) > b_crit
#       Condição de captura: sin(π − α) < b_crit·√A/r
#       → Cone de sombra: ângulo θ_shadow = arcsin(b_crit·√A/r)  medido da anti-radial
#       → Visível se:  α_eff  <  π − θ_shadow
#
#   b_crit Schwarzschild: 3√3·M  (máximo do potencial em r=3M)
#   b_crit Kerr (Bardeen 1973):
#     r_ph± = 2M[1 + cos(2/3 arccos(∓a/M))]   (raios prograde/retrograde)
#     b_crit = (r_ph² + a² ± a·√Δ_ph) / (±√Δ_ph + a)
#
# ABERRAÇÃO RELATIVÍSTICA:
#   A espaçonave em movimento com velocidade local β altera o ângulo de emissão:
#     cos(α_ab) = (cos(α_eff) − β_∥) / (1 − β_∥·cos(α_eff))
#   onde β_∥ = componente de β na direção de emissão (local tetrada Schwarzschild).

_B_CRIT_SCHW_PER_M = 3.0 * math.sqrt(3.0)   # b_crit = 3√3·M


def _photon_b_crit_kerr(M: float, a: float) -> Tuple[float, float]:
    """
    Parâmetros de impacto críticos para Kerr equatorial via fórmula de Bardeen.

    r_ph± = 2M[1 + cos(2/3·arccos(∓a/M))]   (prograde/retrograde)

    A esfera prograde tem menor r (co-rotante com o BH).
    A esfera retrograde tem maior r (contra-rotante).

    No raio da esfera de fótons r_ph, com Δ_ph = r_ph² − 2M·r_ph + a²:
      b_pro =  (r_ph_pro² + a² + a·√Δ_pro) / ( √Δ_pro + a)   [prograde, b > 0]
      b_ret = |(r_ph_ret² + a² − a·√Δ_ret) / (−√Δ_ret + a)|  [retrograde, |b|]

    Retorna (b_crit_prograde, b_crit_retrograde_abs).
    """
    if abs(a) < M * 1e-8:
        bc = _B_CRIT_SCHW_PER_M * M
        return bc, bc

    am = max(-0.9999, min(0.9999, a / M))

    # Raios das esferas de fótons (fórmula analítica de Bardeen 1973)
    r_ph_pro = 2.0 * M * (1.0 + math.cos(2.0/3.0 * math.acos(-am)))  # prograde (menor r)
    r_ph_ret = 2.0 * M * (1.0 + math.cos(2.0/3.0 * math.acos(+am)))  # retrograde (maior r)

    def _b_bardeen(r_ph: float, sign: float) -> float:
        """
        sign = +1 → prograde (b positivo, co-rotante)
        sign = −1 → retrograde (b negativo, contra-rotante)
        Retorna o valor absoluto do parâmetro crítico.
        """
        Delta = r_ph * r_ph - 2.0 * M * r_ph + a * a
        sqD = math.sqrt(max(Delta, 1e-14))
        num = r_ph * r_ph + a * a + sign * a * sqD
        den = sign * sqD + a
        return abs(num / den) if abs(den) > 1e-14 else _B_CRIT_SCHW_PER_M * M

    return _b_bardeen(r_ph_pro, +1.0), _b_bardeen(r_ph_ret, -1.0)


def _shadow_half_angle(r: float, M: float, a: float = 0.0) -> float:
    """
    Meia-abertura θ_shadow do cone de sombra medida desde a direção anti-radial.

    θ_shadow = arcsin(b_crit_eff · √A / r)

    Schwarzschild: b_crit_eff = 3√3·M  (exato)
    Kerr:          b_crit_eff = √(b_pro · b_ret)  (média geométrica — sombra assimétrica
                  é tratada como isotrópica nesta aproximação)

    Limites físicos:
      r → ∞:  θ_shadow → 0     (sombra desprezível de longe)
      r = 3M: θ_shadow = π/2   (esfera de fótons — metade do céu capturada)
      r → 2M: A → 0 → θ_shadow → π/2  (horizonte — sombra total)
    """
    A = 1.0 - 2.0 * M / max(r, 2.0 * M + 1e-10)
    if A <= 1e-12:
        return math.pi / 2.0

    if abs(a) < M * 1e-8:
        b_eff = _B_CRIT_SCHW_PER_M * M
    else:
        b_pro, b_ret = _photon_b_crit_kerr(M, a)
        b_eff = math.sqrt(b_pro * b_ret)   # média geométrica

    sinθ = b_eff * math.sqrt(A) / r
    return math.asin(min(sinθ, 1.0))


def _shadow_half_angle_vec(r: np.ndarray, M: float, a: float = 0.0) -> np.ndarray:
    """Versão vetorizada de _shadow_half_angle."""
    return np.array([_shadow_half_angle(float(ri), M, a) for ri in r])


def _effective_angle_to_observer(phi_sc: float, phi_obs: float = 0.0) -> float:
    """
    Ângulo de emissão α ∈ [0, π] medido desde a radial de saída até o observador distante.

    Geometria: observador em φ_obs a r → ∞, espaçonave em φ_sc.
      Direção ao observador (aprox. plana): (cos φ_obs, sin φ_obs)
      Radial de saída: (cos φ_sc, sin φ_sc)
      cos(α) = cos(φ_obs − φ_sc)   →   α = |φ_obs − φ_sc| mod 2π, ∈ [0, π]
    """
    delta = (phi_obs - phi_sc + math.pi) % (2.0 * math.pi) - math.pi
    return abs(delta)


def _apply_aberration(alpha_eff: float, pr: float, L: float,
                      E: float, r: float, M: float, a: float) -> float:
    """
    Corrige o ângulo de emissão via aberração relativística (local tetrada).

    Velocidade da espaçonave no referencial do observador estático:
      v^(r̂) = pr·√A / E      (radial)
      v^(φ̂) = L·√A / (r·E)   (tangencial, equatorial)

    Projeção de β na direção de emissão:
      β_∥ = v^(r̂)·cos(α) + v^(φ̂)·sin(α)

    Aberração SR:
      cos(α') = (cos α − β_∥) / (1 − β_∥·cos α)
    """
    A = max(1.0 - 2.0 * M / r, 1e-14)
    if abs(a) > 1e-8:
        Delta = r*r - 2.0*M*r + a*a
        A_eff = max(Delta, 1e-14) / (r*r)   # approximate lapse² for Kerr equatorial
    else:
        A_eff = A
    E_s = abs(E) if abs(E) > 1e-12 else 1e-12
    v_r   = pr  * math.sqrt(A_eff) / E_s
    v_phi = L   * math.sqrt(A_eff) / (r * E_s)
    beta_par = v_r * math.cos(alpha_eff) + v_phi * math.sin(alpha_eff)
    beta_par = max(-0.9999, min(0.9999, beta_par))
    denom = 1.0 - beta_par * math.cos(alpha_eff)
    if abs(denom) < 1e-12:
        return alpha_eff
    cos_ab = max(-1.0, min(1.0, (math.cos(alpha_eff) - beta_par) / denom))
    return math.acos(cos_ab)


def _compute_gr_visibility(
    r_arr: np.ndarray, phi_arr: np.ndarray,
    M: float, a: float, phi_obs: float,
    pr_arr: Optional[np.ndarray] = None,
    L_arr: Optional[np.ndarray] = None,
    E_arr: Optional[np.ndarray] = None,
    use_aberration: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calcula, ponto a ponto ao longo de uma trajetória:
      alpha_eff  — ângulo de emissão ∈ [0, π]  (com aberração, se ativado)
      alpha_crit — ângulo limiar    π − θ_shadow(r)
      visible    — bool (True = sinal pode escapar)

    Retorna (alpha_eff, alpha_crit, visible) como arrays de float/bool.
    """
    n = len(r_arr)
    alpha_eff  = np.zeros(n)
    alpha_crit = np.zeros(n)
    visible    = np.zeros(n, dtype=bool)

    use_pr  = pr_arr  is not None
    use_L   = L_arr   is not None
    use_E   = E_arr   is not None

    for i in range(n):
        r   = float(r_arr[i])
        phi = float(phi_arr[i])
        pr  = float(pr_arr[i])  if use_pr  else 0.0
        L   = float(L_arr[i])   if use_L   else 0.0
        E   = float(E_arr[i])   if use_E   else 1.0

        α = _effective_angle_to_observer(phi, phi_obs)
        if use_aberration:
            α = _apply_aberration(α, pr, L, E, r, M, a)
        θ_sh = _shadow_half_angle(r, M, a)
        α_c  = math.pi - θ_sh

        alpha_eff[i]  = α
        alpha_crit[i] = α_c
        visible[i]    = (α < α_c)

    return alpha_eff, alpha_crit, visible


def plot_visibility_map(
    result: MissionResult, outdir: str, m_cfg: Dict[str, Any],
    phi_obs: float = 0.0,
) -> List[str]:
    """
    Análise de visibilidade relativística baseada no Cone de Escape de Fótons.

    Gera dois arquivos:

    ① {name}_visibility_gr.png  — 3 painéis:
       ┌──────────────────────────────────────────────┐
       │ Painel A (esquerda, grande)                  │
       │   Órbita colorida por visibilidade GR        │
       │   Verde = visível; Vermelho = no cone sombra  │
       │   Círculos: horizonte, esfera de fótons, ISCO│
       │   Marcador: observador e cone sombra atual   │
       ├─────────────────────┬────────────────────────┤
       │ Painel B (inf-esq)  │ Painel C (inf-dir)     │
       │ α_eff(τ) vs α_crit  │ θ_shadow(r) teórico    │
       │ Timeline: shadow    │ Curva canônica +       │
       │ entry/exit events   │ pontos da trajetória   │
       └─────────────────────┴────────────────────────┘

    ② {name}_visibility_polar.png — heatmap polar:
       Grade (r × φ) com visibilidade GR, trajetória sobreposta.
    """
    params   = m_cfg.get("params", {})
    M        = float(params.get("M",  1.0))
    a        = float(params.get("a",  0.0))
    E_init   = float(params.get("E",  result.E_initial))
    L_init   = float(params.get("L",  result.L_initial))
    is_kerr  = abs(a) > 1e-10
    is_lt    = "lowthrust" in result.model.lower() or "_lt" in result.model.lower()

    r_hor    = _horizon_radius(params)
    r_isco   = _isco_radius(params)
    # Prograde photon sphere (co-rotating, smaller r — most relevant for orbital dynamics)
    # r_ph_pro = 2M[1 + cos(2/3 · arccos(−a/M))]   [Bardeen 1973]
    r_phot   = 3.0 * M if not is_kerr else 2.0 * M * (
        1.0 + math.cos(2.0/3.0 * math.acos(max(-0.9999, min(0.9999, -a/M)))))

    # ── Concatenar dados de trajetória ────────────────────────
    tau_all, r_all, phi_all, mass_all = result.get_trajectory()

    if len(r_all) == 0:
        print(f"   [visibilidade] Sem trajetória em {result.name}.")
        return []

    # Extrair pr, L, E por ponto (para aberração)
    pr_all, L_all, E_all = [], [], []
    for seg in result.segments:
        seg_pr = np.array(seg.pr, dtype=float) if hasattr(seg, "pr") else np.zeros(len(seg.tau))
        pr_all.append(seg_pr)
        # L: se LT, é série; se geodésica, escalar
        if hasattr(seg, "L") and hasattr(seg.L, "__len__"):
            L_all.append(np.array(seg.L, dtype=float))
        else:
            L_val = getattr(seg, "L", L_init)
            L_all.append(np.full(len(seg.tau), float(L_val)))
        if hasattr(seg, "E") and hasattr(seg.E, "__len__"):
            E_all.append(np.array(seg.E, dtype=float))
        else:
            E_val = getattr(seg, "E", E_init)
            E_all.append(np.full(len(seg.tau), float(E_val)))

    pr_all = np.concatenate(pr_all)
    L_all  = np.concatenate(L_all)
    E_all  = np.concatenate(E_all)

    # Alinhar comprimentos (segmentos podem ter acumulado off-by-one)
    n = min(len(tau_all), len(r_all), len(phi_all), len(pr_all), len(L_all), len(E_all))
    tau_all, r_all, phi_all = tau_all[:n], r_all[:n], phi_all[:n]
    pr_all, L_all, E_all    = pr_all[:n], L_all[:n], E_all[:n]

    # ── Calcular visibilidade GR ──────────────────────────────
    alpha_eff, alpha_crit, visible = _compute_gr_visibility(
        r_all, phi_all, M, a, phi_obs,
        pr_arr=pr_all, L_arr=L_all, E_arr=E_all,
        use_aberration=True,
    )

    vis_frac = float(np.mean(visible))

    # ── Detectar eventos de entrada/saída do cone de sombra ──
    shadow_events: List[Dict] = []
    was_vis = bool(visible[0])
    for i in range(1, n):
        now_vis = bool(visible[i])
        if now_vis != was_vis:
            shadow_events.append({
                "tau": float(tau_all[i]),
                "r":   float(r_all[i]),
                "phi": float(phi_all[i]),
                "kind": "shadow_exit" if now_vis else "shadow_entry",
            })
            was_vis = now_vis

    # ════════════════════════════════════════════════════════
    # FIGURA 1: 3 painéis
    # ════════════════════════════════════════════════════════
    fig = plt.figure(figsize=(16, 10))
    gs  = fig.add_gridspec(2, 2, width_ratios=[1.5, 1], hspace=0.35, wspace=0.30)
    ax_orb   = fig.add_subplot(gs[:, 0])   # esquerda: órbita inteira
    ax_time  = fig.add_subplot(gs[0, 1])   # direita cima: timeline
    ax_theory= fig.add_subplot(gs[1, 1])   # direita baixo: curva teórica

    model_lbl = ("Kerr" if is_kerr else "Schwarzschild") + (" LT" if is_lt else "")
    fig.suptitle(
        f"Visibilidade Relativística — Cone de Escape de Fótons\n"
        f"{result.name}  |  Modelo: {model_lbl}  |  "
        f"Observador: φ_obs = {math.degrees(phi_obs):.1f}°  |  "
        f"Visível: {vis_frac*100:.1f}% do tempo",
        fontsize=10, fontweight="bold", y=0.99,
    )

    # ── Painel A: Órbita colorida ─────────────────────────────
    ax_orb.set_aspect("equal")

    # Zonas físicas (círculos)
    for r_circ, color, alpha_c, lw, label_c in [
        (r_hor,  "black",  0.5, 2.0, f"Horizonte r={r_hor:.2f}M"),
        (r_phot, "#9b59b6", 0.25, 1.4, f"Esfera de fótons r_ph={r_phot:.2f}M"),
        (r_isco, "darkorange", 0.15, 1.2, f"ISCO r={r_isco:.2f}M"),
    ]:
        ax_orb.add_patch(mpatches.Circle((0,0), r_circ, fill=(r_circ <= r_hor),
                                          color=color, alpha=alpha_c, linewidth=lw,
                                          linestyle="-" if r_circ == r_hor else "--",
                                          label=label_c, zorder=3))
    if is_kerr:
        r_ergo = 2.0 * M
        ax_orb.add_patch(mpatches.Circle((0,0), r_ergo, fill=False, color="gold",
                                          alpha=0.5, linewidth=1.0, linestyle=":",
                                          label=f"Ergosfera r={r_ergo:.2f}M", zorder=2))

    # Singularidade
    ax_orb.scatter([0], [0], color="black", s=200, zorder=10)

    # Trajetória: colorida por visibilidade
    x_all = r_all * np.cos(phi_all)
    y_all = r_all * np.sin(phi_all)

    # Traça segmentos contíguos por status de visibilidade
    for seg_vis, seg_color, seg_lw, seg_lbl in [
        (True,  "#27ae60", 1.8, "Visível (sinal escapa)"),
        (False, "#e74c3c", 1.8, "Sombra GR (fóton capturado)"),
    ]:
        mask = visible == seg_vis
        if not mask.any():
            continue
        # Plot segmentos contínuos para evitar linhas cruzando gaps
        groups = np.split(np.where(mask)[0],
                          np.where(np.diff(np.where(mask)[0]) > 1)[0] + 1)
        for g in groups:
            if len(g) < 2:
                continue
            ax_orb.plot(x_all[g], y_all[g], color=seg_color,
                        linewidth=seg_lw, alpha=0.85, zorder=5,
                        label=seg_lbl if g is groups[0] else "_nolegend_")

    # Marcador de posição do observador (raio visual)
    r_obs_marker = max(r_all) * 1.05 if len(r_all) else 20.0
    ax_orb.annotate("", xy=(r_obs_marker * math.cos(phi_obs), r_obs_marker * math.sin(phi_obs)),
                    xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color="#2980b9",
                                    lw=1.2, mutation_scale=14))
    ax_orb.text(r_obs_marker * math.cos(phi_obs) * 0.85,
                r_obs_marker * math.sin(phi_obs) * 0.85 + 0.4,
                "Obs.", color="#2980b9", fontsize=8, ha="center")

    # Eventos de shadow entry/exit
    for ev in shadow_events:
        xev = ev["r"] * math.cos(ev["phi"])
        yev = ev["r"] * math.sin(ev["phi"])
        clr = "#27ae60" if ev["kind"] == "shadow_exit" else "#e74c3c"
        mrkr = "^" if ev["kind"] == "shadow_exit" else "v"
        ax_orb.scatter([xev], [yev], marker=mrkr, color=clr, s=60, zorder=12,
                       edgecolors="white", linewidths=0.6)

    # Manobras
    for burn in result.maneuver_log:
        xb = burn.r_burn * math.cos(burn.phi_burn)
        yb = burn.r_burn * math.sin(burn.phi_burn)
        ax_orb.scatter([xb], [yb], marker="x", color="#c0392b", s=120, zorder=15, linewidths=2.5)

    lim = max(r_all) * 1.12 if len(r_all) else 20.0
    ax_orb.set_xlim(-lim, lim); ax_orb.set_ylim(-lim, lim)
    ax_orb.set_xlabel("x [M]"); ax_orb.set_ylabel("y [M]")
    ax_orb.set_title("Trajetória — Verde: visível / Vermelho: sombra GR", fontsize=9)
    # Reduzir legenda para não poluir
    handles, labels = ax_orb.get_legend_handles_labels()
    unique = {l: h for h, l in zip(handles, labels)}  # deduplicate
    ax_orb.legend(unique.values(), unique.keys(), fontsize=7, loc="upper right",
                  framealpha=0.9, ncol=1)
    ax_orb.grid(True, linestyle=":", alpha=0.4)

    # ── Painel B: Timeline α_eff(τ) vs α_crit(τ) ─────────────
    ax_time.plot(tau_all, np.degrees(alpha_eff),  color="#2c3e50", linewidth=1.3,
                 label="α_eff(τ) — ângulo de emissão")
    ax_time.plot(tau_all, np.degrees(alpha_crit), color="#e74c3c", linewidth=1.5,
                 linestyle="--", label="α_crit = π − θ_shadow (limiar)")
    ax_time.fill_between(tau_all,
                         np.degrees(alpha_crit), 180.0,
                         color="#e74c3c", alpha=0.12, label="Zona de sombra")
    ax_time.fill_between(tau_all,
                         np.degrees(alpha_eff), np.degrees(alpha_crit),
                         where=(~visible), color="#e74c3c", alpha=0.30)

    # Eventos no timeline
    for ev in shadow_events:
        clr = "#27ae60" if ev["kind"] == "shadow_exit" else "#e74c3c"
        ax_time.axvline(ev["tau"], color=clr, linewidth=1.0, linestyle=":", alpha=0.7)

    ax_time.set_xlim(tau_all[0], tau_all[-1])
    ax_time.set_ylim(0, 185)
    ax_time.set_yticks([0, 45, 90, 135, 180])
    ax_time.set_xlabel("Tempo próprio τ [M]", fontsize=8)
    ax_time.set_ylabel("Ângulo [°]", fontsize=8)
    ax_time.set_title("α_eff vs limiar de sombra GR", fontsize=9)
    ax_time.legend(fontsize=7, loc="upper left")
    ax_time.grid(True, linestyle=":", alpha=0.4)

    # ── Painel C: Curva teórica θ_shadow(r) ──────────────────
    r_min_teo = r_hor * 1.01
    r_max_teo = max(r_all) * 1.1 if len(r_all) else 30.0 * M
    r_teo = np.linspace(r_min_teo, r_max_teo, 500)
    theta_teo = np.degrees(_shadow_half_angle_vec(r_teo, M, a))

    ax_theory.plot(r_teo / M, theta_teo, color="#8e44ad", linewidth=2.0,
                   label=f"θ_shadow(r) — {'Kerr (b_eff)' if is_kerr else 'Schwarzschild (exato)'}")

    # Pontos da trajetória
    ax_theory.scatter(r_all / M, np.degrees(_shadow_half_angle_vec(r_all, M, a)),
                      c=np.array(visible, dtype=float), cmap="RdYlGn",
                      s=4, alpha=0.4, zorder=4, label="Trajetória")

    # Referências verticais
    for r_ref, label_ref, color_ref in [
        (r_hor,  f"r_hor={r_hor:.2f}M", "black"),
        (r_phot, f"r_ph={r_phot:.2f}M", "#9b59b6"),
        (r_isco, f"ISCO={r_isco:.2f}M",  "darkorange"),
    ]:
        ax_theory.axvline(r_ref/M, color=color_ref, linewidth=0.9, linestyle="--", alpha=0.6)
        ax_theory.text(r_ref/M + 0.1, 5, label_ref, fontsize=7, color=color_ref,
                       rotation=90, va="bottom")

    ax_theory.axhline(45,  color="gray", linewidth=0.7, linestyle=":", alpha=0.5)
    ax_theory.axhline(90,  color="#e74c3c", linewidth=0.9, linestyle="--", alpha=0.6,
                      label="90° (metade do céu)")
    ax_theory.set_xlabel("r / M", fontsize=8)
    ax_theory.set_ylabel("θ_shadow [°]", fontsize=8)
    ax_theory.set_title("Meia-abertura do cone de sombra vs r", fontsize=9)
    ax_theory.set_ylim(0, 92)
    ax_theory.legend(fontsize=7, loc="upper right")
    ax_theory.grid(True, linestyle=":", alpha=0.4)

    paths = []
    p1 = os.path.join(outdir, f"{result.name}_visibility_gr.png")
    fig.savefig(p1, dpi=180, bbox_inches="tight")
    plt.close(fig)
    paths.append(p1)

    # ════════════════════════════════════════════════════════
    # FIGURA 2: Heatmap polar de visibilidade
    # ════════════════════════════════════════════════════════
    r_min_h = max(r_hor * 1.02, float(np.min(r_all)) * 0.85) if len(r_all) else r_hor*1.1
    r_max_h = float(np.max(r_all)) * 1.15 if len(r_all) else 20.0 * M
    n_r, n_phi = 180, 360
    r_grid   = np.linspace(r_min_h, r_max_h, n_r)
    phi_grid = np.linspace(0, 2 * math.pi, n_phi)
    R_g, P_g = np.meshgrid(r_grid, phi_grid, indexing="ij")   # shape (n_r, n_phi)

    # Compute visibility map on grid (vectorized per row)
    vis_grid = np.zeros((n_r, n_phi), dtype=float)
    for i in range(n_r):
        r_row = R_g[i, :]   # shape (n_phi,)
        phi_row = P_g[i, :]
        # Vectorized angle computation
        delta = (phi_obs - phi_row + math.pi) % (2 * math.pi) - math.pi
        alpha_row = np.abs(delta)
        theta_sh  = _shadow_half_angle(float(r_row[0]), M, a)
        alpha_c   = math.pi - theta_sh
        vis_grid[i, :] = (alpha_row < alpha_c).astype(float)

    # Cartesian for polar plot
    X_g = R_g * np.cos(P_g)
    Y_g = R_g * np.sin(P_g)

    fig2, ax2 = plt.subplots(figsize=(9, 9))
    ax2.set_aspect("equal")

    pcm = ax2.pcolormesh(X_g, Y_g, vis_grid,
                         cmap="RdYlGn", vmin=0, vmax=1,
                         alpha=0.55, shading="auto", zorder=1)
    cbar2 = fig2.colorbar(pcm, ax=ax2, fraction=0.03, pad=0.02)
    cbar2.set_ticks([0, 1])
    cbar2.set_ticklabels(["Sombra (GR)", "Visível"])
    cbar2.set_label("Visibilidade relativística", fontsize=9)

    # Círculos físicos
    for r_circ, clr, lw, ls in [
        (r_hor,  "black", 2.0, "-"),
        (r_phot, "#9b59b6", 1.2, "--"),
        (r_isco, "darkorange", 1.0, ":"),
    ]:
        ax2.add_patch(mpatches.Circle((0,0), r_circ, fill=False, color=clr,
                                       linewidth=lw, linestyle=ls, zorder=6))

    # Trajetória
    ax2.plot(x_all, y_all, color="white", linewidth=0.8, alpha=0.5, zorder=7)
    ax2.scatter(x_all[visible],  y_all[visible],  s=2, color="#2ecc71", alpha=0.7, zorder=8)
    ax2.scatter(x_all[~visible], y_all[~visible], s=2, color="#c0392b", alpha=0.7, zorder=8)

    # Observador
    ax2.annotate("Obs.", xy=(r_max_h * 0.92 * math.cos(phi_obs),
                              r_max_h * 0.92 * math.sin(phi_obs)),
                 fontsize=9, color="#2980b9", ha="center", fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#2980b9", alpha=0.85))

    ax2.set_xlim(-r_max_h*1.05, r_max_h*1.05)
    ax2.set_ylim(-r_max_h*1.05, r_max_h*1.05)
    ax2.set_xlabel("x [M]"); ax2.set_ylabel("y [M]")
    ax2.set_title(
        f"Mapa Polar de Visibilidade — Cone de Escape de Fótons GR\n"
        f"{result.name}  |  {model_lbl}  |  "
        f"{'Sombra assimétrica (Kerr)' if is_kerr else 'Sombra simétrica (Schwarzschild)'}",
        fontsize=10, fontweight="bold",
    )
    ax2.grid(True, linestyle=":", alpha=0.3, zorder=0)

    p2 = os.path.join(outdir, f"{result.name}_visibility_polar.png")
    fig2.savefig(p2, dpi=180, bbox_inches="tight")
    plt.close(fig2)
    paths.append(p2)

    # Relatório no terminal
    n_events = len(shadow_events)
    print(
        f"   Visibilidade GR: visible_frac={vis_frac*100:.1f}%  "
        f"shadow_events={n_events}  "
        f"r_range=[{float(r_all.min()):.2f}, {float(r_all.max()):.2f}] M"
    )
    return paths


# ============================================================
# Análise de Redshift Assintótico (log-log)
# ============================================================

def _ut_theoretical(r: np.ndarray, M: float, E: float,
                    a: float = 0.0, L: float = 0.0) -> np.ndarray:
    """
    dt/dτ teórico em função de r.

    Schwarzschild (a=0):
        u^t = E / A           A = 1 - 2M/r
        → perto do horizonte: u^t ≈ E * 2M / δr   (lei de potência, exp = -1)

    Kerr (a≠0, equatorial θ=π/2):
        u^t = [ (r²+a²+2Ma²/r)·E − 2Ma·L/r ] / Δ    Δ = r²−2Mr+a²
        → perto de r_+: mesma divergência ~1/δr   (exp = -1)
    """
    r2 = r * r
    if abs(a) < 1e-14:
        A = 1.0 - 2.0 * M / r
        A_safe = np.where(np.abs(A) > 1e-300, A, 1e-300)
        return E / A_safe
    else:
        a2 = a * a
        Delta = r2 - 2.0 * M * r + a2
        D_safe = np.where(np.abs(Delta) > 1e-300, Delta, 1e-300)
        return ((r2 + a2 + 2.0 * M * a2 / r) * E - 2.0 * M * a * L / r) / D_safe


def _fit_powerlaw(x: np.ndarray, y: np.ndarray,
                  x_min: float = 0.0, x_max: float = np.inf,
                  min_points: int = 10):
    """
    Ajusta y = A * x^β em escala log-log via regressão linear.

    Retorna (A, beta, r2, x_fit, y_fit) ou None se dados insuficientes.
    """
    mask = (x > x_min) & (x < x_max) & np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    if mask.sum() < min_points:
        return None
    lx = np.log10(x[mask])
    ly = np.log10(y[mask])
    coeffs = np.polyfit(lx, ly, 1)
    beta = coeffs[0]
    log_A = coeffs[1]
    A = 10.0 ** log_A
    # R² em log-log
    ly_pred = np.polyval(coeffs, lx)
    ss_res = np.sum((ly - ly_pred) ** 2)
    ss_tot = np.sum((ly - ly.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    # Curva de ajuste para plot
    x_fit = np.logspace(np.log10(x[mask].min()), np.log10(x[mask].max()), 400)
    y_fit = A * x_fit ** beta
    return A, beta, r2, x_fit, y_fit


def plot_redshift_asymptotic(result: MissionResult, outdir: str,
                              m_cfg: Dict[str, Any]) -> str:
    """
    Análise de redshift gravitacional assintótico perto do horizonte.

    Física esperada — lei de potência:
    ─────────────────────────────────
    Schwarzschild (a=0):
        u^t = dt/dτ = E / (1 − 2M/r) = E·r / δr      δr = r − 2M
        log-log:  log(u^t) = −1·log(δr) + log(E·2M)
        → expoente teórico = −1

    Kerr (a≠0):
        u^t = [(r²+a²+2Ma²/r)·E − 2MaL/r] / Δ        Δ = (r−r₊)(r−r₋)
        perto de r₊:  u^t ~ const / (r − r₊)
        → expoente teórico = −1   (mesma divergência)

    Layout — 3 painéis verticais:
    ┌────────────────────────────────────────────┐
    │ Painel 1: log-log  u^t vs δr              │
    │   • Dados medidos (scatter colorido por τ) │
    │   • Curva teórica exata                    │
    │   • Ajuste de lei de potência (regressão)  │
    │   • Anotação: expoente β e R²              │
    ├────────────────────────────────────────────┤
    │ Painel 2: log-log  latência vs δr          │
    │   (t_coord − τ) — divergência logarítmica  │
    │   Referência: c·ln(1/δr) (Schwarzschild)   │
    ├────────────────────────────────────────────┤
    │ Painel 3: Resíduo (u^t_med/u^t_teo − 1)   │
    │   Erro relativo em % — valida a métrica    │
    │   Banda ±0.1 % e ±1 %                      │
    └────────────────────────────────────────────┘
    """
    params  = m_cfg.get("params", {})
    M       = float(params.get("M",   1.0))
    E_init  = float(params.get("E",   result.E_initial))
    L_init  = float(params.get("L",   result.L_initial))
    a       = float(params.get("a",   0.0))
    is_kerr = abs(a) > 1e-10

    r_hor   = _horizon_radius(params)
    is_lt   = "lowthrust" in result.model.lower() or "_lt" in result.model.lower()

    # ── Concatenar dados de todos os segmentos ─────────────────
    tau_list, r_list, t_list, ut_list = [], [], [], []

    for seg in result.segments:
        tau_seg = np.array(seg.tau,  dtype=float)
        r_seg   = np.array(seg.r,    dtype=float)

        # tcoord
        if hasattr(seg, "tcoord"):
            t_seg = np.array(seg.tcoord, dtype=float)
        elif hasattr(seg, "t"):
            t_seg = np.array(seg.t, dtype=float)
        else:
            t_seg = np.full(len(tau_seg), np.nan)

        # dt/dτ: preferir teórico, senão derivada numérica
        if hasattr(seg, "ut_theory"):
            ut_seg = np.array(seg.ut_theory, dtype=float)
        elif hasattr(seg, "ut_fd"):
            ut_seg = np.array(seg.ut_fd, dtype=float)
        else:
            # Derivada central de tcoord(tau)
            ut_seg = np.gradient(t_seg, tau_seg)

        tau_list.append(tau_seg)
        r_list.append(r_seg)
        t_list.append(t_seg)
        ut_list.append(ut_seg)

    if not tau_list:
        print(f"   [redshift] Sem segmentos para {result.name}.")
        return ""

    tau_all = np.concatenate(tau_list)
    r_all   = np.concatenate(r_list)
    t_all   = np.concatenate(t_list)
    ut_all  = np.concatenate(ut_list)

    delta_r  = r_all - r_hor          # distância ao horizonte
    latency  = t_all - tau_all        # atraso de sinal

    # ── Filtragem: apenas pontos físicos com δr > 0 ────────────
    valid = (delta_r > 1e-14) & np.isfinite(ut_all) & (ut_all > 0) & np.isfinite(latency)
    if valid.sum() < 5:
        print(f"   [redshift] Dados insuficientes perto do horizonte em {result.name}.")
        return ""

    dr_v  = delta_r[valid]
    ut_v  = ut_all[valid]
    lat_v = latency[valid]
    tau_v = tau_all[valid]

    # ── Curva teórica exata ────────────────────────────────────
    dr_range  = np.logspace(
        np.log10(max(dr_v.min(), 1e-6)),
        np.log10(dr_v.max() * 1.2),
        600,
    )
    r_range   = dr_range + r_hor
    ut_theory = _ut_theoretical(r_range, M, E_init, a=a, L=L_init)
    # Latência teórica Schwarzschild: Δt ≈ − E·r_s·ln(δr/r_s) + const
    if not is_kerr:
        lat_ref_scale = float(E_init * 2.0 * M)
        lat_theory    = lat_ref_scale * (-np.log(dr_range / (2.0 * M)))
        lat_theory[lat_theory < 0] = np.nan
    else:
        lat_theory = None

    # ── Ajuste de lei de potência (região próxima ao horizonte) ─
    # Usa apenas δr < 20% de r_hor para a região assintótica
    dr_cut  = r_hor * 0.20
    fit_res = _fit_powerlaw(dr_v, ut_v, x_max=dr_cut, min_points=8)
    if fit_res is None:
        # Tenta com mais pontos (toda a faixa)
        fit_res = _fit_powerlaw(dr_v, ut_v, min_points=5)

    # ── Gradiente de cor: τ para mostrar evolução temporal ──────
    tau_norm = (tau_v - tau_v.min()) / max(tau_v.max() - tau_v.min(), 1e-14)

    # ── Figura ──────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(11, 12),
                             gridspec_kw={"height_ratios": [2.5, 1.5, 1.5]})
    fig.subplots_adjust(hspace=0.38)
    ax_ll, ax_lat, ax_res = axes

    model_lbl = ("Kerr" if is_kerr else "Schwarzschild") + (" LT" if is_lt else "")
    fig.suptitle(
        f"Redshift Gravitacional Assintótico — {result.name}\n"
        f"Modelo: {model_lbl}  |  r_hor = {r_hor:.4f} M  |  E = {E_init:.5f}",
        fontsize=11, fontweight="bold", y=0.98,
    )

    # ══════════════════════════════════════════════════════════
    # Painel 1: log-log   u^t vs δr
    # ══════════════════════════════════════════════════════════
    sc = ax_ll.scatter(
        dr_v, ut_v,
        c=tau_norm, cmap="plasma", s=6, alpha=0.55, zorder=4,
        label="u^t medido",
    )
    cbar = fig.colorbar(sc, ax=ax_ll, pad=0.01, fraction=0.025)
    cbar.set_label("τ normalizado", fontsize=8)

    ax_ll.loglog(r_range, ut_theory,
                 color="#e74c3c", linewidth=2.0, zorder=5,
                 label=f"Teórico: u^t = f(r) [{model_lbl}]")

    # Linha de referência pura  u^t ~ A/δr  (expoente -1)
    A_ref    = float(E_init * r_hor)  # u^t ≈ E·r_hor / δr
    ut_ref   = A_ref / dr_range
    ax_ll.loglog(dr_range, ut_ref,
                 color="#7f8c8d", linewidth=1.2, linestyle=":",
                 zorder=3, label=r"Referência: $E\cdot r_+/\delta r$ (exp = −1)")

    # Ajuste numérico
    fit_txt = "Ajuste: dados insuficientes"
    if fit_res is not None:
        A_fit, beta_fit, r2_fit, x_fit, y_fit = fit_res
        ax_ll.loglog(x_fit, y_fit,
                     color="#27ae60", linewidth=2.2, linestyle="--",
                     zorder=6, label=f"Ajuste: β = {beta_fit:.4f}  (R²={r2_fit:.6f})")
        dev = abs(beta_fit - (-1.0))
        fit_txt = (
            f"Lei de potência  u^t = {A_fit:.4g} · δr^β\n"
            f"  β_ajustado = {beta_fit:+.6f}\n"
            f"  β_teórico  = −1.000000\n"
            f"  |Δβ|       = {dev:.2e}\n"
            f"  R²         = {r2_fit:.8f}"
        )

    ax_ll.set_xscale("log")
    ax_ll.set_yscale("log")
    ax_ll.set_xlabel("δr = r − r_hor  [M]", fontsize=9)
    ax_ll.set_ylabel("u^t = dt/dτ  (fator de redshift)", fontsize=9)
    ax_ll.set_title("Escala log-log: u^t vs distância ao horizonte", fontsize=9)
    ax_ll.legend(fontsize=8, loc="upper right")
    ax_ll.grid(True, which="both", linestyle=":", alpha=0.4)

    # Badge de resultado
    ax_ll.text(
        0.02, 0.97, fit_txt,
        transform=ax_ll.transAxes,
        ha="left", va="top", fontsize=8, family="monospace",
        bbox=dict(boxstyle="round,pad=0.4", fc="#eafaf1", ec="#27ae60", alpha=0.92),
        zorder=7,
    )

    # ══════════════════════════════════════════════════════════
    # Painel 2: latência vs δr (log-log ou semilogy)
    # ══════════════════════════════════════════════════════════
    pos_lat = lat_v > 0
    if pos_lat.sum() > 3:
        ax_lat.scatter(dr_v[pos_lat], lat_v[pos_lat],
                       c=tau_norm[pos_lat], cmap="viridis",
                       s=5, alpha=0.5, zorder=4, label="Latência medida")
        ax_lat.set_xscale("log")
        ax_lat.set_yscale("log")

        if lat_theory is not None:
            # Latitude teórica só para pontos positivos
            valid_th = (lat_theory > 0) & np.isfinite(lat_theory)
            if valid_th.sum() > 2:
                ax_lat.loglog(
                    dr_range[valid_th], lat_theory[valid_th],
                    color="#c0392b", linewidth=1.8, linestyle="-",
                    zorder=5, label=r"Teórico: $−E\,r_s\,\ln(\delta r / r_s)$ (Schw.)",
                )
    else:
        # Se não há latência positiva, plota em escala linear
        ax_lat.scatter(dr_v, lat_v, c=tau_norm, cmap="viridis",
                       s=5, alpha=0.5, zorder=4, label="Latência medida")
        ax_lat.set_xscale("log")

    ax_lat.set_xlabel("δr = r − r_hor  [M]", fontsize=9)
    ax_lat.set_ylabel("t_coord − τ  [M]", fontsize=9)
    ax_lat.set_title(
        r"Latência de sinal: $t_\mathrm{coord} - \tau$ vs δr"
        "\n(divergência logarítmica — não de lei de potência)",
        fontsize=9,
    )
    ax_lat.legend(fontsize=8, loc="upper right")
    ax_lat.grid(True, which="both", linestyle=":", alpha=0.4)

    # ══════════════════════════════════════════════════════════
    # Painel 3: Resíduo relativo (u^t_medido / u^t_teórico − 1)
    # ══════════════════════════════════════════════════════════
    # Interpola curva teórica nos mesmos δr dos dados
    ut_teo_at_data = _ut_theoretical(
        np.array(dr_v + r_hor), M, E_init, a=a, L=L_init
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        residual_pct = 100.0 * (ut_v / ut_teo_at_data - 1.0)

    res_finite = residual_pct[np.isfinite(residual_pct)]
    res_rms    = float(np.sqrt(np.mean(res_finite**2))) if len(res_finite) else float("nan")
    res_max    = float(np.max(np.abs(res_finite)))      if len(res_finite) else float("nan")

    ax_res.scatter(
        dr_v, residual_pct,
        c=tau_norm, cmap="coolwarm", s=5, alpha=0.55, zorder=4,
        label=f"Resíduo: RMS={res_rms:.3e}%  max={res_max:.3e}%",
    )
    ax_res.set_xscale("log")
    ax_res.axhline(0.0,  color="black",   linewidth=1.0, zorder=2)
    ax_res.axhline(+0.1, color="#27ae60", linewidth=0.9, linestyle="--",
                   zorder=2, label="±0.1%")
    ax_res.axhline(-0.1, color="#27ae60", linewidth=0.9, linestyle="--", zorder=2)
    ax_res.axhline(+1.0, color="#f39c12", linewidth=0.9, linestyle=":",
                   zorder=2, label="±1%")
    ax_res.axhline(-1.0, color="#f39c12", linewidth=0.9, linestyle=":", zorder=2)
    ax_res.fill_between(
        [dr_v.min(), dr_v.max()], -0.1, 0.1,
        color="#2ecc71", alpha=0.08, zorder=1,
    )
    ax_res.set_xlabel("δr = r − r_hor  [M]", fontsize=9)
    ax_res.set_ylabel("(u^t_med / u^t_teo − 1) × 100  [%]", fontsize=9)
    ax_res.set_title(
        "Resíduo relativo vs previsão teórica exata\n"
        "(valida a implementação da métrica)",
        fontsize=9,
    )
    ax_res.legend(fontsize=8, loc="upper right")
    ax_res.grid(True, which="both", linestyle=":", alpha=0.3)

    # Limita escala y para não explodir com outliers perto do horizonte numérico
    q95 = float(np.nanpercentile(np.abs(res_finite), 95)) if len(res_finite) else 5.0
    ax_res.set_ylim(-min(q95 * 3, 20.0), min(q95 * 3, 20.0))

    path = os.path.join(outdir, f"{result.name}_redshift_asymptotic.png")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    # ── Relatório no terminal ──────────────────────────────────
    if fit_res is not None:
        A_fit, beta_fit, r2_fit, *_ = fit_res
        dev = abs(beta_fit - (-1.0))
        print(
            f"   Redshift log-log: β={beta_fit:+.6f} (teo=-1)  "
            f"|Δβ|={dev:.2e}  R²={r2_fit:.8f}  "
            f"resíduo_rms={res_rms:.3e}%"
        )
    else:
        print(f"   Redshift log-log: ajuste indisponível (pontos insuficientes).")
    return path


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

            # Tabelas
            print_budget_tables(result)

            # Plots principais
            if result.segments:
                orbit_path = plot_orbit(result, outdir, m_cfg)
                mass_path  = plot_mass(result, outdir)
                inv_path   = plot_invariant(result, outdir, m_cfg)
                plots_msg  = f"\n   Plots: {orbit_path}, {mass_path}"
                if inv_path:
                    plots_msg += f"\n   Invariante:  {inv_path}"
                print(plots_msg)

                # Redshift assintótico (log-log + ajuste de lei de potência)
                rs_path = plot_redshift_asymptotic(result, outdir, m_cfg)
                if rs_path:
                    print(f"   Redshift:    {rs_path}")

                # Mapa de visibilidade relativística (cone de escape)
                vis_paths = plot_visibility_map(result, outdir, m_cfg, phi_obs=0.0)
                for vp in vis_paths:
                    print(f"   Visibilidade: {vp}")
            else:
                print(f"\n   [AVISO] Gráficos ignorados: sem segmentos gerados.")

            # Telemetria
            if result.segments:
                tele_paths = plot_telemetry(result, outdir, m_cfg, observer_dir=(1.0, 0.0))
                if tele_paths:
                    print(f"   Telemetry plots: {os.path.join(outdir, 'telemetry_plots')}")

            if not result.ok:
                all_ok = False

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
    parser.add_argument("--yaml", default=None, help="Caminho para mission.yaml")
    parser.add_argument("--out", default="out/missions", help="Diretório de saída")
    args = parser.parse_args()

    yaml_path = args.yaml or _find_yaml()
    print(f"YAML: {yaml_path}")
    print(f"Out:  {args.out}")

    ok = run_all_missions(yaml_path, outdir=args.out)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()