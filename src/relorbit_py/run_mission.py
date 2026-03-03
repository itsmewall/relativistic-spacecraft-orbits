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
                mass_path = plot_mass(result, outdir)
                inv_path  = plot_invariant(result, outdir, m_cfg)
                plots_msg = f"\n   Plots: {orbit_path}, {mass_path}"
                if inv_path:
                    plots_msg += f"\n   Invariante: {inv_path}"
                print(plots_msg)
            else:
                print(f"\n   [AVISO] Gráficos ignorados: sem segmentos gerados.")

            # Telemetria (NOVO)
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