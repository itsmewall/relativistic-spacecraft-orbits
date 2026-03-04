# src/relorbit_py/plots_invariant_telemetry.py
"""Plots de invariante físico e telemetria."""
from __future__ import annotations

import math
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from relorbit_py.mission import MissionResult
from relorbit_py.plots_orbit import _horizon_radius


# ── Bandas de qualidade ───────────────────────────────────────

_QUALITY_BANDS = [
    (1e-10, "#2ecc71", "Machine ε"),
    (1e-7,  "#a8e6a3", "Excellent"),
    (1e-4,  "#fff3a3", "Good"),
    (1e-1,  "#ffc880", "Warning"),
    (1e9,   "#ff9999", "Poor"),
]

_QUALITY_COLOR = {
    "excellent": "#2ecc71",
    "good":      "#f39c12",
    "warning":   "#e67e22",
    "poor":      "#e74c3c",
    "no_data":   "#95a5a6",
}

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