# src/relorbit_py/plots_visibility_redshift.py
"""Mapa de visibilidade GR e análise de redshift assintótico."""
from __future__ import annotations

import math
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from relorbit_py.mission import MissionResult
from relorbit_py.plots_orbit import _horizon_radius, _isco_radius


# ── Constante ─────────────────────────────────────────────────
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