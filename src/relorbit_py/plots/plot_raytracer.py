# src/relorbit_py/plot_raytracer.py
"""
Visualizações do ray tracer de telemetria relativística.

PAINÉIS PRODUZIDOS
==================
  1. deflection_map    — mapa b → Δφ (a LUT inteira, linha por linha = "os 1000 raios")
  2. visibility        — fracção de visibilidade ao longo da trajectória
  3. impact_parameter  — b*(τ) directo e lensado ao longo da trajectória
  4. redshift          — z(τ) gravitacional + Doppler
  5. time_delay        — atraso de coordenada Δt(τ) vs propagação recta
  6. sky_map           — posição angular aparente do sinal no céu do receptor
  7. ray_fan           — leque de raios nulos em coordenadas polares (2D)
"""
from __future__ import annotations

import math
import os
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

from relorbit_py.telemetry.telemetry_raytracer import TelemetryResult


# ── Paleta comum ──────────────────────────────────────────────────────────────
_C0, _C1, _C2 = "#1976D2", "#E64A19", "#388E3C"


def _fig(title: str, nrows: int = 1, ncols: int = 1,
         width: float = 11.0, height: float = 4.5):
    fig, ax = plt.subplots(nrows, ncols, figsize=(width, height * nrows))
    fig.suptitle(title, fontsize=11, fontweight="bold", y=1.01)
    return fig, ax


def _save(fig, path: str) -> str:
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


# ── 1. Mapa de deflexão (a LUT inteira = "os 1000 raios") ───────────────────

def plot_deflection_map(result: TelemetryResult, outdir: str,
                        name: str = "telemetry") -> str:
    """
    Mapa completo b → Δφ: todos os raios da LUT.
    Raios capturados → tracejado cinza. Ponto de retorno → círculo.
    Linha vertical = b_crit (esfera de fótons).
    """
    lut = result.lut
    if lut is None:
        return ""

    b    = lut.b_arr
    phi  = np.degrees(lut.phi_arr)
    cap  = lut.cap_arr
    wind = lut.wind_arr

    fig, ax = _fig(f"{name} — Mapa de Deflexão (LUT: {len(b)} raios)")
    ax = np.atleast_1d(ax)[0]

    # Raios que chegaram (sem captura)
    ok = ~cap
    sc = ax.scatter(b[ok], phi[ok], c=wind[ok], cmap="viridis",
                    s=8, alpha=0.8, label="chegaram ao receptor")
    plt.colorbar(sc, ax=ax, label="inversões radiais (winding)")

    # Raios capturados
    if np.any(cap):
        ax.scatter(b[cap], phi[cap], c="gray", s=6, alpha=0.4,
                   marker="x", label="capturados pelo BH")

    # Linhas de referência
    phi_range = phi[ok]
    ax.axhline(360.0,  color=_C2, lw=0.8, ls="--", label="Δφ = 360°")
    ax.axhline(180.0,  color=_C1, lw=0.8, ls=":",  label="Δφ = 180°")
    ax.axhline(0.0,    color="k",  lw=0.5, ls="-")

    ax.set_xlabel("Parâmetro de impacto b [M]")
    ax.set_ylabel("Deflexão total Δφ [°]")
    ax.set_ylim(max(phi_range.min() - 10, -10), min(phi_range.max() + 10, 1500))
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.25)

    path = os.path.join(outdir, f"{name}_deflection_map.png")
    return _save(fig, path)


# ── 2. Visibilidade ao longo da trajectória ───────────────────────────────────

def plot_visibility(result: TelemetryResult, outdir: str,
                    name: str = "telemetry") -> str:
    tau     = result.tau
    vis     = result.visible.astype(float)
    n_img   = np.array([p.n_images for p in result.points])

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    fig.suptitle(f"{name} — Visibilidade da Telemetria", fontsize=11,
                 fontweight="bold")

    # Banda de visibilidade (sombreado)
    ax0.fill_between(tau, 0, vis, alpha=0.3, color=_C0, label="visível")
    ax0.plot(tau, vis, color=_C0, lw=0.8)
    ax0.set_ylabel("Visível (0/1)")
    ax0.set_ylim(-0.05, 1.15)
    frac = result.visibility_fraction
    ax0.set_title(f"Fracção visível: {frac*100:.1f}%  |  "
                  f"Imagens lensadas: {result.n_lensed_images}", fontsize=9)
    ax0.grid(True, alpha=0.25)

    # Número de imagens
    ax1.plot(tau, n_img, color=_C2, lw=1.2, drawstyle="steps-post")
    ax1.set_ylabel("Nº imagens gravitacionais")
    ax1.set_xlabel("Tempo próprio τ [M]")
    ax1.set_ylim(-0.1, max(n_img.max() + 0.5, 2.5))
    ax1.grid(True, alpha=0.25)

    path = os.path.join(outdir, f"{name}_visibility.png")
    return _save(fig, path)


# ── 3. Parâmetro de impacto ───────────────────────────────────────────────────

def plot_impact_parameter(result: TelemetryResult, outdir: str,
                           name: str = "telemetry") -> str:
    tau = result.tau
    b0  = result.b_direct
    b1  = result.b_lensed

    fig, ax = _fig(f"{name} — Parâmetro de Impacto b*(τ)")
    ax = np.atleast_1d(ax)[0]

    vis = result.visible
    ax.plot(tau[vis], b0[vis], ".", color=_C0, ms=3, label="imagem directa b*")
    lensed_ok = ~np.isnan(b1)
    if np.any(lensed_ok):
        ax.plot(tau[lensed_ok], b1[lensed_ok], ".", color=_C1, ms=3,
                label="imagem lensada b*")

    ax.set_xlabel("Tempo próprio τ [M]")
    ax.set_ylabel("Parâmetro de impacto b [M]")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)

    # Anotação b_crit
    M, a = result.meta.get("M", 1.0), result.meta.get("a", 0.0)
    b_crit = 3.0 * math.sqrt(3.0) * M * (1.0 - 0.4 * a / max(M, 1e-30))
    ax.axhline(b_crit, color="gray", ls="--", lw=0.8,
               label=f"b_crit ≈ {b_crit:.2f} M")
    ax.legend(fontsize=8)

    path = os.path.join(outdir, f"{name}_impact_parameter.png")
    return _save(fig, path)


# ── 4. Redshift ───────────────────────────────────────────────────────────────

def plot_redshift(result: TelemetryResult, outdir: str,
                  name: str = "telemetry") -> str:
    tau = result.tau
    z0  = result.z_direct
    z1  = result.z_lensed
    vis = result.visible

    fig, ax = _fig(f"{name} — Redshift Relativístico z(τ)")
    ax = np.atleast_1d(ax)[0]

    ax.plot(tau[vis], z0[vis] * 1e3, ".", color=_C0, ms=3,
            label="imagem directa")
    lensed_ok = ~np.isnan(z1)
    if np.any(lensed_ok & vis):
        ax.plot(tau[lensed_ok & vis], z1[lensed_ok & vis] * 1e3, ".",
                color=_C1, ms=3, label="imagem lensada")

    ax.axhline(0.0, color="k", lw=0.5, ls="--")
    ax.set_xlabel("Tempo próprio τ [M]")
    ax.set_ylabel("Redshift z × 10³")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)

    # Anotação: componentes
    z_mean = float(np.nanmean(z0[vis])) if np.any(vis) else 0.0
    ax.set_title(f"z_médio = {z_mean*1e3:.2f} × 10⁻³  "
                 f"(z>0 = afastamento, z<0 = aproximação)", fontsize=9)

    path = os.path.join(outdir, f"{name}_redshift.png")
    return _save(fig, path)


# ── 5. Atraso de tempo ────────────────────────────────────────────────────────

def plot_time_delay(result: TelemetryResult, outdir: str,
                    name: str = "telemetry") -> str:
    tau   = result.tau
    delay = result.t_delay
    vis   = result.visible

    fig, ax = _fig(f"{name} — Atraso de Coordenada Δt(τ)")
    ax = np.atleast_1d(ax)[0]

    ax.plot(tau[vis], delay[vis], ".", color=_C0, ms=3)
    ax.axhline(0.0, color="k", lw=0.5, ls="--")
    ax.fill_between(tau[vis], delay[vis], 0,
                    where=(delay[vis] > 0), color=_C0, alpha=0.15,
                    label="atraso Shapiro (≥0)")
    ax.fill_between(tau[vis], delay[vis], 0,
                    where=(delay[vis] < 0), color=_C1, alpha=0.15,
                    label="avanço (geometria)")

    ax.set_xlabel("Tempo próprio τ [M]")
    ax.set_ylabel("Δt = t_fly − r_obs  [M]")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    d_mean = float(np.nanmean(delay[vis])) if np.any(vis) else 0.0
    ax.set_title(f"Δt médio = {d_mean:.3f} M  (Shapiro delay + deflexão)", fontsize=9)

    path = os.path.join(outdir, f"{name}_time_delay.png")
    return _save(fig, path)


# ── 6. Leque de raios em coordenadas polares ──────────────────────────────────

def plot_ray_fan(result: TelemetryResult, outdir: str,
                 name: str = "telemetry",
                 n_rays_show: int = 60) -> str:
    """
    Visualização em coordenadas polares de raios seleccionados da LUT.
    Cada raio é integrado step-a-step e desenhado como uma curva.
    """
    lut = result.lut
    if lut is None:
        return ""

    M = result.meta.get("M", 1.0)
    a = result.meta.get("a", 0.0)
    r_s = lut.r_s

    # Seleccionar raios representativos (não capturados)
    ok = ~lut.cap_arr
    b_sel = lut.b_arr[ok]
    # Subsample uniformly
    step = max(1, len(b_sel) // n_rays_show)
    b_show = b_sel[::step][:n_rays_show]

    from relorbit_py.telemetry.null_geodesic_kerr import _kerr_null_potential

    r_hor = M + math.sqrt(max(M**2 - a**2, 0.0))
    r_cap = r_hor * 1.005
    r_obs = lut.cfg.r_obs if lut.cfg else 1000.0

    fig = plt.figure(figsize=(10, 10))
    ax  = fig.add_subplot(111, projection="polar")
    fig.suptitle(f"{name} — Leque de {n_rays_show} Raios Nulos (Kerr)",
                 fontsize=11, fontweight="bold")

    dl = 0.2
    n_steps = 6000

    cmap = plt.cm.plasma
    b_min, b_max = b_show.min(), b_show.max()

    for bi, b in enumerate(b_show):
        r, phi = float(r_s), 0.0
        sigma = 1.0
        rs_path, phis_path = [r], [phi]

        for _ in range(n_steps):
            D = max(r**2 - 2*M*r + a**2, 1e-30)
            T = r**2 + a**2 - a*b
            Rv = T**2 - D*(b - a)**2
            if Rv < 0:
                if sigma > 0:
                    sigma = -1.0
                else:
                    break
            dr   = sigma * math.sqrt(max(Rv, 0.0)) / r**2
            dphi = (b - a + a*T/D) / r**2
            r   += dl * dr
            phi += dl * dphi
            if r <= r_cap:
                rs_path.append(r_cap); phis_path.append(phi)
                break
            if sigma > 0 and r >= min(r_obs, 80.0):
                rs_path.append(r); phis_path.append(phi)
                break
            rs_path.append(r); phis_path.append(phi)

        color = cmap((b - b_min) / max(b_max - b_min, 1e-30))
        ax.plot(phis_path, rs_path, color=color, lw=0.6, alpha=0.6)

    # BH (horizonte)
    theta_c = np.linspace(0, 2*math.pi, 200)
    ax.fill(theta_c, [r_hor]*200, color="black", alpha=0.9, zorder=10)

    # Esfera de fótons
    r_ph = 3.0 * M  # Schwarzschild; Kerr: ligeiramente diferente
    ax.plot(theta_c, [r_ph]*200, color="gold", lw=0.8, ls="--", label="esfera fótons")

    # Posição da nave
    ax.plot([0.0], [r_s], "r*", ms=14, label=f"nave (r={r_s:.0f}M)", zorder=20)

    ax.set_rmax(min(r_obs, 80.0))
    ax.set_title("", pad=15)
    ax.legend(loc="upper right", fontsize=8, bbox_to_anchor=(1.25, 1.1))

    # Barra de cores
    sm = plt.cm.ScalarMappable(cmap=cmap,
                                norm=mcolors.Normalize(vmin=b_min, vmax=b_max))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, shrink=0.5, pad=0.1, label="b [M]")

    path = os.path.join(outdir, f"{name}_ray_fan.png")
    return _save(fig, path)


# ── 7. Mapa do céu (posição aparente) ────────────────────────────────────────

def plot_sky_map(result: TelemetryResult, outdir: str,
                 name: str = "telemetry") -> str:
    """
    Posição aparente do sinal no céu do receptor ao longo da trajectória.
    A deflexão gravitacional desloca a posição aparente em relação à posição real.
    """
    tau  = result.tau
    phi_s= result.phi_s
    dphi = result.dphi_direct
    vis  = result.visible

    fig, ax = _fig(f"{name} — Posição Aparente no Céu do Receptor")
    ax = np.atleast_1d(ax)[0]

    if np.any(vis):
        # Posição real da nave (sem deflexão)
        phi_real = phi_s[vis] % (2*math.pi)
        # Posição aparente (depois da deflexão gravitacional)
        # O receptor em phi_obs=0 vê o sinal vindo de phi_obs - b direction
        # Aproximação: deslocamento angular aparente ≈ dphi - phi_real
        phi_app = (phi_s[vis] - dphi[vis]) % (2*math.pi)

        scatter = ax.scatter(np.degrees(phi_real), np.degrees(phi_app),
                             c=tau[vis], cmap="viridis", s=4, alpha=0.7)
        plt.colorbar(scatter, ax=ax, label="τ [M]")
        ax.plot([0, 360], [0, 360], "k--", lw=0.5, label="sem deflexão")
        ax.set_xlabel("Posição real φ_s [°]")
        ax.set_ylabel("Posição aparente [°]")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.25)

    path = os.path.join(outdir, f"{name}_sky_map.png")
    return _save(fig, path)


# ── Função principal: todos os plots ─────────────────────────────────────────

def plot_raytracer_results(
    result:  TelemetryResult,
    outdir:  str,
    name:    str = "telemetry",
    do_fan:  bool = True,
) -> List[str]:
    """
    Gera todos os plots do ray tracer e retorna a lista de paths.

    Parâmetros
    ----------
    result  : TelemetryResult de TelemetryRayTracer.run()
    outdir  : directório de saída
    name    : prefixo dos ficheiros
    do_fan  : gerar o leque de raios (mais lento)
    """
    os.makedirs(outdir, exist_ok=True)
    paths = []

    def _try(fn, *args, **kwargs) -> str:
        try:
            p = fn(*args, **kwargs)
            if p:
                paths.append(p)
                print(f"   [RT plot] {os.path.basename(p)}")
            return p
        except Exception as e:
            print(f"   [RT plot] AVISO: {fn.__name__} falhou: {e}")
            return ""

    _try(plot_deflection_map,  result, outdir, name)
    _try(plot_visibility,      result, outdir, name)
    _try(plot_impact_parameter, result, outdir, name)
    _try(plot_redshift,        result, outdir, name)
    _try(plot_time_delay,      result, outdir, name)
    _try(plot_sky_map,         result, outdir, name)
    if do_fan:
        _try(plot_ray_fan, result, outdir, name, n_rays_show=80)

    return paths


# ── Impressão do relatório ────────────────────────────────────────────────────

def print_raytracer_report(result: TelemetryResult, name: str = "telemetry"):
    """Imprime o sumário do ray tracer no formato do pipeline."""
    s = result.summary()
    print(f"   [RT] Modo: {s['mode']}  |  Pontos: {s['n_points']}")
    print(f"   [RT] LUT: {s['lut_n_arrived']} raios chegaram, "
          f"{s['lut_n_captured']} capturados")
    print(f"   [RT] Cobertura LUT: {s['lut_phi_range_deg'][0]:.1f}° — "
          f"{s['lut_phi_range_deg'][1]:.1f}°" if s['lut_phi_range_deg'] else "")
    print(f"   [RT] Visibilidade: {s['visibility_fraction']*100:.1f}%  |  "
          f"Imagens lensadas: {s['n_lensed_images']}")
    print(f"   [RT] b*_directo: [{s['b_direct_range'][0]:.3f}, "
          f"{s['b_direct_range'][1]:.3f}] M  (médio: {s['b_direct_mean']:.3f} M)")
    print(f"   [RT] z_directo: [{s['z_direct_range'][0]:+.4f}, "
          f"{s['z_direct_range'][1]:+.4f}]  (médio: {s['z_direct_mean']:+.4f})")
    print(f"   [RT] Δt_médio: {s['t_delay_mean']:.3f} M  (Shapiro + curvatura)")
    print(f"   [RT] Tempo LUT: {s['time_lut_s']:.2f}s  |  "
          f"Tempo query: {s['time_query_s']*1000:.1f}ms")
