# src/relorbit_py/null_geodesic_kerr.py
"""
Integrador de geodésicas nulas em Kerr equatorial — núcleo do ray tracer de telemetria.

FÍSICA
======
Geodésicas nulas no espaço-tempo de Kerr Boyer-Lindquist equatorial (θ = π/2).
Duas quantidades conservadas: energia E e momento angular L = b·E (b = parâmetro de impacto).

Equações de movimento (com E=1, L=b):
    r² dr/dλ = σ √R(r)
    r² dφ/dλ = (b − a) + a(r²+a²−ab)/Δ
    r² dt/dλ = (r²+a²)(r²+a²−ab)/Δ + a(a−b)

    R(r) = (r²+a²−ab)² − Δ(b−a)²      [potencial radial]
    Δ(r) = r² − 2Mr + a²

    σ = ±1: sinal radial (+1 = afastando, −1 = aproximando)
    Ponto de retorno: R(r)=0 com σ→−1 (inversão de direcção)
    Captura: r ≤ r₊·(1+ε) com r₊ = M + √(M²−a²)

ARQUITECTURA
============
Três modos de uso (do mais barato ao mais caro):

  1. LUT (Look-Up Table) — recomendado para trajectórias
     Integra N_lut raios uma única vez para um r_s fixo.
     Consultas por interpolação: O(log N_lut) por ponto.

  2. Bisecção — precisão máxima, 40 iterações por parâmetro
     Encontra b* exacto tal que φ(b*) = φ_obs.
     Útil para calcular atraso de tempo e redshift com alta precisão.

  3. Scan 1000 raios — modo diagnóstico / visualização
     Dispara N raios em grelha de b, grava todas as trajectórias.
     Permite visualizar o mapa de deflexão e imagens gravitacionais.

REDSHIFT
========
Factor de redshift gravitacional + Doppler combinado:

    1 + z = (k_μ u^μ)_emissão / (k_μ u^μ)_recepção

Para observador circular em Kerr:
    Ω_circ = √M / (r^{3/2} ± a√M)   (+ prograde, − retrograde)
    u^t = 1 / √(−g_tt − 2g_{tφ}Ω − g_{φφ}Ω²)
    u^φ = Ω u^t
    k_t u^t + k_φ u^φ = −u^t + b·u^φ

Para receptor estático em r_obs >> M:
    k_μ u^μ|_obs ≈ −1/√(1−2M/r_obs)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


# ── Constantes e tipos ────────────────────────────────────────────────────────

@dataclass
class KerrNullConfig:
    """Parâmetros do espaço-tempo e do integrador."""
    M:          float = 1.0         # massa do BH [geometrizado]
    a:          float = 0.0         # spin específico [M]
    r_obs:      float = 1000.0      # raio do receptor [M]
    n_lut:      int   = 1000        # número de raios na LUT
    n_steps:    int   = 12_000      # passos RK4 por raio
    dl_coarse:  float = 0.5         # passo longe do BH [M]
    dl_fine:    float = 0.05        # passo perto do BH [M]
    r_switch:   float = 20.0        # raio de transição fina/grossa [M]
    n_bisect:   int   = 50          # iterações de bissecção
    n_scan:     int   = 1000        # raios no modo scan

    @property
    def r_horizon(self) -> float:
        return self.M + math.sqrt(max(self.M**2 - self.a**2, 0.0))

    @property
    def b_crit_approx(self) -> float:
        """Parâmetro de impacto crítico aproximado (Schwarzschild: 3√3 M)."""
        if abs(self.a) < 1e-10:
            return 3.0 * math.sqrt(3.0) * self.M
        # Kerr: estimativa numérica (b_crit_pro < b_crit_schw < b_crit_retro)
        return 3.0 * math.sqrt(3.0) * self.M * (1.0 - 0.4 * self.a / self.M)


@dataclass
class NullGeodesicResult:
    """Resultado de um raio: parâmetro de impacto, deflexão, atraso, redshift."""
    b:              float           # parâmetro de impacto L/E [M]
    dphi:           float           # deflexão total Δφ [rad]
    t_coord:        float           # tempo coordenado do percurso [M]
    captured:       bool            # caiu no horizonte?
    winding:        int             # número de voltas em torno do BH
    redshift_z:     float = 0.0    # z gravitacional + Doppler
    time_delay:     float = 0.0    # atraso vs luz recta [M]


@dataclass
class TelemetryPoint:
    """Resultado do ray tracer para um ponto da trajectória."""
    tau:        float               # tempo próprio da nave [M]
    r_s:        float               # raio da nave [M]
    phi_s:      float               # ângulo da nave [rad]
    visible:    bool                # algum raio chega ao receptor
    n_images:   int = 0            # número de imagens (1=directo, 2+=lensado)
    b_images:   List[float] = field(default_factory=list)  # b* de cada imagem
    z_images:   List[float] = field(default_factory=list)  # redshift de cada imagem
    t_delays:   List[float] = field(default_factory=list)  # atraso de cada imagem [M]
    dphi_images:List[float] = field(default_factory=list)  # Δφ de cada imagem [rad]


# ── Núcleo: equações do movimento ─────────────────────────────────────────────

def _kerr_null_potential(r: np.ndarray, M: float, a: float, b: np.ndarray
                          ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Potencial radial R(r) e Δ(r) para geodésicas nulas Kerr equatorial.
    R(r) = (r²+a²−ab)² − Δ(b−a)²
    """
    D = np.maximum(r**2 - 2.0*M*r + a**2, 1e-30)
    T = r**2 + a**2 - a*b
    R = T**2 - D*(b - a)**2
    return R, D


def _kerr_rhs_batch(r: np.ndarray, M: float, a: float, b: np.ndarray,
                     sigma: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    RHS vectorizado das equações de movimento para N raios simultâneos.
    Retorna (dr/dλ, dφ/dλ, dt/dλ) × dl.
    """
    R, D = _kerr_null_potential(r, M, a, b)
    T = r**2 + a**2 - a*b
    R_clamp = np.maximum(R, 0.0)
    r2 = r**2

    dr  = sigma * np.sqrt(R_clamp) / r2
    dphi = (b - a + a * T / D) / r2
    dt   = ((r2 + a**2) * T / D + a * (a - b)) / r2
    return dr, dphi, dt


# ── Integrador batch (numpy vectorizado) ──────────────────────────────────────

def integrate_null_batch(
    M:      float,
    a:      float,
    b_arr:  np.ndarray,
    r_s:    float,
    r_obs:  float,
    n_steps: int   = 12_000,
    dl:     float  = 0.5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Integra N raios nulos de r_s até r_obs (ou captura) em paralelo.

    Parâmetros
    ----------
    b_arr   : array (N,) de parâmetros de impacto
    r_s     : raio fonte [M]
    r_obs   : raio receptor [M]
    n_steps : número máximo de passos RK4
    dl      : passo de integração [M] (uniform)

    Retorna
    -------
    phi_arr : deflexão acumulada Δφ [rad] (N,)
    t_arr   : tempo coordenado acumulado [M] (N,)
    cap_arr : capturado? (N, bool)
    wind_arr: número de pontos de retorno (N, int)
    """
    b      = np.asarray(b_arr, dtype=np.float64)
    n      = len(b)
    r      = np.full(n, float(r_s))
    phi    = np.zeros(n)
    t_c    = np.zeros(n)
    sigma  = np.ones(n)
    alive  = np.ones(n, dtype=bool)
    cap    = np.zeros(n, dtype=bool)
    wind   = np.zeros(n, dtype=int)

    r_cap = (M + math.sqrt(max(M**2 - a**2, 0.0))) * 1.005

    for _ in range(n_steps):
        if not np.any(alive):
            break
        idx = np.where(alive)[0]
        r_a   = r[idx]
        sig_a = sigma[idx]
        b_a   = b[idx]

        R, D = _kerr_null_potential(r_a, M, a, b_a)

        # Ponto de retorno: R < 0 e sigma > 0 → inverter direcção
        turn = (R < 0) & (sig_a > 0)
        if np.any(turn):
            sigma[idx[turn]] = -1.0
            wind[idx[turn]]  += 1
            sig_a = sigma[idx]
            R, D = _kerr_null_potential(r_a, M, a, b_a)

        # Captura: R < 0 após inversão (ou ingoing já capturado)
        cap_mask = (R < 0) & (sigma[idx] < 0)
        if np.any(cap_mask):
            cap[idx[cap_mask]]   = True
            alive[idx[cap_mask]] = False
            idx  = idx[~cap_mask]
            if len(idx) == 0:
                continue
            r_a   = r[idx];   sig_a = sigma[idx]; b_a = b[idx]
            R, D  = _kerr_null_potential(r_a, M, a, b_a)

        if len(idx) == 0:
            continue

        T = r_a**2 + a**2 - a*b_a
        R_c = np.maximum(R, 0.0)
        r2  = r_a**2

        # RK4
        k1r = sig_a * np.sqrt(R_c) / r2
        k1p = (b_a - a + a*T/D) / r2
        k1t = ((r2+a**2)*T/D + a*(a-b_a)) / r2

        r2_  = (r_a + 0.5*dl*k1r)**2
        R2, D2 = _kerr_null_potential(r_a+0.5*dl*k1r, M, a, b_a)
        T2 = (r_a+0.5*dl*k1r)**2 + a**2 - a*b_a
        R2c = np.maximum(R2, 0.0)
        k2r = sig_a * np.sqrt(R2c) / r2_
        k2p = (b_a - a + a*T2/D2) / r2_
        k2t = ((r2_+a**2)*T2/D2 + a*(a-b_a)) / r2_

        r3_ = (r_a + 0.5*dl*k2r)**2
        R3, D3 = _kerr_null_potential(r_a+0.5*dl*k2r, M, a, b_a)
        T3 = (r_a+0.5*dl*k2r)**2 + a**2 - a*b_a
        R3c = np.maximum(R3, 0.0)
        k3r = sig_a * np.sqrt(R3c) / r3_
        k3p = (b_a - a + a*T3/D3) / r3_
        k3t = ((r3_+a**2)*T3/D3 + a*(a-b_a)) / r3_

        r4_ = (r_a + dl*k3r)**2
        R4, D4 = _kerr_null_potential(r_a+dl*k3r, M, a, b_a)
        T4 = (r_a+dl*k3r)**2 + a**2 - a*b_a
        R4c = np.maximum(R4, 0.0)
        k4r = sig_a * np.sqrt(R4c) / r4_
        k4p = (b_a - a + a*T4/D4) / r4_
        k4t = ((r4_+a**2)*T4/D4 + a*(a-b_a)) / r4_

        r[idx]   += dl/6*(k1r + 2*k2r + 2*k3r + k4r)
        phi[idx] += dl/6*(k1p + 2*k2p + 2*k3p + k4p)
        t_c[idx] += dl/6*(k1t + 2*k2t + 2*k3t + k4t)

        # Captura por horizonte
        newly_cap = r[idx] <= r_cap
        if np.any(newly_cap):
            cap[idx[newly_cap]]   = True
            alive[idx[newly_cap]] = False

        # Chegada ao receptor (outgoing e r ≥ r_obs)
        still = alive[idx]
        arr_mask = still & (r[idx] >= r_obs) & (sigma[idx] > 0)
        if np.any(arr_mask):
            alive[idx[arr_mask]] = False

    return phi, t_c, cap, wind


# ── Look-Up Table ─────────────────────────────────────────────────────────────

class NullGeodesicLUT:
    """
    Tabela pré-computada de N raios nulos para um raio fonte fixo r_s.

    Estrutura:
      b_arr    : parâmetros de impacto (ordenados)
      phi_arr  : deflexão Δφ correspondente (monotone em b para |b|>b_crit)
      t_arr    : tempo coordenado de voo
      cap_arr  : capturado (bool)
      wind_arr : número de inversões radiais

    Uso:
      lut = NullGeodesicLUT.build(cfg, r_s)
      b_star, t_star = lut.query_phi(phi_target)
    """
    def __init__(self):
        self.b_arr   = np.array([])
        self.phi_arr = np.array([])
        self.t_arr   = np.array([])
        self.cap_arr = np.array([], dtype=bool)
        self.wind_arr= np.array([], dtype=int)
        self.r_s     = 0.0
        self.cfg     = None
        # Sorted view (non-captured, sorted by phi)
        self._b_ok   = np.array([])
        self._phi_ok = np.array([])
        self._t_ok   = np.array([])

    @classmethod
    def build(cls, cfg: KerrNullConfig, r_s: float) -> "NullGeodesicLUT":
        """
        Dispara cfg.n_lut raios nulos e constrói a LUT.
        Tempo típico: ~0.4s para 1000 raios, r_s=10M, Kerr a=0.5.
        """
        M, a = cfg.M, cfg.a
        r_hor = cfg.r_horizon

        # Grelha de b: densa perto de b_crit, esparsa longe
        b_min = r_hor * 1.02
        b_max = max(cfg.r_obs * 5.0, 200.0 * M)
        b_crit = cfg.b_crit_approx

        n_dense = cfg.n_lut * 2 // 3
        n_far   = cfg.n_lut - n_dense

        b_dense = np.linspace(b_min, b_crit * 2.5, n_dense)
        b_far   = np.geomspace(b_crit * 2.5, b_max, n_far + 1)[1:]
        b_arr   = np.unique(np.concatenate([b_dense, b_far]))

        phi, t_c, cap, wind = integrate_null_batch(
            M, a, b_arr, r_s, cfg.r_obs,
            n_steps=cfg.n_steps, dl=cfg.dl_coarse,
        )

        lut = cls()
        lut.b_arr    = b_arr
        lut.phi_arr  = phi
        lut.t_arr    = t_c
        lut.cap_arr  = cap
        lut.wind_arr = wind
        lut.r_s      = r_s
        lut.cfg      = cfg
        lut._rebuild_index()
        return lut

    def _rebuild_index(self):
        ok = ~self.cap_arr
        b  = self.b_arr[ok]
        phi= self.phi_arr[ok]
        t  = self.t_arr[ok]
        order = np.argsort(phi)
        self._phi_ok = phi[order]
        self._b_ok   = b[order]
        self._t_ok   = t[order]

    def query_phi(self, dphi_target: float,
                  winding: int = 0) -> Tuple[Optional[float], Optional[float]]:
        """
        Interpola na LUT para encontrar b* tal que Δφ(b*) ≈ dphi_target.

        Parâmetros
        ----------
        dphi_target : deflexão alvo em rad (0 ≤ Δφ ≤ 2π para directo)
        winding     : imagens extras (winding=1 → Δφ + 2π)

        Retorna
        -------
        (b_star, t_star) ou (None, None) se fora do intervalo
        """
        target = dphi_target + 2.0 * math.pi * winding
        idx = np.searchsorted(self._phi_ok, target)
        if idx == 0 or idx >= len(self._phi_ok):
            return None, None
        alpha = ((target - self._phi_ok[idx-1])
                 / max(self._phi_ok[idx] - self._phi_ok[idx-1], 1e-30))
        b_interp = self._b_ok[idx-1] + alpha * (self._b_ok[idx] - self._b_ok[idx-1])
        t_interp = self._t_ok[idx-1] + alpha * (self._t_ok[idx] - self._t_ok[idx-1])
        return float(b_interp), float(t_interp)

    def query_batch(self, dphi_targets: np.ndarray,
                    winding: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Versão vectorizada de query_phi para N targets simultâneos.

        Retorna
        -------
        b_arr   : (N,) parâmetros de impacto (nan = sem imagem)
        t_arr   : (N,) tempos de voo (nan = sem imagem)
        found   : (N, bool) visibilidade
        """
        targets = np.asarray(dphi_targets) + 2.0 * math.pi * winding
        idx = np.searchsorted(self._phi_ok, targets)

        valid = (idx > 0) & (idx < len(self._phi_ok))
        b_out = np.full(len(targets), np.nan)
        t_out = np.full(len(targets), np.nan)

        if np.any(valid):
            iv = idx[valid]
            alpha = ((targets[valid] - self._phi_ok[iv-1])
                     / np.maximum(self._phi_ok[iv] - self._phi_ok[iv-1], 1e-30))
            b_out[valid] = self._b_ok[iv-1] + alpha*(self._b_ok[iv] - self._b_ok[iv-1])
            t_out[valid] = self._t_ok[iv-1] + alpha*(self._t_ok[iv] - self._t_ok[iv-1])

        return b_out, t_out, valid

    @property
    def phi_range_deg(self) -> Tuple[float, float]:
        if len(self._phi_ok) == 0:
            return (0.0, 0.0)
        return (math.degrees(self._phi_ok[0]), math.degrees(self._phi_ok[-1]))

    @property
    def n_arrived(self) -> int:
        return int(np.sum(~self.cap_arr))

    @property
    def n_captured(self) -> int:
        return int(np.sum(self.cap_arr))


# ── Bisecção de precisão ──────────────────────────────────────────────────────

def bisect_impact_parameter(
    cfg:       KerrNullConfig,
    r_s:       float,
    phi_s:     float,
    phi_obs:   float,
    winding:   int = 0,
    lut:       Optional[NullGeodesicLUT] = None,
) -> Optional[NullGeodesicResult]:
    """
    Encontra b* com alta precisão para que o fotão de (r_s, phi_s)
    chegue ao receptor em (r_obs, phi_obs) após `winding` voltas.

    Usa a LUT como aproximação inicial se fornecida.
    """
    M, a    = cfg.M, cfg.a
    r_obs   = cfg.r_obs
    target  = (phi_obs - phi_s) % (2.0 * math.pi) + 2.0 * math.pi * winding

    # Bracket inicial
    if lut is not None:
        b_guess, _ = lut.query_phi(target, winding=0)
        if b_guess is None:
            return None
        b_lo = b_guess * 0.96
        b_hi = b_guess * 1.04
    else:
        # Scan grosso para encontrar bracket
        r_hor = cfg.r_horizon
        b_scan = np.concatenate([
            np.linspace(r_hor*1.05, 6.0*M, 40),
            np.linspace(6.0*M, 30.0*M, 30),
            np.linspace(30.0*M, 200.0*M, 15),
        ])
        phi_c, t_c, cap_c, _ = integrate_null_batch(
            M, a, b_scan, r_s, r_obs,
            n_steps=cfg.n_steps, dl=cfg.dl_coarse,
        )
        ok   = ~cap_c
        b_ok = b_scan[ok]; phi_ok = phi_c[ok]
        order = np.argsort(phi_ok)
        phi_ok = phi_ok[order]; b_ok = b_ok[order]
        idx = np.searchsorted(phi_ok, target)
        if idx == 0 or idx >= len(phi_ok):
            return None
        b_lo = float(b_ok[idx-1])
        b_hi = float(b_ok[idx])

    # Bissecção
    for _ in range(cfg.n_bisect):
        b_m = 0.5 * (b_lo + b_hi)
        phi_m, t_m, cap_m, wind_m = integrate_null_batch(
            M, a, np.array([b_m]), r_s, r_obs,
            n_steps=cfg.n_steps * 2, dl=cfg.dl_fine,
        )
        if cap_m[0]:
            b_lo = b_m
            continue
        if abs(b_hi - b_lo) < 1e-10:
            break
        # Escolher lado correcto
        phi_lo, _, cap_lo, _ = integrate_null_batch(
            M, a, np.array([b_lo]), r_s, r_obs,
            n_steps=cfg.n_steps, dl=cfg.dl_coarse,
        )
        if (phi_lo[0] - target) * (phi_m[0] - target) < 0:
            b_hi = b_m
        else:
            b_lo = b_m

    b_star = 0.5 * (b_lo + b_hi)
    phi_f, t_f, cap_f, wind_f = integrate_null_batch(
        M, a, np.array([b_star]), r_s, r_obs,
        n_steps=cfg.n_steps * 2, dl=cfg.dl_fine,
    )
    if cap_f[0]:
        return None

    return NullGeodesicResult(
        b=b_star,
        dphi=float(phi_f[0]),
        t_coord=float(t_f[0]),
        captured=False,
        winding=int(wind_f[0]),
    )


# ── Redshift ──────────────────────────────────────────────────────────────────

def compute_redshift(
    M:          float,
    a:          float,
    b:          float,
    r_s:        float,
    r_obs:      float,
    omega_s:    float = 0.0,    # velocidade angular da nave [rad/M]
    prograde:   bool  = True,
) -> float:
    """
    Factor de redshift total: gravitacional + Doppler (primeiro guess).

    Combina:
      - Dilatação gravitacional: g_tt(r_s) vs g_tt(r_obs)
      - Doppler: movimento do emissor com velocidade angular Ω_s

    Retorna 1+z  (z>0 = redshift, z<0 = blueshift).

    Para a nave em órbita circular prograde: omega_s = √M / (r_s^{3/2} + a√M).
    Para o receptor estático em r_obs >> M: omega_obs ≈ 0.
    """
    gtt_s     = -(1.0 - 2.0*M/r_s)
    gtphi_s   = -2.0*M*a/r_s
    gphiphi_s = r_s**2 + a**2 + 2.0*M*a**2/r_s

    # u^t emitter (ZAMO orbital ou fornecido)
    Omega_e = omega_s
    norm2_e = -(gtt_s + 2.0*gtphi_s*Omega_e + gphiphi_s*Omega_e**2)
    if norm2_e <= 0.0:
        return 1.0
    ut_e  = 1.0 / math.sqrt(norm2_e)
    uphi_e = Omega_e * ut_e

    # k_t = -1, k_φ = b  (normalização E=1)
    ku_emit = -ut_e + b * uphi_e

    # Receptor estático em r_obs
    gtt_obs = -(1.0 - 2.0*M/r_obs)
    if -gtt_obs <= 0.0:
        return 1.0
    ku_obs = -1.0 / math.sqrt(-gtt_obs)   # = -1/√(1-2M/r_obs)

    if abs(ku_emit) < 1e-30:
        return 1.0

    # ω_obs / ω_emit = ku_obs / ku_emit   (ambos negativos → ratio positivo)
    freq_ratio = ku_obs / ku_emit
    if freq_ratio <= 0.0:
        return 1.0
    return 1.0 / freq_ratio   # 1 + z


def circular_orbit_omega(M: float, a: float, r: float, prograde: bool = True) -> float:
    """Velocidade angular da órbita circular Kerr."""
    sgn = +1.0 if prograde else -1.0
    denom = r**1.5 + sgn * a * math.sqrt(M)
    if abs(denom) < 1e-30:
        return 0.0
    return math.sqrt(M) / denom


# ── Modo scan: 1000 raios com trajectórias completas ─────────────────────────

def scan_rays(
    cfg:    KerrNullConfig,
    r_s:    float,
    phi_s:  float = 0.0,
    n_rays: Optional[int] = None,
) -> dict:
    """
    Modo "1000 raios": dispara n_rays raios e grava as trajectórias completas.
    Retorna dict com arrays para visualização.

    Usado para:
      - Mapa de deflexão b → Δφ
      - Visualização de imagens gravitacionais
      - Diagnóstico do espaço de parâmetros
    """
    M, a  = cfg.M, cfg.a
    n     = n_rays or cfg.n_scan
    r_hor = cfg.r_horizon
    b_crit= cfg.b_crit_approx

    # Grelha: 2/3 densa perto de b_crit, 1/3 longe
    n1 = 2 * n // 3; n2 = n - n1
    b_arr = np.concatenate([
        np.linspace(r_hor * 1.02, b_crit * 2.5, n1),
        np.geomspace(b_crit * 2.5, cfg.r_obs * 3.0, n2),
    ])

    phi, t_c, cap, wind = integrate_null_batch(
        M, a, b_arr, r_s, cfg.r_obs,
        n_steps=cfg.n_steps, dl=cfg.dl_coarse,
    )

    return {
        "b":        b_arr,
        "dphi":     phi,
        "dphi_deg": np.degrees(phi),
        "t_coord":  t_c,
        "captured": cap,
        "winding":  wind,
        "n_arrived":int(np.sum(~cap)),
        "n_captured": int(np.sum(cap)),
        "r_s":      r_s,
        "phi_s":    phi_s,
        "M":        M,
        "a":        a,
    }