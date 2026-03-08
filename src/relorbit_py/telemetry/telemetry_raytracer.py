# src/relorbit_py/telemetry_raytracer.py
"""
Ray tracer de telemetria relativística — Kerr 6-DOF.

ARQUITECTURA
============
Para cada ponto da trajectória da nave (τ, r_s, φ_s), determina:
  1. Visibilidade: existe geodésica nula de (r_s, φ_s) até o receptor?
  2. Parâmetro de impacto b* (directo e imagens lensadas)
  3. Redshift z combinado (gravidade + Doppler)
  4. Atraso de tempo Δt em relação a propagação recta

ALGORITMO (3 modos)
===================
  A. LUT + Interpolação [padrão — O(N_lut) 1×, depois O(log N) por ponto]
     Pré-computa N_lut = 1000 raios para o r_s médio da trajectória.
     Cada ponto da trajectória é resolvido por busca binária na LUT.
     Custo total: ~0.5s para 1000 raios LUT + ~2ms para 5000 pontos.

  B. Scan-por-ponto [diagnóstico — O(N_rays × N_traj)]
     Dispara N_rays raios por ponto de trajectória.
     Custo: ~0.3s/ponto × N_traj.  Muito lento para trajectórias longas.

  C. Bissecção [máxima precisão — O(N_bisect × N_traj)]
     Bissecção do parâmetro b para cada ponto. ~50 iterações cada.
     Útil para calcular atrasos de tempo e redshifts exactos.

INTEGRAÇÃO COM O PIPELINE EXISTENTE
====================================
  from relorbit_py.telemetry.telemetry_raytracer import (
      TelemetryRayTracer,
      RayTracerConfig,
  )
  
  rt = TelemetryRayTracer.from_kerr_trajectory(traj, receiver_phi=0.0)
  result = rt.run()   # retorna TelemetryResult
  plots  = rt.plot(result, outdir="out/")
"""
from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from relorbit_py.telemetry.null_geodesic_kerr import (
    KerrNullConfig,
    NullGeodesicLUT,
    TelemetryPoint,
    circular_orbit_omega,
    compute_redshift,
    scan_rays,
)


# ── Configuração do ray tracer ────────────────────────────────────────────────

@dataclass
class RayTracerConfig:
    """
    Parâmetros do ray tracer de telemetria.

    Atributos chave
    ---------------
    receiver_r      : raio do receptor (Terra) [M] — padrão 1000M
    receiver_phi    : ângulo do receptor [rad] — padrão 0
    n_lut           : número de raios na LUT (o "1000 raios")
    mode            : 'lut' | 'bisect' | 'scan'
    n_images_max    : máximo de imagens gravitacionais (1=só directo, 2+=lensado)
    redshift_model  : 'full' | 'grav_only' | 'none'
    subsample       : usar só 1 em cada `subsample` pontos da trajectória
    """
    receiver_r:     float   = 1000.0
    receiver_phi:   float   = 0.0
    n_lut:          int     = 1000
    n_steps_lut:    int     = 12_000
    mode:           str     = "lut"         # 'lut' | 'bisect' | 'scan'
    n_images_max:   int     = 2             # directo + 1 imagem lensada
    redshift_model: str     = "full"        # 'full' | 'grav_only' | 'none'
    subsample:      int     = 1             # 1 = todos os pontos
    dl_coarse:      float   = 0.5
    dl_fine:        float   = 0.05
    n_bisect:       int     = 50
    n_scan_per_pt:  int     = 1000          # raios em modo scan


# ── Resultado do ray tracer ───────────────────────────────────────────────────

@dataclass
class TelemetryResult:
    """Resultado completo do ray tracer para toda a trajectória."""
    points:     List[TelemetryPoint] = field(default_factory=list)
    lut:        Optional[NullGeodesicLUT] = None
    scan_data:  Optional[dict] = None  # dados do scan diagnóstico
    meta:       Dict[str, Any] = field(default_factory=dict)

    # Arrays derivados (preenchidos por compute_arrays())
    tau:        np.ndarray = field(default_factory=lambda: np.array([]))
    r_s:        np.ndarray = field(default_factory=lambda: np.array([]))
    phi_s:      np.ndarray = field(default_factory=lambda: np.array([]))
    visible:    np.ndarray = field(default_factory=lambda: np.array([]))
    b_direct:   np.ndarray = field(default_factory=lambda: np.array([]))
    z_direct:   np.ndarray = field(default_factory=lambda: np.array([]))
    t_delay:    np.ndarray = field(default_factory=lambda: np.array([]))
    b_lensed:   np.ndarray = field(default_factory=lambda: np.array([]))
    z_lensed:   np.ndarray = field(default_factory=lambda: np.array([]))
    dphi_direct: np.ndarray = field(default_factory=lambda: np.array([]))

    def compute_arrays(self):
        """Converte lista de TelemetryPoint em arrays numpy."""
        if not self.points:
            return
        n = len(self.points)
        self.tau      = np.array([p.tau     for p in self.points])
        self.r_s      = np.array([p.r_s     for p in self.points])
        self.phi_s    = np.array([p.phi_s   for p in self.points])
        self.visible  = np.array([p.visible  for p in self.points], dtype=bool)

        self.b_direct   = np.full(n, np.nan)
        self.z_direct   = np.full(n, np.nan)
        self.t_delay    = np.full(n, np.nan)
        self.b_lensed   = np.full(n, np.nan)
        self.z_lensed   = np.full(n, np.nan)
        self.dphi_direct= np.full(n, np.nan)

        for i, p in enumerate(self.points):
            if p.visible and p.b_images:
                self.b_direct[i]    = p.b_images[0]
                self.z_direct[i]    = p.z_images[0] if p.z_images else np.nan
                self.t_delay[i]     = p.t_delays[0] if p.t_delays else np.nan
                self.dphi_direct[i] = p.dphi_images[0] if p.dphi_images else np.nan
                if len(p.b_images) > 1:
                    self.b_lensed[i]  = p.b_images[1]
                    self.z_lensed[i]  = p.z_images[1] if len(p.z_images)>1 else np.nan

    @property
    def visibility_fraction(self) -> float:
        if len(self.visible) == 0:
            return 0.0
        return float(np.mean(self.visible))

    @property
    def n_lensed_images(self) -> int:
        return int(np.sum(~np.isnan(self.b_lensed)))

    def summary(self) -> Dict[str, Any]:
        return {
            "n_points":            len(self.points),
            "visibility_fraction": self.visibility_fraction,
            "n_lensed_images":     self.n_lensed_images,
            "b_direct_mean":       float(np.nanmean(self.b_direct)),
            "b_direct_range":      [float(np.nanmin(self.b_direct)),
                                    float(np.nanmax(self.b_direct))],
            "z_direct_mean":       float(np.nanmean(self.z_direct)),
            "z_direct_range":      [float(np.nanmin(self.z_direct)),
                                    float(np.nanmax(self.z_direct))],
            "t_delay_mean":        float(np.nanmean(self.t_delay)),
            "lut_n_arrived":       self.lut.n_arrived if self.lut else 0,
            "lut_n_captured":      self.lut.n_captured if self.lut else 0,
            "lut_phi_range_deg":   list(self.lut.phi_range_deg) if self.lut else [],
            "mode":                self.meta.get("mode", "?"),
            "time_lut_s":          self.meta.get("time_lut_s", 0.0),
            "time_query_s":        self.meta.get("time_query_s", 0.0),
        }


# ── Ray Tracer principal ──────────────────────────────────────────────────────

class TelemetryRayTracer:
    """
    Ray tracer de telemetria relativística em Kerr.

    Uso mínimo:
        rt     = TelemetryRayTracer(M, a, tau, r_arr, phi_arr, cfg)
        result = rt.run()
        plots  = rt.plot(result, outdir="out/plots")
    """
    def __init__(
        self,
        M:      float,
        a:      float,
        tau:    np.ndarray,
        r_arr:  np.ndarray,
        phi_arr:np.ndarray,
        cfg:    RayTracerConfig,
        omega_arr: Optional[np.ndarray] = None,   # Ω(τ) da nave
    ):
        self.M        = M
        self.a        = a
        self.tau      = np.asarray(tau)
        self.r_arr    = np.asarray(r_arr)
        self.phi_arr  = np.asarray(phi_arr)
        self.omega_arr= omega_arr
        self.cfg      = cfg
        self._null_cfg = KerrNullConfig(
            M=M, a=a,
            r_obs=cfg.receiver_r,
            n_lut=cfg.n_lut,
            n_steps=cfg.n_steps_lut,
            dl_coarse=cfg.dl_coarse,
            dl_fine=cfg.dl_fine,
            n_bisect=cfg.n_bisect,
            n_scan=cfg.n_scan_per_pt,
        )

    @classmethod
    def from_kerr_trajectory(
        cls,
        traj,                   # TrajectoryCoupledKerr do motor C++
        cfg:    Optional[RayTracerConfig] = None,
        receiver_phi: float = 0.0,
        receiver_r:   float = 1000.0,
    ) -> "TelemetryRayTracer":
        """Constrói o ray tracer a partir de uma trajectória Kerr 6-DOF."""
        if cfg is None:
            cfg = RayTracerConfig(
                receiver_r=receiver_r,
                receiver_phi=receiver_phi,
            )
        else:
            cfg.receiver_phi = receiver_phi
            cfg.receiver_r   = receiver_r

        tau = np.array(traj.tau)
        r   = np.array(traj.r)
        phi = np.array(traj.phi)

        # Velocidade angular da nave: dφ/dτ (derivada numérica)
        if len(tau) > 1:
            omega = np.gradient(phi, tau)
        else:
            omega = np.zeros_like(tau)

        return cls(
            M=traj.M, a=traj.a,
            tau=tau, r_arr=r, phi_arr=phi,
            cfg=cfg, omega_arr=omega,
        )

    # ── Modo LUT ─────────────────────────────────────────────────────────────

    def _run_lut(self) -> TelemetryResult:
        """
        Modo principal: LUT + interpolação vectorizada.
        Passo 1: constrói LUT (N_lut raios) para r_s representativo.
        Passo 2: para cada ponto da trajectória, interpola b* e calcula z.
        """
        M, a    = self.M, self.a
        cfg     = self.cfg
        ncfg    = self._null_cfg
        phi_obs = cfg.receiver_phi
        r_obs   = cfg.receiver_r

        # Subsample da trajectória
        sl = slice(None, None, cfg.subsample)
        tau  = self.tau[sl]
        r_s  = self.r_arr[sl]
        phi_s= self.phi_arr[sl]
        omega= self.omega_arr[sl] if self.omega_arr is not None else np.zeros_like(tau)

        # ── LUT: r_s representativo (mediana)
        r_s_med = float(np.median(r_s))
        t0 = time.perf_counter()
        lut = NullGeodesicLUT.build(ncfg, r_s_med)
        t_lut = time.perf_counter() - t0
        print(f"[RT] LUT pronta: {lut.n_arrived} raios chegaram, "
              f"{lut.n_captured} capturados  [{t_lut:.2f}s]")
        print(f"[RT] Cobertura angular: "
              f"{lut.phi_range_deg[0]:.1f}° — {lut.phi_range_deg[1]:.1f}°")

        # ── Query: Δφ necessária para cada ponto
        dphi_needed = (phi_obs - phi_s) % (2.0 * math.pi)

        t0 = time.perf_counter()

        # Imagem directa (winding=0)
        b0, t0_arr, found0 = lut.query_batch(dphi_needed, winding=0)

        # Imagem lensada (winding=1, Δφ + 2π)
        b1, t1_arr, found1 = (
            lut.query_batch(dphi_needed, winding=1)
            if cfg.n_images_max >= 2
            else (np.full(len(tau), np.nan),
                  np.full(len(tau), np.nan),
                  np.zeros(len(tau), bool))
        )

        t_query = time.perf_counter() - t0

        # ── Redshift para cada ponto visível
        # Tempo de voo recta (vácuo plano): t_straight ≈ r_obs (distância aprox.)
        t_straight = r_obs  # approximation for large r_obs

        points = []
        for i in range(len(tau)):
            vis = bool(found0[i]) or bool(found1[i])
            pt  = TelemetryPoint(
                tau=float(tau[i]),
                r_s=float(r_s[i]),
                phi_s=float(phi_s[i]),
                visible=vis,
            )
            # Imagem directa
            if found0[i]:
                b_star = float(b0[i])
                t_fly  = float(t0_arr[i])
                z = (compute_redshift(M, a, b_star, float(r_s[i]), r_obs,
                                      omega_s=float(omega[i]))
                     if cfg.redshift_model != "none" else 1.0)
                pt.b_images.append(b_star)
                pt.z_images.append(float(z - 1.0))
                pt.t_delays.append(float(t_fly - t_straight))
                pt.dphi_images.append(float(dphi_needed[i]))
                pt.n_images += 1

            # Imagem lensada
            if found1[i] and cfg.n_images_max >= 2:
                b_star2 = float(b1[i])
                t_fly2  = float(t1_arr[i])
                z2 = (compute_redshift(M, a, b_star2, float(r_s[i]), r_obs,
                                       omega_s=float(omega[i]))
                      if cfg.redshift_model != "none" else 1.0)
                pt.b_images.append(b_star2)
                pt.z_images.append(float(z2 - 1.0))
                pt.t_delays.append(float(t_fly2 - t_straight))
                pt.dphi_images.append(float(dphi_needed[i] + 2*math.pi))
                pt.n_images += 1

            points.append(pt)

        result = TelemetryResult(points=points, lut=lut)
        result.meta.update({
            "mode":       "lut",
            "n_lut":      cfg.n_lut,
            "time_lut_s": t_lut,
            "time_query_s": t_query,
            "r_s_lut":    r_s_med,
        })
        result.compute_arrays()
        return result

    # ── Modo scan por ponto ───────────────────────────────────────────────────

    def _run_scan(self) -> TelemetryResult:
        """
        Modo diagnóstico: N_scan_per_pt raios por ponto de trajectória.
        Muito mais lento que LUT mas mostra a estrutura completa do espaço de raios.
        """
        M, a = self.M, self.a
        cfg  = self.cfg
        ncfg = self._null_cfg
        phi_obs = cfg.receiver_phi
        r_obs   = cfg.receiver_r
        tol_phi = math.radians(1.0)   # janela de chegada ±1°

        sl = slice(None, None, max(cfg.subsample, 5))  # forçar subsample mínimo
        tau  = self.tau[sl]
        r_s  = self.r_arr[sl]
        phi_s= self.phi_arr[sl]
        omega= self.omega_arr[sl] if self.omega_arr is not None else np.zeros_like(tau)

        points = []
        last_scan = None

        t0_all = time.perf_counter()
        for i, (tau_i, r_i, phi_i, om_i) in enumerate(zip(tau, r_s, phi_s, omega)):
            dphi_needed = (phi_obs - phi_i) % (2.0 * math.pi)
            sd = scan_rays(ncfg, r_i, phi_i, n_rays=cfg.n_scan_per_pt)
            if i == 0:
                last_scan = sd  # guardar o primeiro para plots

            ok = ~sd["captured"]
            b_ok   = sd["b"][ok]
            phi_ok = sd["dphi"][ok]
            t_ok   = sd["t_coord"][ok]
            order  = np.argsort(phi_ok)
            phi_ok = phi_ok[order]; b_ok = b_ok[order]; t_ok = t_ok[order]

            pt = TelemetryPoint(
                tau=float(tau_i), r_s=float(r_i), phi_s=float(phi_i),
                visible=False,
            )
            for winding in range(cfg.n_images_max):
                target = dphi_needed + 2.0*math.pi*winding
                idx = np.searchsorted(phi_ok, target)
                if idx == 0 or idx >= len(phi_ok):
                    continue
                if abs(phi_ok[idx-1] - target) < tol_phi:
                    b_hit = float(b_ok[idx-1])
                    t_hit = float(t_ok[idx-1])
                elif abs(phi_ok[idx] - target) < tol_phi:
                    b_hit = float(b_ok[idx])
                    t_hit = float(t_ok[idx])
                else:
                    # interpolação
                    alpha = ((target - phi_ok[idx-1])
                             / max(phi_ok[idx]-phi_ok[idx-1], 1e-30))
                    b_hit = float(b_ok[idx-1] + alpha*(b_ok[idx]-b_ok[idx-1]))
                    t_hit = float(t_ok[idx-1] + alpha*(t_ok[idx]-t_ok[idx-1]))

                z = compute_redshift(M, a, b_hit, float(r_i), r_obs, float(om_i))
                pt.b_images.append(b_hit)
                pt.z_images.append(float(z - 1.0))
                pt.t_delays.append(float(t_hit - r_obs))
                pt.dphi_images.append(float(target))
                pt.n_images += 1
                pt.visible = True

            points.append(pt)
            if (i+1) % 20 == 0:
                elapsed = time.perf_counter() - t0_all
                print(f"[RT scan] {i+1}/{len(tau)}  t={elapsed:.1f}s")

        result = TelemetryResult(points=points, scan_data=last_scan)
        result.meta["mode"] = "scan"
        result.meta["n_scan_per_pt"] = cfg.n_scan_per_pt
        result.compute_arrays()
        return result

    # ── Modo bissecção ────────────────────────────────────────────────────────

    def _run_bisect(self) -> TelemetryResult:
        """
        Modo de alta precisão: bissecção b* para cada ponto.
        Recomendado para calcular atrasos de tempo e redshifts exactos.
        """
        from relorbit_py.telemetry.null_geodesic_kerr import bisect_impact_parameter
        M, a = self.M, self.a
        cfg  = self.cfg
        ncfg = self._null_cfg
        phi_obs = cfg.receiver_phi
        r_obs   = cfg.receiver_r

        sl = slice(None, None, cfg.subsample)
        tau  = self.tau[sl]
        r_s  = self.r_arr[sl]
        phi_s= self.phi_arr[sl]
        omega= self.omega_arr[sl] if self.omega_arr is not None else np.zeros_like(tau)

        # LUT para bracket inicial (acelera bissecção 10×)
        r_med = float(np.median(r_s))
        t0 = time.perf_counter()
        lut = NullGeodesicLUT.build(ncfg, r_med)
        t_lut = time.perf_counter() - t0
        print(f"[RT bisect] LUT pronta [{t_lut:.2f}s]")

        points = []
        t0_all = time.perf_counter()
        for i, (tau_i, r_i, phi_i, om_i) in enumerate(zip(tau, r_s, phi_s, omega)):
            pt = TelemetryPoint(
                tau=float(tau_i), r_s=float(r_i), phi_s=float(phi_i),
                visible=False,
            )
            for winding in range(cfg.n_images_max):
                res = bisect_impact_parameter(ncfg, float(r_i), float(phi_i),
                                              phi_obs, winding=winding, lut=lut)
                if res is None:
                    continue
                z = compute_redshift(M, a, res.b, float(r_i), r_obs, float(om_i))
                pt.b_images.append(res.b)
                pt.z_images.append(float(z - 1.0))
                pt.t_delays.append(float(res.t_coord - r_obs))
                pt.dphi_images.append(res.dphi)
                pt.n_images += 1
                pt.visible = True

            points.append(pt)
            if (i+1) % 50 == 0:
                elapsed = time.perf_counter() - t0_all
                print(f"[RT bisect] {i+1}/{len(tau)}  t={elapsed:.1f}s")

        result = TelemetryResult(points=points, lut=lut)
        result.meta.update({"mode": "bisect", "time_lut_s": t_lut})
        result.compute_arrays()
        return result

    # ── Dispatcher ───────────────────────────────────────────────────────────

    def run(self) -> TelemetryResult:
        """Executa o ray tracer no modo configurado."""
        print(f"[RT] Modo: {self.cfg.mode}  |  "
              f"N_traj={len(self.tau[::self.cfg.subsample])}  |  "
              f"N_lut={self.cfg.n_lut}")
        if self.cfg.mode == "lut":
            return self._run_lut()
        elif self.cfg.mode == "scan":
            return self._run_scan()
        elif self.cfg.mode == "bisect":
            return self._run_bisect()
        else:
            raise ValueError(f"Modo desconhecido: {self.cfg.mode!r}")

    # ── Diagnóstico: scan único ───────────────────────────────────────────────

    def diagnostic_scan(self, r_s: Optional[float] = None,
                         n_rays: int = 1000) -> dict:
        """
        Dispara n_rays raios a partir de r_s para visualizar o mapa de deflexão.
        Retorna dict com dados para plot_scan_map().
        """
        if r_s is None:
            r_s = float(np.median(self.r_arr))
        return scan_rays(self._null_cfg, r_s=r_s, n_rays=n_rays)


# ── Função de conveniência ────────────────────────────────────────────────────

def raytrace_kerr_trajectory(
    traj,
    receiver_phi:   float = 0.0,
    receiver_r:     float = 1000.0,
    n_lut:          int   = 1000,
    subsample:      int   = 1,
    n_images:       int   = 2,
    mode:           str   = "lut",
) -> TelemetryResult:
    """
    API simplificada para ray tracing de uma trajectória Kerr 6-DOF.

    Parâmetros
    ----------
    traj         : TrajectoryCoupledKerr (resultado de simulate_kerr_6dof_rk4)
    receiver_phi : longitude do receptor [rad]
    receiver_r   : distância do receptor [M]
    n_lut        : número de raios na LUT (os "1000 raios" do enunciado)
    subsample    : usar 1 em cada N pontos (economiza memória e tempo)
    n_images     : imagens gravitacionais (1=só directo, 2=directo+lensado)
    mode         : 'lut' (rápido) | 'scan' (diagnóstico) | 'bisect' (preciso)

    Retorna
    -------
    TelemetryResult com arrays de visibilidade, redshift, atraso, b*
    """
    cfg = RayTracerConfig(
        receiver_r=receiver_r,
        receiver_phi=receiver_phi,
        n_lut=n_lut,
        subsample=subsample,
        n_images_max=n_images,
        mode=mode,
    )
    rt = TelemetryRayTracer.from_kerr_trajectory(traj, cfg=cfg,
                                                   receiver_phi=receiver_phi,
                                                   receiver_r=receiver_r)
    return rt.run()
