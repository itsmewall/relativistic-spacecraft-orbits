# src/relorbit_py/mission.py
"""
Planejamento de missões relativísticas com manobras impulsivas e gerenciamento de massa.

Arquitetura multi-segmento:
  O C++ não aplica manobras no loop de integração. Em vez disso, cada segmento
  [tau_i, tau_{i+1}] é simulado de forma limpa, e ao fim de cada segmento a
  manobra é aplicada diretamente no estado Python antes de iniciar o próximo.

Saltos de estado (unidades geométricas G=c=1):
  pr_novo = pr + dv_r          (kick radial no 4-momento pr = dr/dτ)
  L_novo  = L  + dv_phi        (kick tangencial no momento angular específico)

  dv_phi  é a variação de L, NÃO uma velocidade angular diretamente.
  Para estimar o Δv físico a partir de dv_phi, usamos:
    v_phi ≈ L / r  (velocidade tangencial aproximada no ponto de queima)
    Δv_tan ≈ |dv_phi| / r_burn

Equação de Tsiolkovsky (entrada: dv_ms em m/s, campo opcional no YAML):
  m_f = m_i * exp(-|Δv| / v_eff)    com v_eff = isp_s * g0  [m/s]

Se `dv_ms` não for especificado no YAML, é estimado automaticamente da norma
geométrica e convertido para m/s: Δv_ms = sqrt(dv_r² + dv_phi²) * c.
Esse valor representa o pior caso (kick puramente translacional a ~c).
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from .units import G0, C as SPEED_OF_LIGHT


# ============================================================
# Estruturas de resultado
# ============================================================

@dataclass
class ManeuverRecord:
    """Registro de uma manobra executada."""
    index: int
    tau_scheduled: float    # τ da manobra no YAML
    tau_actual: float       # τ real ao fim do segmento
    r_burn: float           # r no ponto de queima
    phi_burn: float         # φ no ponto de queima

    # Estado antes/depois
    pr_before: float
    L_before: float
    pr_after: float
    L_after: float

    dv_r: float
    dv_phi: float
    dv_ms: float            # Δv físico usado no Tsiolkovsky [m/s]
    dv_ms_source: str       # "explicit" | "estimated_geom" | "zero" (sem isp)

    mass_before: float
    mass_after: float
    fuel_consumed: float
    fuel_remaining: float

    ok: bool                # False se combustível insuficiente
    reason: str


@dataclass
class MissionResult:
    """Resultado completo de uma missão multi-segmento."""
    name: str
    model: str

    # Segmentos concatenados (pode ser acessado como se fosse um traj único)
    segments: List[Any]                     # lista de TrajectorySchwarzschildEq / TrajectoryKerrEq
    maneuver_log: List[ManeuverRecord]

    # Configurações originais
    M: float
    E_initial: float
    L_initial: float
    mass0: float
    dry_mass: float
    isp_s: Optional[float]

    # Status global
    ok: bool
    abort_reason: str

    # Arrays concatenados (lazy, gerados em get_trajectory())
    _tau: Optional[np.ndarray] = field(default=None, repr=False)
    _r:   Optional[np.ndarray] = field(default=None, repr=False)
    _phi: Optional[np.ndarray] = field(default=None, repr=False)
    _mass: Optional[np.ndarray] = field(default=None, repr=False)

    def get_trajectory(self):
        """Retorna arrays concatenados de tau, r, phi de todos os segmentos."""
        if self._tau is not None:
            return self._tau, self._r, self._phi, self._mass

        taus, rs, phis, masses = [], [], [], []
        
        # Se a missão falhou e não há segmentos, retorna arrays vazios para evitar o crash
        if not self.segments:
            return np.array([]), np.array([]), np.array([]), np.array([])

        for i, seg in enumerate(self.segments):
            taus.append(np.array(seg.tau, dtype=float))
            rs.append(np.array(seg.r, dtype=float))
            phis.append(np.array(seg.phi, dtype=float))
            
            # Recupera a massa correspondente ao segmento do log de manobras
            # O primeiro segmento usa mass0, os subsequentes usam a massa após a manobra anterior
            m_val = self.mass0 if i == 0 else self.maneuver_log[i-1].mass_after
            masses.append(np.full(len(seg.tau), float(m_val)))

        self._tau  = np.concatenate(taus)
        self._r    = np.concatenate(rs)
        self._phi  = np.concatenate(phis)
        self._mass = np.concatenate(masses)
        return self._tau, self._r, self._phi, self._mass

    def _attach_mass(traj: Any, current_mass: float, n: int) -> None:
        """
        Não tentamos mais modificar o objeto C++.
        A massa agora é inferida por segmento no get_trajectory.
        """
        pass

    def delta_v_budget(self) -> List[Dict[str, Any]]:
        """Tabela de Delta-V budget."""
        rows = []
        total_dv = 0.0
        for m in self.maneuver_log:
            total_dv += m.dv_ms
            rows.append({
                "burn #":        m.index + 1,
                "tau_scheduled": m.tau_scheduled,
                "r_burn [M]":    round(m.r_burn, 4),
                "dv_r [geom]":   m.dv_r,
                "dv_phi [geom]": m.dv_phi,
                "Δv [m/s]":      round(m.dv_ms, 2),
                "Δv_src":        m.dv_ms_source,
                "ok":            m.ok,
            })
        rows.append({
            "burn #": "TOTAL", "tau_scheduled": "-",
            "r_burn [M]": "-", "dv_r [geom]": "-",
            "dv_phi [geom]": "-",
            "Δv [m/s]": round(total_dv, 2),
            "Δv_src": "-", "ok": self.ok,
        })
        return rows

    def mass_budget(self) -> List[Dict[str, Any]]:
        """Tabela de Mass budget."""
        rows = []
        for m in self.maneuver_log:
            rows.append({
                "burn #":           m.index + 1,
                "tau_scheduled":    m.tau_scheduled,
                "mass_before [kg]": round(m.mass_before, 3),
                "mass_after [kg]":  round(m.mass_after, 3),
                "fuel_consumed [kg]": round(m.fuel_consumed, 3),
                "fuel_remaining [kg]": round(m.fuel_remaining, 3),
                "ok":               m.ok,
                "reason":           m.reason,
            })
        tau, r, phi, mass = self.get_trajectory()
        final_mass = float(mass[-1]) if len(mass) and np.isfinite(mass[-1]) else float("nan")
        initial_mass = self.mass0
        rows.append({
            "burn #": "FINAL",
            "tau_scheduled": "-",
            "mass_before [kg]": round(initial_mass, 3),
            "mass_after [kg]":  round(final_mass, 3),
            "fuel_consumed [kg]": round(initial_mass - final_mass, 3),
            "fuel_remaining [kg]": round(final_mass - self.dry_mass, 3),
            "ok": self.ok,
            "reason": "" if self.ok else self.abort_reason,
        })
        return rows

    def print_summary(self) -> None:
        """Imprime resumo da missão no terminal."""
        print(f"\n{'='*60}")
        print(f"MISSÃO: {self.name}  ({self.model})")
        print(f"{'='*60}")
        print(f"  Status: {'OK ✓' if self.ok else 'FALHA ✗  ' + self.abort_reason}")
        print(f"  M={self.M}  E0={self.E_initial:.4f}  L0={self.L_initial:.4f}")
        print(f"  Massa inicial: {self.mass0:.1f} kg  |  Seca: {self.dry_mass:.1f} kg")
        if self.isp_s is not None:
            print(f"  Isp: {self.isp_s:.0f} s  |  v_eff: {self.isp_s*G0/1000:.1f} km/s")

        print(f"\n  Delta-V Budget:")
        _print_table(self.delta_v_budget())

        print(f"\n  Mass Budget:")
        _print_table(self.mass_budget())


# ============================================================
# Spacecraft
# ============================================================

class Spacecraft:
    """
    Gerencia a massa da sonda e aplica a equação de Tsiolkovsky.

    Parâmetros:
        dry_mass_kg   : massa seca (sem combustível) [kg]
        fuel_mass_kg  : massa de combustível inicial [kg]
        isp_s         : impulso específico [s]. None = sem propulsão (geodésica pura).
    """
    def __init__(
        self,
        dry_mass_kg: float,
        fuel_mass_kg: float,
        isp_s: Optional[float] = None,
    ):
        self.dry_mass   = float(dry_mass_kg)
        self.fuel_mass  = float(fuel_mass_kg)
        self.mass       = self.dry_mass + self.fuel_mass
        self.isp_s      = isp_s
        self.veff_ms    = float(isp_s) * G0 if isp_s is not None else None

    @property
    def fuel_remaining(self) -> float:
        return max(0.0, self.mass - self.dry_mass)

    def burn(self, dv_ms: float) -> float:
        """
        Aplica um queima de |dv_ms| metros/segundo.

        Retorna o combustível consumido [kg].
        Lança RuntimeError se o combustível for insuficiente.
        """
        if self.veff_ms is None:
            # Sem propulsão (Isp não definido): queima sem consumo de massa
            return 0.0

        dv = abs(float(dv_ms))
        if dv == 0.0:
            return 0.0

        m_i = self.mass
        m_f = m_i * math.exp(-dv / self.veff_ms)
        consumed = m_i - m_f

        if m_f < self.dry_mass:
            deficit = self.dry_mass - m_f
            raise RuntimeError(
                f"Combustível insuficiente: necessário {consumed:.2f} kg, "
                f"disponível {self.fuel_remaining:.2f} kg "
                f"(deficit {deficit:.2f} kg)"
            )

        self.mass = m_f
        self.fuel_mass = self.mass - self.dry_mass
        return consumed


# ============================================================
# Funções de Δv
# ============================================================

def _dv_ms_from_geom(dv_r: float, dv_phi: float, r_burn: float) -> float:
    """
    Estimativa do Δv físico em m/s a partir de kicks geométricos.

    Componentes:
      v_r_change   ≈ dv_r * c          (dr/dτ é adimensional em G=c=1)
      v_phi_change ≈ (dv_phi / r) * c  (L/r = v_phi em G=c=1)

    Retorna sqrt((dv_r*c)² + (dv_phi*c/r)²) em m/s.
    """
    v_r   = abs(dv_r)
    v_phi = abs(dv_phi) / max(float(r_burn), 1e-300)
    return math.sqrt(v_r**2 + v_phi**2) * SPEED_OF_LIGHT


def _apply_maneuver_to_state(
    r: float, phi: float, pr: float,
    L: float, E: float,
    dv_r: float, dv_phi: float,
) -> tuple[float, float, float]:
    """
    Aplica manobra impulsiva: salto instantâneo em pr e L.
    Retorna (pr_new, L_new, E_new).

    NOTA: Em GR, manobras reais mudam E também (o motor fornece energia).
    Aqui mantemos E constante e mudamos apenas pr e L — válido para
    kicks tangenciais pequenos onde ΔE ≈ 0 é uma boa aproximação.
    Para manobras maiores, o usuário deve especificar `dE` no YAML.
    """
    pr_new = pr + float(dv_r)
    L_new  = L  + float(dv_phi)
    return pr_new, L_new, E


# ============================================================
# Simulação multi-segmento
# ============================================================

def run_mission(
    mission_cfg: Dict[str, Any],
    simulate_fn,                  # simulate_case(case, suite_name) -> traj
) -> MissionResult:
    """
    Executa uma missão completa com manobras impulsivas.

    Algoritmo:
      1. Ordena as manobras por τ.
      2. Para cada segmento [τ_anterior, τ_queima]:
         a. Simula o trecho geodésico.
         b. Extrai estado final (r, φ, pr).
         c. Aplica Tsiolkovsky → consome combustível.
         d. Aplica salto de estado (pr, L) → novo estado inicial.
      3. Simula o segmento final [τ_última_queima, τ_fim].
      4. Concatena e reporta.
    """
    name  = mission_cfg.get("name", "missao")
    model = mission_cfg.get("model", "schwarzschild_equatorial")

    params = mission_cfg.get("params", {}) or {}
    M   = float(params.get("M", 1.0))
    E   = float(params.get("E", 1.0))
    L   = float(params.get("L", 0.0))

    state0 = mission_cfg.get("state0", [10.0, 0.0])
    r0     = float(state0[0])
    phi0   = float(state0[1])
    pr0    = float(mission_cfg.get("pr0", 0.0))

    span     = mission_cfg.get("span", [0.0, 500.0])
    tau0_g   = float(span[0])
    tauf_g   = float(span[1])

    # Massa e propulsão
    mass0       = float(mission_cfg.get("mass0", 1000.0))
    dry_mass    = float(mission_cfg.get("dry_mass", mass0 * 0.3))
    isp_s       = mission_cfg.get("isp_s", None)
    if isp_s is not None:
        isp_s = float(isp_s)
    fuel_mass   = mass0 - dry_mass

    spacecraft = Spacecraft(dry_mass_kg=dry_mass, fuel_mass_kg=fuel_mass, isp_s=isp_s)

    # Solver base
    solver_base = dict(mission_cfg.get("solver", {}) or {})
    solver_base.pop("maneuvers", None)   # manobras gerenciadas em Python, não C++

    # Manobras ordenadas por τ
    raw_maneuvers = list(
        mission_cfg.get("solver", {}).get("maneuvers", [])
        or mission_cfg.get("maneuvers", [])
        or []
    )
    maneuvers = sorted(raw_maneuvers, key=lambda m: float(m.get("tau", 0.0)))

    segments: List[Any]              = []
    maneuver_log: List[ManeuverRecord] = []
    ok          = True
    abort_reason = ""

    # Estado corrente
    tau_now = tau0_g
    r_now   = r0
    phi_now = phi0
    pr_now  = pr0
    L_now   = L
    E_now   = E

    def _make_segment_case(tau_start, tau_end, r, phi, pr, L_val, E_val) -> Dict[str, Any]:
        """Monta um dict de caso para simulate_case."""
        seg_params = dict(params)
        seg_params["L"] = L_val
        seg_params["E"] = E_val
        c = {
            "name":    f"{name}_seg_{tau_start:.3f}_{tau_end:.3f}",
            "model":   model,
            "params":  seg_params,
            "state0":  [r, phi],
            "pr0":     pr,
            "span":    [tau_start, tau_end],
            "solver":  dict(solver_base),
        }
        # Repassa capture_r/capture_eps se houver
        for k in ("capture_r", "capture_eps"):
            if k in params:
                c[k] = params[k]
        return c

    # ---- Loop pelos segmentos ----
    for burn_idx, burn in enumerate(maneuvers):
        tau_burn = float(burn.get("tau", 0.0))
        dv_r     = float(burn.get("dv_r", 0.0))
        dv_phi   = float(burn.get("dv_phi", 0.0))

        # Δv em m/s para Tsiolkovsky
        if "dv_ms" in burn and burn["dv_ms"] is not None:
            dv_ms_val = abs(float(burn["dv_ms"]))
            dv_ms_src = "explicit"
        else:
            dv_ms_val = _dv_ms_from_geom(dv_r, dv_phi, r_now)
            dv_ms_src = "estimated_geom"
        if isp_s is None:
            dv_ms_src = "zero(no_isp)"

        # Garante τ da queima > τ atual
        if tau_burn <= tau_now:
            # queima já passada ou no mesmo instante: aplica direto
            tau_seg_end = tau_now
        else:
            tau_seg_end = tau_burn

        # Simula segmento até a queima
        if tau_seg_end > tau_now:
            seg_case = _make_segment_case(tau_now, tau_seg_end, r_now, phi_now, pr_now, L_now, E_now)
            try:
                traj = simulate_fn(seg_case, model)
                traj = _attach_mass(traj, spacecraft.mass, len(traj.tau))
                segments.append(traj)
                # Estado ao fim do segmento
                r_now   = float(traj.r[-1])
                phi_now = float(traj.phi[-1])
                pr_now  = float(traj.pr[-1])
                tau_now = float(traj.tau[-1])

                # Aborta se capturado antes da queima
                status_str = str(getattr(traj, "status", ""))
                if "CAPTURE" in status_str or "ERROR" in status_str:
                    abort_reason = f"Capturado antes da queima #{burn_idx+1} (τ={tau_burn})"
                    ok = False
                    break
            except Exception as ex:
                abort_reason = f"Erro no segmento antes da queima #{burn_idx+1}: {ex}"
                ok = False
                break

        # Aplica manobra
        mass_before = spacecraft.mass
        consumed    = 0.0
        burn_ok     = True
        burn_reason = ""

        try:
            consumed = spacecraft.burn(dv_ms_val)
        except RuntimeError as ex:
            burn_ok = False
            burn_reason = str(ex)
            ok = False

        mass_after = spacecraft.mass

        pr_new, L_new, E_new = _apply_maneuver_to_state(
            r_now, phi_now, pr_now, L_now, E_now, dv_r, dv_phi
        )

        maneuver_log.append(ManeuverRecord(
            index           = burn_idx,
            tau_scheduled   = tau_burn,
            tau_actual      = tau_now,
            r_burn          = r_now,
            phi_burn        = phi_now,
            pr_before       = pr_now,
            L_before        = L_now,
            pr_after        = pr_new,
            L_after         = L_new,
            dv_r            = dv_r,
            dv_phi          = dv_phi,
            dv_ms           = dv_ms_val,
            dv_ms_source    = dv_ms_src,
            mass_before     = mass_before,
            mass_after      = mass_after,
            fuel_consumed   = consumed,
            fuel_remaining  = spacecraft.fuel_remaining,
            ok              = burn_ok,
            reason          = burn_reason,
        ))

        if not burn_ok:
            abort_reason = f"Queima #{burn_idx+1}: {burn_reason}"
            break

        # Atualiza estado para o próximo segmento
        pr_now  = pr_new
        L_now   = L_new
        E_now   = E_new

    # ---- Segmento final (após última queima) ----
    if ok and tau_now < tauf_g:
        seg_case = _make_segment_case(tau_now, tauf_g, r_now, phi_now, pr_now, L_now, E_now)
        try:
            traj = simulate_fn(seg_case, model)
            _attach_mass(traj, spacecraft.mass, len(traj.tau))
            segments.append(traj)
        except Exception as ex:
            abort_reason = f"Erro no segmento final: {ex}"
            ok = False

    return MissionResult(
        name         = name,
        model        = model,
        segments     = segments,
        maneuver_log = maneuver_log,
        M            = M,
        E_initial    = E,
        L_initial    = L,
        mass0        = mass0,
        dry_mass     = dry_mass,
        isp_s        = isp_s,
        ok           = ok,
        abort_reason = abort_reason,
    )


class _SegmentWrapper:
    """Wrapper leve que encapsula traj C++ e guarda massa localmente."""
    __slots__ = ("_traj", "_mission_mass")

    def __init__(self, traj, mission_mass: float):
        object.__setattr__(self, "_traj", traj)
        object.__setattr__(self, "_mission_mass", [mission_mass] * len(list(traj.tau)))

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_traj"), name)


def _attach_mass(traj, current_mass: float, n: int) -> _SegmentWrapper:
    return _SegmentWrapper(traj, current_mass)


# ============================================================
# Utilitários de impressão
# ============================================================

def _print_table(rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    headers = list(rows[0].keys())
    widths  = {h: max(len(str(h)), max(len(str(r.get(h, ""))) for r in rows)) for h in headers}

    def fmt_row(row: Dict[str, Any]) -> str:
        return "  ".join(str(row.get(h, "")).ljust(widths[h]) for h in headers)

    print("  " + fmt_row({h: h for h in headers}))
    print("  " + "  ".join("-" * widths[h] for h in headers))
    for row in rows:
        print("  " + fmt_row(row))