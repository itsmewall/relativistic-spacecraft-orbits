# src/relorbit_py/validate_perihelion.py
"""
Validação clássica de Schwarzschild:
  1. Precessão do periélio — medição de Δφ/órbita vs teoria GR exata (integral numérica)
  2. Estabilidade ISCO em r = 6M — troca BOUND ↔ CAPTURE cruzando o ISCO

Funções exportadas:
  theoretical_precession_schw(M, E, L, r_start) -> Optional[float]
  validate_schw_perihelion(case, plotdir, time_plotdir) -> dict
  validate_schw_isco(case, plotdir, time_plotdir) -> dict
  _pn_precession(M, L) -> float   (PN fallback, somente para referência educacional)
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .validate_models import validate_schw
from .validate_helpers import status_endswith as _status_endswith


# ============================================================
# 0.  Potencial efetivo e localização dos pontos de retorno
# ============================================================

def _pr2(M: float, E: float, L: float, r: float) -> float:
    """pr² = E² − Veff(r),   Veff = (1−2M/r)(1+L²/r²)."""
    A = 1.0 - 2.0 * M / r
    B = 1.0 + (L * L) / (r * r)
    return E * E - A * B


def _find_turning_points(
    M: float,
    E: float,
    L: float,
    r_start: Optional[float] = None,
) -> Optional[Tuple[float, float]]:
    """
    Localiza os pontos de retorno (r_min, r_max) da órbita, onde pr = 0.

    Algoritmo:
      1. Varre r ∈ [r_hor, r_far] e detecta trocas de sinal de pr²(r).
      2. Refina cada raiz com brentq.
      3. Seleciona o par de raízes consecutivas para o qual pr² > 0 no interior
         E que contenha r_start (se fornecido), priorizando o par que engloba
         a posição inicial da partícula.

    Retorna None se scipy não estiver disponível ou se não encontrar par válido.
    """
    try:
        from scipy import optimize  # type: ignore[import]
    except ImportError:
        return None

    r_hor = 2.001 * M
    r_far = min(max(500.0 * M, 30.0 * L * L / max(M, 1e-300)), 1e9)

    rs = np.linspace(r_hor, r_far, 10_000)
    vals = np.vectorize(lambda r: _pr2(M, E, L, r))(rs)
    idxs = np.where(np.diff(np.sign(vals)))[0]

    roots: List[float] = []
    for idx in idxs:
        try:
            root = optimize.brentq(
                lambda r: _pr2(M, E, L, r),
                float(rs[idx]),
                float(rs[idx + 1]),
                xtol=1e-12,
                rtol=1e-10,
            )
            roots.append(root)
        except Exception:
            pass

    if len(roots) < 2:
        return None

    # Itera pares consecutivos: escolhe o primeiro que tem pr²>0 no interior
    # e (se r_start dado) que contém r_start.
    for i in range(len(roots) - 1):
        ra, rb = roots[i], roots[i + 1]
        if _pr2(M, E, L, 0.5 * (ra + rb)) <= 0.0:
            continue                                    # região não-física
        if r_start is not None and not (ra <= r_start <= rb):
            continue                                    # não contém r_start
        return (ra, rb)

    return None


# ============================================================
# 1.  Teoria exata GR via integração numérica
# ============================================================

def theoretical_precession_schw(
    M: float,
    E: float,
    L: float,
    r_start: Optional[float] = None,
) -> Optional[float]:
    """
    Precessão do periélio GR EXATA (Schwarzschild equatorial) por integração numérica.

    Calcula:
        Δφ_orbit = 2 ∫_{r_min}^{r_max}  (L/r²) / √(E² − Veff(r))  dr

    e retorna Δφ_prec = Δφ_orbit − 2π (precessão líquida por órbita),
    ou None se scipy não estiver disponível ou a órbita não for bound.

    Parâmetros:
        r_start : posição inicial da partícula (usada para selecionar o bracket
                  correto quando há múltiplas regiões onde pr² > 0).
                  Necessário para órbitas excêntricas em campo forte.

    Nota sobre a aproximação PN:
        Para órbitas excêntricas com r_min ≈ 7–10M, a aproximação PN de 1ª ordem
        (Δφ ≈ 6πM²/L²) pode divergir da solução exata em 50–100%.
        Esta função fornece o valor EXATO e deve ser usada como critério.
    """
    try:
        from scipy import integrate  # type: ignore[import]
    except ImportError:
        return None

    if M <= 0 or not all(math.isfinite(v) for v in [M, E, L]):
        return None

    tp = _find_turning_points(M, E, L, r_start)
    if tp is None:
        return None
    r_min, r_max = tp

    def integrand(r: float) -> float:
        p2 = _pr2(M, E, L, r)
        if p2 <= 0.0:
            return 0.0
        return (L / (r * r)) / math.sqrt(p2)

    try:
        result, _abserr = integrate.quad(
            integrand,
            r_min,
            r_max,
            limit=500,
            epsabs=1e-10,
            epsrel=1e-8,
            points=[0.5 * (r_min + r_max)],
        )
    except Exception:
        return None

    return float(2.0 * result - 2.0 * math.pi)


def _pn_precession(M: float, L: float) -> float:
    """
    Aproximação pós-Newtoniana de 1ª ordem:
        Δφ_prec ≈ 6πM/p = 6πM²/L²   (p = L²/M = semi-látus recto)

    ATENÇÃO: válida apenas para M/p ≪ 1 e órbitas de baixa excentricidade.
    Para órbitas excêntricas em campo forte (r_min ≈ 7–12M), o erro pode ser
    50–100%; use somente como referência educacional, NÃO como critério de validação.
    """
    p = (L * L) / max(M, 1e-300)
    return 6.0 * math.pi * M / max(p, 1e-300)


# ============================================================
# 2.  Medição de Δφ a partir dos eventos periapse
# ============================================================

def _measure_precession_from_events(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Extrai os eventos 'periapse' e calcula Δφ_prec por órbita.

    Δφ_prec_k = (φ_{k+1} − φ_k) − 2π,   k = 0, …, n_orb − 1

    Retorno:
        n_orbits           : número de intervalos inter-periapse medidos
        delta_phi_mean     : Δφ_prec médio (rad)   [None se < 2 periapses]
        delta_phi_std      : desvio-padrão de Δφ_prec
        delta_phi_list     : lista de Δφ_prec por órbita
        tau_peri           : tempos próprios de cada periapse
        phi_peri_unwrapped : fases unwrapped de cada periapse
    """
    periapses = sorted(
        [e for e in events if str(e.get("kind", "")) == "periapse"],
        key=lambda e: float(e.get("tau", 0.0)),
    )

    if len(periapses) < 2:
        return {
            "n_orbits": 0,
            "delta_phi_mean": None,
            "delta_phi_std": None,
            "delta_phi_list": [],
            "tau_peri": [float(e["tau"]) for e in periapses],
            "phi_peri_unwrapped": [],
        }

    phi_raw  = np.array([float(e["phi"]) for e in periapses], dtype=float)
    tau_peri = [float(e["tau"]) for e in periapses]

    # NÃO usar np.unwrap: phi é monotone crescente no integrador C++ (dphi/dtau = L/r² > 0),
    # portanto phi acumula livremente (ex: 0, 8.02, 16.04 rad para Δφ_orbit≈8 rad/órbita).
    # np.unwrap interpretaria saltos > π como erros de fase e "corrigiria" para ≈1.74 rad,
    # resultando em Δφ_prec = 1.74 - 2π ≈ -4.54 rad (ERRADO).
    # O valor correto é np.diff(phi_raw) - 2π = 8.02 - 2π ≈ 1.74 rad.
    dphi           = np.diff(phi_raw)           # Δφ_orbit bruto por órbita (phi acumulado)
    delta_phi_prec = dphi - 2.0 * math.pi      # precessão líquida

    return {
        "n_orbits":            int(len(delta_phi_prec)),
        "delta_phi_mean":      float(np.mean(delta_phi_prec)),
        "delta_phi_std":       float(np.std(delta_phi_prec)),
        "delta_phi_list":      [float(v) for v in delta_phi_prec.tolist()],
        "tau_peri":            tau_peri,
        "phi_peri_unwrapped":  [float(v) for v in phi_raw.tolist()],  # valores reais acumulados
    }


# ============================================================
# 3.  validate_schw_perihelion
# ============================================================

def validate_schw_perihelion(
    case: Dict[str, Any],
    plotdir: Optional[str] = None,
    time_plotdir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Valida a precessão do periélio Schwarzschild.

    Além de todos os checks de ``validate_schw``, verifica:

    (a) Pelo menos ``min_orbits_precession`` órbitas completas detectadas.
    (b) Δφ_prec > 0  (precessão prograde, fisicamente obrigatório em Schwarzschild).
    (c) std(Δφ_prec) / |mean(Δφ_prec)| ≤ precession_consistency_max.
    (d) |Δφ_sim − Δφ_exact| / |Δφ_exact| ≤ precession_rel_err_max,
        onde Δφ_exact é a integral numérica exata da equação orbital GR.

    A aproximação PN (Δφ ≈ 6πM²/L²) é exibida no relatório para referência
    educacional, mas NÃO é usada como critério de aprovação.

    Critérios lidos de case["criteria"]:
      precession_rel_err_max      (default 0.05)  — vs integral GR exata
      precession_consistency_max  (default 0.10)  — std / |mean|
      min_orbits_precession       (default 3)
    """
    base = validate_schw(case, plotdir=plotdir, time_plotdir=time_plotdir)

    params  = case.get("params", {}) or {}
    M       = float(params.get("M", case.get("M", 1.0)))
    E       = float(params.get("E", case.get("E")))
    L       = float(params.get("L", case.get("L")))

    state0  = case.get("state0", None)
    r_start = float(state0[0]) if (isinstance(state0, list) and len(state0) >= 1) else None

    crit              = case.get("criteria", {}) or {}
    prec_rel_err_max  = float(crit.get("precession_rel_err_max",     0.05))
    prec_consist_max  = float(crit.get("precession_consistency_max", 0.10))
    min_orbits        = int(crit.get("min_orbits_precession", 3))

    # --- medição a partir dos eventos armazenados por validate_schw ---
    events    = base.get("events", []) or []
    meas      = _measure_precession_from_events(events)

    n_orbits  = meas["n_orbits"]
    dphi_mean = meas["delta_phi_mean"]
    dphi_std  = meas["delta_phi_std"]
    dphi_list = meas["delta_phi_list"]

    # --- teoria exata (scipy) ---
    dphi_exact = theoretical_precession_schw(M, E, L, r_start=r_start)

    # --- PN (somente log) ---
    dphi_pn = _pn_precession(M, L)

    # --- turning points (diagnóstico) ---
    tp           = _find_turning_points(M, E, L, r_start) if r_start is not None else None
    r_min_theory = float(tp[0]) if tp is not None else None
    r_max_theory = float(tp[1]) if tp is not None else None

    # --- checks ---
    prec_ok = True
    prec_reasons: List[str] = []

    # (a) mínimo de órbitas
    if n_orbits < min_orbits:
        prec_ok = False
        prec_reasons.append(f"apenas {n_orbits} órbita(s) detectada(s) (mín={min_orbits})")

    # (b) sinal prograde
    if dphi_mean is not None and dphi_mean <= 0.0:
        prec_ok = False
        prec_reasons.append(
            f"Δφ_prec={dphi_mean:.4f} ≤ 0 (deve ser prograde em Schwarzschild)"
        )

    # (c) consistência
    consistency = None
    if dphi_mean is not None and dphi_std is not None and abs(dphi_mean) > 1e-15:
        consistency = abs(dphi_std) / abs(dphi_mean)
        if consistency > prec_consist_max:
            prec_ok = False
            prec_reasons.append(
                f"std/|mean|={consistency:.3f} > {prec_consist_max:.2f}"
            )

    # (d) comparação com teoria exata
    prec_rel_err = None
    if dphi_mean is not None and dphi_exact is not None and abs(dphi_exact) > 1e-15:
        prec_rel_err = abs(dphi_mean - dphi_exact) / abs(dphi_exact)
        if prec_rel_err > prec_rel_err_max:
            prec_ok = False
            prec_reasons.append(
                f"rel_err vs Δφ_exact={prec_rel_err:.4f} > {prec_rel_err_max:.2f}"
            )
    elif dphi_mean is not None and dphi_exact is None:
        prec_reasons.append("scipy indisponível — comparação com teoria exata ignorada (não conta como falha)")

    # --- resultado ---
    passed = bool(base.get("passed", False)) and prec_ok

    msg = str(base.get("message", "")) or ""
    for reason in prec_reasons:
        msg = (msg + " | " + reason).strip(" |")

    result = dict(base)
    result.update({
        "passed": passed,
        "message": msg,
        # métricas medidas
        "precession_n_orbits":         n_orbits,
        "precession_delta_phi_mean":   dphi_mean,
        "precession_delta_phi_std":    dphi_std,
        "precession_delta_phi_list":   dphi_list,
        "precession_consistency":      consistency,
        # referências
        "precession_delta_phi_exact":  dphi_exact,   # integração GR exata
        "precession_delta_phi_pn":     float(dphi_pn),  # PN (referência educacional)
        "precession_r_min_theory":     r_min_theory,
        "precession_r_max_theory":     r_max_theory,
        # erros
        "precession_rel_err":          prec_rel_err,
        "precession_ok":               prec_ok,
        "precession_reasons":          prec_reasons,
        # critérios
        "criteria_precession": {
            "precession_rel_err_max":     prec_rel_err_max,
            "precession_consistency_max": prec_consist_max,
            "min_orbits_precession":      min_orbits,
        },
    })
    return result


# ============================================================
# 4.  validate_schw_isco
# ============================================================

def validate_schw_isco(
    case: Dict[str, Any],
    plotdir: Optional[str] = None,
    time_plotdir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Valida a troca de estabilidade em r_ISCO = 6M (Schwarzschild, equatorial).

    Regra física central:
      r_start > 6M → órbita estável   → status deve ser BOUND
      r_start < 6M → órbita instável  → status deve ser CAPTURE

    Além do check automático, aceita asserção explícita via
    ``isco_expected_stable`` em case["criteria"]:
      true  → espera BOUND
      false → espera CAPTURE
    """
    base = validate_schw(case, plotdir=plotdir, time_plotdir=time_plotdir)

    params = case.get("params", {}) or {}
    M      = float(params.get("M", case.get("M", 1.0)))
    r_isco = 6.0 * M

    crit = case.get("criteria", {}) or {}
    isco_expected_stable: Optional[bool] = crit.get("isco_expected_stable", None)
    if isco_expected_stable is not None:
        isco_expected_stable = bool(isco_expected_stable)

    state0  = case.get("state0", None)
    r_start = float(state0[0]) if (isinstance(state0, list) and len(state0) >= 1) else float("nan")

    status_str = str(base.get("status", ""))
    is_bound   = _status_endswith(status_str, "BOUND")
    is_capture = _status_endswith(status_str, "CAPTURE")

    theory_stable: Optional[bool] = None
    if math.isfinite(r_start):
        theory_stable = r_start > r_isco

    isco_ok = True
    isco_reasons: List[str] = []

    # check explícito da criteria
    if isco_expected_stable is not None:
        if isco_expected_stable and not is_bound:
            isco_ok = False
            isco_reasons.append(
                f"isco_expected_stable=True mas status={status_str} "
                f"(r_start={r_start:.4f} > r_isco={r_isco:.4f})"
            )
        if not isco_expected_stable and not is_capture:
            isco_ok = False
            isco_reasons.append(
                f"isco_expected_stable=False mas status={status_str} "
                f"(r_start={r_start:.4f} < r_isco={r_isco:.4f})"
            )

    # check físico automático
    if theory_stable is not None:
        if theory_stable and is_capture:
            isco_ok = False
            isco_reasons.append(
                f"r_start={r_start:.4f} > r_isco={r_isco:.4f} (estável) mas simulação CAPTUROU"
            )
        if not theory_stable and is_bound:
            isco_ok = False
            isco_reasons.append(
                f"r_start={r_start:.4f} < r_isco={r_isco:.4f} (instável) mas simulação ficou BOUND"
            )

    passed = bool(base.get("passed", False)) and isco_ok

    msg = str(base.get("message", "")) or ""
    for reason in isco_reasons:
        msg = (msg + " | " + reason).strip(" |")

    result = dict(base)
    result.update({
        "passed":             passed,
        "message":            msg,
        "isco_r_M":           float(r_isco),
        "isco_r_start":       r_start,
        "isco_theory_stable": theory_stable,
        "isco_ok":            isco_ok,
        "isco_reasons":       isco_reasons,
        "criteria_isco": {
            "r_isco":               r_isco,
            "isco_expected_stable": isco_expected_stable,
        },
    })
    return result