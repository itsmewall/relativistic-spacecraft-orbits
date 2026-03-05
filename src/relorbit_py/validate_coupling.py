"""
validate_coupling.py
====================
Valida o critério do Item 8:

    "Mudar atitude muda trajetória (acoplamento real)."

Estratégia
----------
Corremos 4 missões Kerr 6-DOF idênticas excepto na atitude inicial:

  GEO) sem thrust — geodésica de referência
  A)   nozzle aponta +r  (radial outward)  — q = identidade
  B)   nozzle aponta +φ  (tangencial)      — q = rot 90° em z
  C)   nozzle aponta −r  (radial inward)   — q = rot 180° em z

Todas usam o MESMO a_geom_override, MESMA órbita inicial, MESMO dt.
Se o acoplamento for real:
  • r_final(A) ≠ r_final(B) ≠ r_final(C)  (atitude diferente → órbita diferente)
  • ΔL(A) ≈ 0   (força radial não acumula L)
  • ΔL(B) >> 0  (força tangencial acumula L)
  • Δpr(A) grande, Δpr(B) ≈ 0

Asserts executados em ordem:
  1. Asserts de DIRECÇÃO  — diagnosticam DCM/quaternion/projecção
  2. Asserts FÍSICOS      — confirmam acoplamento orbital real

NOTAS DE ESCALA (raiz do bug original)
---------------------------------------
O integrador C++ usa F_geom = F/(m·c²) onde c = 3×10⁸ m/s (SI), mas τ
está em unidades geométricas (c=G=1).  O factor c² = 9×10¹⁶ torna
a_geom ≈ 3×10⁻¹⁹ M⁻¹ para F=30 N — 9×10¹⁶ vezes menor do que o
necessário para ΔL > 0.05.

Fix: a_geom_override = F_newton / mass0_kg = 0.03 M⁻¹,
que bypassa a fórmula SI directamente no EngineCfg.

NOTAS DE AVERAGING
-------------------
tau_span = 300 M = 1.87 órbitas (T ≈ 161 M).  Nozzle fixo no frame
inercial: n_phi = cos(φ(τ)) oscila ±1, e ∫cos dτ cancela ao longo de
órbitas completas.  Fix: TAU_SPAN = T/4 ≈ 40.2 M (< meio período).

Resultado: tabela + assertions + exit 0 se PASS, exit 1 se FAIL.
"""
from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

# ── importar engine ───────────────────────────────────────────────────────────
try:
    from relorbit_py.simulate_kerr_6dof import run_kerr_6dof_mission, ResultKerr6DOF
except ImportError as e:
    print(f"[ERRO] Não foi possível importar simulate_kerr_6dof: {e}", file=sys.stderr)
    print("       Garante que o módulo C++ está compilado e o PYTHONPATH está correcto.",
          file=sys.stderr)
    sys.exit(2)


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURAÇÃO BASE — Item 8 oficial
# ═══════════════════════════════════════════════════════════════════════════════

M  = 1.0    # massa BH [geom]
a  = 0.5    # spin Kerr
E0 = 0.95   # energia específica
L0 = 3.8    # momento angular específico
r0 = 10.0   # raio inicial [M]

# ── FIX AVERAGING: tau_span = T/4 ─────────────────────────────────────────────
# Para E=0.95, L=3.8, M=1, a=0.5, r0=10:
#   dphi/dtau ≈ 0.0391 rad/M  →  T ≈ 160.8 M  →  T/4 ≈ 40.2 M
# T/4 maximiza ∫cos(φ)dτ (= máximo ΔL tangencial).
# tau_span > T/2 faz o integral cancelar → ΔL → 0.
ORBIT_PERIOD_M = 160.8
TAU_SPAN = ORBIT_PERIOD_M / 4.0     # ≈ 40.2 M
DT       = 0.005

# ── FIX ESCALA: a_geom_override = F/m (c=1) ──────────────────────────────────
F_N   = 30.0      # [N] — mantido para dm/dτ via Isp
ISP   = 3000.0
MASS0 = 1000.0
DRY   = 300.0
A_GEOM = F_N / MASS0     # = 0.03 M⁻¹  (correcto para c=1)


def _q_rotz(angle_deg: float):
    """
    Quaternion de rotação em torno de z, em graus.
    Formato (q0=w, q1=x, q2=y, q3=z).  Convenção Eigen: Quaterniond(w, x, y, z).
    Verificação: rotz(90°) → R·[1,0,0] = [0,1,0]  ✓
    """
    h = math.radians(angle_deg) / 2.0
    return (math.cos(h), 0.0, 0.0, math.sin(h))


def _build_cfg(label: str,
               q_init: tuple,
               nozzle: list,
               F_newton: float = F_N,
               a_geom: float = A_GEOM) -> Dict[str, Any]:
    """Constrói dict de missão para um sub-caso do Item 8."""
    q0, q1, q2, q3 = q_init
    # a_geom_override = 0 quando F_newton = 0 (geodésica de referência)
    a_geom_ov = a_geom if F_newton > 0 else 0.0
    return {
        "name": label,
        "model": "kerr_6dof",
        "params": {
            "M": M, "a": a, "E": E0, "L": L0,
            "capture_r": 2.0, "capture_eps": 1e-12,
        },
        "state0": [r0, 0.0],
        "pr0": 0.0,
        "attitude0": {
            "q0": q0, "q1": q1, "q2": q2, "q3": q3,
            "wx": 0.0, "wy": 0.0, "wz": 0.0,
        },
        "inertia": {"Ixx": 100.0, "Iyy": 200.0, "Izz": 150.0},
        "engine": {
            "F_newton":        F_newton,
            "a_geom_override": a_geom_ov,  # ← PATCH escala Item 8
            "isp_s":           ISP,
            "mass0_kg":        MASS0,
            "dry_mass_kg":     DRY,
            "nozzle_body":     nozzle,
            "tau_on":          0.0,
            "tau_off":         TAU_SPAN,    # ← PATCH averaging Item 8
        },
        "ext_torque": {"tx": 0.0, "ty": 0.0, "tz": 0.0},
        "tidal": {
            "enabled": False, "model": "NONE",
            "fd_eps_r": 1e-5, "Q_from_inertia": True,
            "spin_correction": False,
        },
        "span":   [0.0, TAU_SPAN],
        "solver": {"dt": DT, "record_every": 100},
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 4 MISSÕES EXPLÍCITAS DO ITEM 8
# (também referenciadas pelo run_mission.py via kerr_6dof_cases.yaml)
# ═══════════════════════════════════════════════════════════════════════════════

# Geodésica de referência — sem thrust
CFG_GEO   = _build_cfg("geodesic_ref",    _q_rotz(0.0),   [1.0, 0.0, 0.0], F_newton=0.0)

# q=identidade + nozzle +x → n_iner = R·[1,0,0] = [1,0,0] → thrust em +r (em φ=0)
CFG_R_OUT = _build_cfg("radial_outward",  _q_rotz(0.0),   [1.0, 0.0, 0.0])

# q=rotz(90°) + nozzle +x → R·[1,0,0] = [0,1,0] → n_phi = 1 em φ=0 → acumula L
CFG_PHI   = _build_cfg("tangential",      _q_rotz(90.0),  [1.0, 0.0, 0.0])

# q=rotz(180°) + nozzle +x → R·[1,0,0] = [-1,0,0] → n_r = -1 (radial inward)
CFG_R_IN  = _build_cfg("radial_inward",   _q_rotz(180.0), [1.0, 0.0, 0.0])


# ═══════════════════════════════════════════════════════════════════════════════
# MÉTRICAS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Metrics:
    label:           str
    status:          str
    r_final:         float
    r_min:           float
    r_max:           float
    L_final:         float
    delta_L:         float
    pr_final:        float
    delta_pr:        float
    mass_consumed:   float
    eps_rms:         float
    qnorm_err:       float
    thrust_r_mean:   float
    thrust_phi_mean: float

    def row(self) -> str:
        return (
            f"  {self.label:<20} "
            f"r_f={self.r_final:8.4f}M  "
            f"r∈[{self.r_min:.3f},{self.r_max:.3f}]  "
            f"ΔL={self.delta_L:+.4f}  "
            f"Δpr={self.delta_pr:+.5f}  "
            f"Δm={self.mass_consumed:.1f}kg  "
            f"ε={self.eps_rms:.2e}  "
            f"‖q‖err={self.qnorm_err:.2e}  "
            f"[{self.status}]"
        )


def extract(result: ResultKerr6DOF) -> Metrics:
    """Extrai métricas escalares de um ResultKerr6DOF."""
    t = result.traj
    if t is None or not list(t.tau):
        return Metrics(
            label=result.name, status="NO_DATA",
            r_final=float("nan"), r_min=float("nan"), r_max=float("nan"),
            L_final=float("nan"), delta_L=float("nan"),
            pr_final=float("nan"), delta_pr=float("nan"),
            mass_consumed=float("nan"), eps_rms=float("nan"),
            qnorm_err=float("nan"),
            thrust_r_mean=float("nan"), thrust_phi_mean=float("nan"),
        )

    r   = np.asarray(t.r,          dtype=float)
    L   = np.asarray(t.L,          dtype=float)
    pr  = np.asarray(t.pr,         dtype=float)
    eps = np.asarray(t.epsilon,    dtype=float)
    qn  = np.asarray(t.qnorm,      dtype=float)
    mass= np.asarray(t.mass,       dtype=float)
    tr  = np.asarray(t.thrust_r,   dtype=float)
    tph = np.asarray(t.thrust_phi, dtype=float)

    return Metrics(
        label           = result.name,
        status          = str(t.status),
        r_final         = float(r[-1]),
        r_min           = float(np.min(r)),
        r_max           = float(np.max(r)),
        L_final         = float(L[-1]),
        delta_L         = float(L[-1] - L[0]),
        pr_final        = float(pr[-1]),
        delta_pr        = float(pr[-1] - pr[0]),
        mass_consumed   = float(mass[0] - mass[-1]) if len(mass) > 1 else 0.0,
        eps_rms         = float(np.sqrt(np.mean(eps**2))) if len(eps) else float("nan"),
        qnorm_err       = float(np.max(np.abs(qn - 1.0))) if len(qn) else float("nan"),
        thrust_r_mean   = float(np.mean(np.abs(tr[tr   != 0]))) if np.any(tr   != 0) else 0.0,
        thrust_phi_mean = float(np.mean(np.abs(tph[tph != 0]))) if np.any(tph  != 0) else 0.0,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# AUTO-SCALING (só activado quando direcção OK mas escala insuficiente)
# ═══════════════════════════════════════════════════════════════════════════════

def find_min_a_geom(dr_limiar: float = 0.01,
                    dl_limiar: float = 0.05) -> Optional[float]:
    """
    Multiplica a_geom por 10 a cada iteração até os critérios físicos passarem.
    Devolve o a_geom mínimo encontrado, ou None se não convergir em 20 iterações.
    """
    print("\n  [SCALE_SEARCH] Procurando a_geom mínimo para critérios físicos...")
    ag = A_GEOM
    for trial in range(20):
        cfgs = {
            "radial_outward": _build_cfg("ro",  _q_rotz(  0), [1,0,0], F_N, ag),
            "tangential":     _build_cfg("tan", _q_rotz( 90), [1,0,0], F_N, ag),
            "geodesic_ref":   _build_cfg("geo", _q_rotz(  0), [1,0,0], 0.0, 0.0),
        }
        mets: Dict[str, Metrics] = {k: extract(run_kerr_6dof_mission(v))
                                     for k, v in cfgs.items()}
        dr = abs(mets["radial_outward"].r_final - mets["geodesic_ref"].r_final)
        dl = abs(mets["tangential"].delta_L)
        print(f"    trial {trial:2d}: a_geom={ag:.3e}  |ΔL|={dl:.5f}  Δr={dr:.5f}")
        if dl > dl_limiar and dr > dr_limiar:
            print(f"  [SCALE_SEARCH] PASS  →  a_geom_override mínimo = {ag:.4e} M⁻¹")
            print(f"  [SCALE_SEARCH] F equivalente (c=1) = {ag * MASS0:.3e} N")
            return ag
        ag *= 10.0
    print("  [SCALE_SEARCH] Não convergiu em 20 iterações.")
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 72)
    print("  ITEM 8 — Validação do Acoplamento Órbita–Atitude via Thrust")
    print(f"  M={M}  a={a}  E={E0}  L={L0}  r0={r0}M")
    print(f"  τ_span = {TAU_SPAN:.1f} M  (T/4, T ≈ {ORBIT_PERIOD_M:.0f} M)")
    print(f"  a_geom_override = {A_GEOM:.4f} M⁻¹  (F/m, c=1)")
    print("=" * 72)

    cases = [
        ("geodesic_ref",   CFG_GEO),
        ("radial_outward", CFG_R_OUT),
        ("tangential",     CFG_PHI),
        ("radial_inward",  CFG_R_IN),
    ]

    results: Dict[str, Metrics] = {}
    for label, cfg in cases:
        print(f"\n  → Integrando: {label} ...", end="", flush=True)
        res = run_kerr_6dof_mission(cfg)
        m   = extract(res)
        results[label] = m
        print(f" done  status={m.status}")

    # ── Tabela principal ──────────────────────────────────────────────────────
    print()
    print("─" * 72)
    print("  RESULTADOS")
    print("─" * 72)
    for lbl in ["geodesic_ref", "radial_outward", "tangential", "radial_inward"]:
        print(results[lbl].row())
    print("─" * 72)

    # ── Tabela de thrust (diagnóstico de direcção e escala) ───────────────────
    print()
    print("  TABELA DE THRUST (diagnóstico)")
    print()
    print(f"  {'caso':<20}  {'thrust_r_mean':>18}  {'thrust_phi_mean':>18}")
    print(f"  {'─'*20}  {'─'*18}  {'─'*18}")
    for lbl in ["geodesic_ref", "radial_outward", "tangential", "radial_inward"]:
        m = results[lbl]
        print(f"  {lbl:<20}  {m.thrust_r_mean:18.6e}  {m.thrust_phi_mean:18.6e}")
    print()

    geo = results["geodesic_ref"]
    ro  = results["radial_outward"]
    tan = results["tangential"]
    ri  = results["radial_inward"]

    checks: List[tuple] = []   # (desc, passed, detalhe)

    def check(desc: str, cond: bool, detail: str) -> None:
        checks.append((desc, cond, detail))
        sym = "✓" if cond else "✗"
        print(f"  [{sym}] {desc}")
        print(f"        {detail}")
        print()

    # ── ① ASSERTS DE DIRECÇÃO ─────────────────────────────────────────────────
    # Falha aqui → problema de geometria (DCM/quaternion), NÃO de escala.
    LIMIAR_DIR = 1e-6
    print("  ─── ASSERTS DE DIRECÇÃO ──────────────────────────────────────")
    print()

    check(
        "tangential: thrust_phi_mean > 0  [direcção φ activa]",
        tan.thrust_phi_mean > LIMIAR_DIR,
        f"thrust_phi_mean={tan.thrust_phi_mean:.3e}  thrust_r_mean={tan.thrust_r_mean:.3e}\n"
        f"        SE FALHA: n_phi≡0 → DCM errada ou projecção −sinφ/cosφ errada"
    )
    check(
        "radial_outward: thrust_r_mean > 0  [direcção r activa]",
        ro.thrust_r_mean > LIMIAR_DIR,
        f"thrust_r_mean={ro.thrust_r_mean:.3e}  thrust_phi_mean={ro.thrust_phi_mean:.3e}"
    )
    check(
        "radial_inward: thrust_r_mean > 0  [direcção −r activa]",
        ri.thrust_r_mean > LIMIAR_DIR,
        f"thrust_r_mean={ri.thrust_r_mean:.3e}"
    )
    check(
        "geodesic: thrust ≡ 0  [motor desligado]",
        geo.thrust_r_mean < LIMIAR_DIR and geo.thrust_phi_mean < LIMIAR_DIR,
        f"thrust_r={geo.thrust_r_mean:.3e}  thrust_phi={geo.thrust_phi_mean:.3e}"
    )

    direction_ok = (
        tan.thrust_phi_mean > LIMIAR_DIR
        and ro.thrust_r_mean  > LIMIAR_DIR
        and ri.thrust_r_mean  > LIMIAR_DIR
    )

    # ── ② CRITÉRIOS DE ACOPLAMENTO FÍSICO ────────────────────────────────────
    print("  ─── CRITÉRIOS DE ACOPLAMENTO FÍSICO ─────────────────────────")
    print()

    # 1. Desvio orbital vs geodésica
    dr_ro  = abs(ro.r_final  - geo.r_final)
    dr_tan = abs(tan.r_final - geo.r_final)
    dr_ri  = abs(ri.r_final  - geo.r_final)
    check(
        "Thrust produz desvio orbital vs geodésica",
        dr_ro > 0.01 and dr_tan > 0.01 and dr_ri > 0.01,
        f"Δr_final: outward={dr_ro:.4f}M  tangential={dr_tan:.4f}M  "
        f"inward={dr_ri:.4f}M  (limiar = 0.01 M)"
    )

    # 2. Atitudes diferentes → trajetórias diferentes (acoplamento confirmado)
    dr_ro_tan = abs(ro.r_final - tan.r_final)
    dr_ro_ri  = abs(ro.r_final - ri.r_final)
    dr_tan_ri = abs(tan.r_final - ri.r_final)
    check(
        "Atitudes diferentes → r_final diferentes  (acoplamento real)",
        dr_ro_tan > 0.01 and dr_ro_ri > 0.01,
        f"Δr(outward vs tangential)={dr_ro_tan:.4f}M  "
        f"Δr(outward vs inward)={dr_ro_ri:.4f}M  "
        f"Δr(tangential vs inward)={dr_tan_ri:.4f}M"
    )

    # 3. Thrust radial: acumula pr, NÃO acumula L
    check(
        "Thrust radial (+r): |ΔL| << |Δpr|  (força radial não acumula L)",
        abs(ro.delta_L) < 0.5 * abs(ro.delta_pr) and abs(ro.delta_pr) > 0.01,
        f"ΔL={ro.delta_L:+.5f}  Δpr={ro.delta_pr:+.5f}  "
        f"|ΔL|/|Δpr|={abs(ro.delta_L)/max(abs(ro.delta_pr),1e-10):.3f}  (esperado <<1)"
    )

    # 4. Thrust tangencial: acumula L
    check(
        "Thrust tangencial (+φ): |ΔL| significativo  (esperado |ΔL| > 0.05)",
        abs(tan.delta_L) > 0.05,
        f"ΔL={tan.delta_L:+.4f}  (limiar = 0.05)"
    )

    # 5. |ΔL(tangential)| >> |ΔL(radial)|  — diferença qualitativa entre atitudes
    check(
        "Thrust tangencial acumula L >> thrust radial  (>10×)",
        abs(tan.delta_L) > 10.0 * max(abs(ro.delta_L), 1e-10),
        f"|ΔL(tan)|={abs(tan.delta_L):.4f}  |ΔL(radial)|={abs(ro.delta_L):.4f}  "
        f"ratio={abs(tan.delta_L)/max(abs(ro.delta_L),1e-10):.1f}"
    )

    # 6. Outward e inward têm efeitos opostos em r
    check(
        "Thrust radial outward/inward: efeitos opostos em r",
        (ro.r_final - geo.r_final) * (ri.r_final - geo.r_final) < 0,
        f"r_geo={geo.r_final:.4f}  r_out={ro.r_final:.4f}  r_in={ri.r_final:.4f}  "
        f"(outward empurra para fora, inward para dentro)"
    )

    # 7. r_max(tangential) > r_max(geodesic)
    check(
        "r_max(tangential) > r_max(geodesic)  (aceleração tangencial expande órbita)",
        tan.r_max > geo.r_max,
        f"r_max(tan)={tan.r_max:.4f}M  r_max(geo)={geo.r_max:.4f}M"
    )

    # 8. Conservação numérica ε_rms
    for lbl, m in results.items():
        if m.status not in ("BOUND", "UNBOUND", "CAPTURE", "OK"):
            continue
        check(
            f"Conservação numérica ε_rms [{lbl}]",
            math.isfinite(m.eps_rms) and m.eps_rms < 1.0,
            f"ε_rms = {m.eps_rms:.2e}  (limiar < 1.0 em coords geométricas)"
        )

    # 9. Quaternion normalizado
    for lbl, m in results.items():
        if not math.isfinite(m.qnorm_err):
            continue
        check(
            f"Quaternion normalizado [{lbl}]",
            m.qnorm_err < 1e-6,
            f"‖q‖−1 max = {m.qnorm_err:.2e}  (limiar = 1e-6)"
        )

    # 10. Massa consumida igual nos 3 casos com thrust
    dm_ro  = ro.mass_consumed
    dm_tan = tan.mass_consumed
    dm_ri  = ri.mass_consumed
    dm_diff = max(abs(dm_ro - dm_tan), abs(dm_ro - dm_ri), abs(dm_tan - dm_ri))
    check(
        "Massa consumida igual para todos os casos com thrust",
        dm_diff < 1.0,
        f"Δm: outward={dm_ro:.2f}  tangential={dm_tan:.2f}  inward={dm_ri:.2f}  "
        f"dispersão={dm_diff:.3f} kg"
    )

    # ── Resumo ────────────────────────────────────────────────────────────────
    n_pass = sum(1 for _, p, _ in checks if p)
    n_fail = len(checks) - n_pass
    print("─" * 72)
    print(f"  TOTAL: {n_pass}/{len(checks)} critérios PASS")

    # ── Auto-scaling se direcção OK mas físico falha ──────────────────────────
    physics_fail = (
        abs(tan.delta_L) < 0.05
        or abs(ro.r_final - geo.r_final) < 0.01
    )
    if direction_ok and physics_fail:
        print()
        print("  NOTA: direcções OK mas escala insuficiente.")
        print(f"  a_geom_override actual = {A_GEOM:.3e} M⁻¹")
        print("  Iniciando auto-scaling para encontrar a_geom mínimo...")
        find_min_a_geom()

    # ── Interpretação física ──────────────────────────────────────────────────
    print()
    print("  FLUXO DE CÁLCULO (kerr_6dof.cpp)")
    print()
    print("    k_body = nozzle_body.normalized()           // body frame")
    print("    R      = dcm_from_quaternion(q)              // body → ZAMO")
    print("    n_iner = R * k_body                          // direcção ZAMO 3D")
    print("    n_r    = +cosφ·n_x + sinφ·n_y               // projecção radial")
    print("    n_φ    = −sinφ·n_x + cosφ·n_y               // projecção tangencial")
    print("    a_geom = a_geom_override  (se > 0)")
    print("           = F/(m·c²_SI)      (caso contrário — 9×10¹⁶ menor, ERRADO)")
    print("    dpr/dτ += a_geom * n_r * √Δ/r")
    print("    dL/dτ  += √ρ²   * a_geom * n_φ")
    print()
    print(f"  Com a_geom_override = {A_GEOM:.4f} M⁻¹ e τ_span = {TAU_SPAN:.1f} M:")
    print(f"    ΔL(tangential) ≈ r · a_geom · ∫cos(φ)dτ ≈ {r0*A_GEOM*25.6:.2f}")
    print(f"    (∫cos dτ ≈ 25.6 M para T/4 com T = {ORBIT_PERIOD_M:.0f} M)")
    print()

    if n_fail == 0:
        print("  ✓  ITEM 8 VALIDADO — acoplamento real confirmado.")
        print("=" * 72)
        sys.exit(0)
    else:
        print(f"  ✗  {n_fail} critério(s) falharam — ver detalhes acima.")
        print("=" * 72)
        sys.exit(1)


if __name__ == "__main__":
    main()