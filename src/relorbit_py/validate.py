# src/relorbit_py/validate.py
from __future__ import annotations

import argparse
import copy
import json
import math
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from . import engine_hello
from .simulate import load_cases_yaml, simulate_case
from .plots import (
    rel_drift as _rel_drift,
    plot_newton as _plot_newton,
    plot_schw as _plot_schw,
    plot_convergence_newton as _plot_convergence_newton,
    plot_convergence_overlay_newton as _plot_convergence_overlay_newton,
)

# ============================================================
# Helpers gerais
# ============================================================

def _unwrap_traj(res: Any) -> Any:
    if res is None:
        raise TypeError("simulate_case retornou None (bug).")

    # Newton
    if hasattr(res, "t") and hasattr(res, "y"):
        return res

    # Schwarzschild
    if hasattr(res, "tau") and hasattr(res, "r") and hasattr(res, "phi"):
        return res

    # Fallbacks
    if isinstance(res, dict) and "traj" in res:
        return res["traj"]

    if hasattr(res, "data") and isinstance(getattr(res, "data"), dict) and "traj" in res.data:
        return res.data["traj"]

    raise TypeError(f"Formato inesperado do retorno de simulate_case: {type(res)}")


def _get_solver_dt_nsteps(case: Dict[str, Any]) -> Tuple[float, int]:
    if isinstance(case.get("solver"), dict):
        s = case["solver"]
        if "dt" in s:
            dt = float(s["dt"])
            n_steps = int(s.get("n_steps", 0) or 0)
            return dt, n_steps

    if "dt" in case:
        dt = float(case["dt"])
        n_steps = int(case.get("n_steps", 0) or 0)
        return dt, n_steps

    raise KeyError(
        f"dt ausente no caso '{case.get('name','<sem-nome>')}'. "
        "Esperado em case.solver.dt (novo) ou case.dt (legado)."
    )


def _set_solver_dt_nsteps(case: Dict[str, Any], dt: float, n_steps: int) -> None:
    """Atualiza (in-place) dt e n_steps, suportando formato novo (case.solver) e legado."""
    if isinstance(case.get("solver"), dict):
        case["solver"]["dt"] = float(dt)
        case["solver"]["n_steps"] = int(n_steps)
        return
    # legado
    case["dt"] = float(dt)
    case["n_steps"] = int(n_steps)


def _get_span(case: Dict[str, Any]) -> Tuple[float, float]:
    if "span" in case:
        a, b = case["span"]
        return float(a), float(b)
    if "t0" in case and "tf" in case:
        return float(case["t0"]), float(case["tf"])
    if "tau0" in case and "tauf" in case:
        return float(case["tau0"]), float(case["tauf"])
    raise KeyError(f"span ausente no caso '{case.get('name','<sem-nome>')}'. Esperado case.span=[a,b].")


def _case_pr0(case: Dict[str, Any]) -> float:
    if "pr0" in case:
        return float(case["pr0"])
    params = case.get("params", {}) or {}
    if "pr0" in params:
        return float(params["pr0"])
    return 0.0


def _fmt_e(x: Any, width: int = 12, prec: int = 3) -> str:
    if x is None:
        return " " * (width - 4) + "None"
    try:
        return f"{float(x):>{width}.{prec}e}"
    except Exception:
        s = str(x)
        if len(s) > width:
            return (s[: width - 3] + "...").rjust(width)
        return s.rjust(width)


def _fmt_f(x: Any, width: int = 10, prec: int = 6) -> str:
    if x is None:
        return " " * (width - 4) + "None"
    try:
        return f"{float(x):>{width}.{prec}f}"
    except Exception:
        s = str(x)
        if len(s) > width:
            return (s[: width - 3] + "...").rjust(width)
        return s.rjust(width)


def _short_msg(msg: str, maxlen: int = 76) -> str:
    if not msg:
        return ""
    msg = msg.replace("\n", " ").strip()
    if len(msg) <= maxlen:
        return msg
    return msg[: maxlen - 3] + "..."


def _status_endswith(status_str: str, expected_tail: str) -> bool:
    return str(status_str).endswith(str(expected_tail))


def _estimate_order_from_three(e_dt: float, e_dt2: float) -> Optional[float]:
    if not (np.isfinite(e_dt) and np.isfinite(e_dt2)):
        return None
    if e_dt <= 0.0 or e_dt2 <= 0.0:
        return None
    return float(math.log(e_dt / e_dt2, 2.0))


def _is_finite_array(x: np.ndarray) -> bool:
    return bool(np.all(np.isfinite(np.asarray(x, dtype=float))))


def _is_monotone_increasing(x: np.ndarray, tol: float = 0.0) -> bool:
    x = np.asarray(x, dtype=float).reshape(-1)
    if x.size < 2:
        return True
    dx = np.diff(x)
    return bool(np.all(dx >= -float(tol)))


def _finite_diff_first_derivative(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    dy/dx por diferenças finitas (centrada no miolo, one-sided nas bordas).
    """
    y = np.asarray(y, dtype=float).reshape(-1)
    x = np.asarray(x, dtype=float).reshape(-1)
    n = y.size
    if n == 0:
        return np.array([], dtype=float)
    if n == 1:
        return np.array([0.0], dtype=float)
    dy = np.zeros(n, dtype=float)

    # bordas
    dx0 = x[1] - x[0]
    if abs(dx0) < 1e-300:
        dx0 = 1e-300
    dy[0] = (y[1] - y[0]) / dx0

    dxn = x[-1] - x[-2]
    if abs(dxn) < 1e-300:
        dxn = 1e-300
    dy[-1] = (y[-1] - y[-2]) / dxn

    # centro
    for i in range(1, n - 1):
        dxi = x[i + 1] - x[i - 1]
        if abs(dxi) < 1e-300:
            dxi = 1e-300
        dy[i] = (y[i + 1] - y[i - 1]) / dxi
    return dy


def _nan_abs_max(x: Any) -> Optional[float]:
    """
    max(abs(x)) ignorando NaN/Inf. Retorna None se não houver nenhum valor finito.
    """
    try:
        arr = np.asarray(x, dtype=float).reshape(-1)
    except Exception:
        return None
    if arr.size == 0:
        return None
    m = np.isfinite(arr)
    if not np.any(m):
        return None
    return float(np.max(np.abs(arr[m])))


def _stable_vt_theory(M: float, E: float, L: float, r: np.ndarray, pr: np.ndarray) -> np.ndarray:
    """
    vt = dv/dtau em EF ingoing.
    Forma estável (equivalente em geodésica ideal):
        vt = (1 + L^2/r^2) / (E - pr)
    Evita cancelamento catastrófico quando (E+pr)->0 e A->0.
    """
    rr = np.maximum(np.asarray(r, dtype=float), 1e-300)
    prr = np.asarray(pr, dtype=float)
    B = 1.0 + (float(L) * float(L)) / (rr * rr)

    denom = (float(E) - prr)
    denom_floor = 1e-14
    denom_safe = np.where(np.abs(denom) < denom_floor, np.sign(denom) * denom_floor, denom)
    denom_safe = np.where(denom_safe == 0.0, denom_floor, denom_safe)
    return B / denom_safe


def _plot_schw_time(
    case_name: str,
    tau: np.ndarray,
    tcoord: Optional[np.ndarray],
    vcoord: Optional[np.ndarray],
    ut_num: Optional[np.ndarray],
    ut_th: Optional[np.ndarray],
    vt_num: Optional[np.ndarray],
    vt_th: Optional[np.ndarray],
    outdir_time: str,
) -> None:
    """
    Pacote de plots de tempo (t Schwarzschild e v EF):
      - t(τ), v(τ)
      - (t-τ), (v-τ)
      - dt/dτ (num vs theory) + log
      - dv/dτ (num vs theory) + log
      - erros relativos (log) quando theory disponível
    """
    import matplotlib.pyplot as plt

    os.makedirs(outdir_time, exist_ok=True)

    # t(τ)
    if tcoord is not None:
        plt.figure()
        plt.plot(tau, tcoord)
        plt.xlabel("tau")
        plt.ylabel("tcoord")
        plt.title(f"{case_name}: t(τ) Schwarzschild")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir_time, f"{case_name}_tcoord_vs_tau.png"), dpi=150)
        plt.close()

        # (t-τ)
        plt.figure()
        plt.plot(tau, tcoord - tau)
        plt.xlabel("tau")
        plt.ylabel("tcoord - tau")
        plt.title(f"{case_name}: (t-τ)")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir_time, f"{case_name}_t_minus_tau.png"), dpi=150)
        plt.close()

    # v(τ)
    if vcoord is not None:
        plt.figure()
        plt.plot(tau, vcoord)
        plt.xlabel("tau")
        plt.ylabel("vcoord")
        plt.title(f"{case_name}: v(τ) EF ingoing")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir_time, f"{case_name}_vcoord_vs_tau.png"), dpi=150)
        plt.close()

        # (v-τ)
        plt.figure()
        plt.plot(tau, vcoord - tau)
        plt.xlabel("tau")
        plt.ylabel("vcoord - tau")
        plt.title(f"{case_name}: (v-τ)")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir_time, f"{case_name}_v_minus_tau.png"), dpi=150)
        plt.close()

    # dt/dτ
    if (ut_num is not None) and (ut_th is not None):
        plt.figure()
        plt.plot(tau, ut_num, label="num (FD)")
        plt.plot(tau, ut_th, label="theory")
        plt.xlabel("tau")
        plt.ylabel("dt/dtau")
        plt.title(f"{case_name}: dt/dτ (num vs theory)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(outdir_time, f"{case_name}_dt_dtau_overlay.png"), dpi=150)
        plt.close()

        plt.figure()
        plt.plot(tau, np.abs(ut_num) + 1e-300, label="|num|")
        plt.plot(tau, np.abs(ut_th) + 1e-300, label="|theory|")
        plt.yscale("log")
        plt.xlabel("tau")
        plt.ylabel("|dt/dtau| (log)")
        plt.title(f"{case_name}: |dt/dτ| (log)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(outdir_time, f"{case_name}_dt_dtau_log.png"), dpi=150)
        plt.close()

        with np.errstate(divide="ignore", invalid="ignore"):
            rel = np.abs((ut_num - ut_th) / np.maximum(np.abs(ut_th), 1e-300))
        plt.figure()
        plt.plot(tau, rel + 1e-300)
        plt.yscale("log")
        plt.xlabel("tau")
        plt.ylabel("rel_err |(ut_num-ut_th)/ut_th| (log)")
        plt.title(f"{case_name}: rel error dt/dτ (log)")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir_time, f"{case_name}_dt_dtau_relerr_log.png"), dpi=150)
        plt.close()

    # dv/dτ
    if (vt_num is not None) and (vt_th is not None):
        plt.figure()
        plt.plot(tau, vt_num, label="num (FD)")
        plt.plot(tau, vt_th, label="theory")
        plt.xlabel("tau")
        plt.ylabel("dv/dtau")
        plt.title(f"{case_name}: dv/dτ (num vs theory)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(outdir_time, f"{case_name}_dv_dtau_overlay.png"), dpi=150)
        plt.close()

        plt.figure()
        plt.plot(tau, np.abs(vt_num) + 1e-300, label="|num|")
        plt.plot(tau, np.abs(vt_th) + 1e-300, label="|theory|")
        plt.yscale("log")
        plt.xlabel("tau")
        plt.ylabel("|dv/dtau| (log)")
        plt.title(f"{case_name}: |dv/dτ| (log)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(outdir_time, f"{case_name}_dv_dtau_log.png"), dpi=150)
        plt.close()

        with np.errstate(divide="ignore", invalid="ignore"):
            rel = np.abs((vt_num - vt_th) / np.maximum(np.abs(vt_th), 1e-300))
        plt.figure()
        plt.plot(tau, rel + 1e-300)
        plt.yscale("log")
        plt.xlabel("tau")
        plt.ylabel("rel_err |(vt_num-vt_th)/vt_th| (log)")
        plt.title(f"{case_name}: rel error dv/dτ (log)")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir_time, f"{case_name}_dv_dtau_relerr_log.png"), dpi=150)
        plt.close()


# ============================================================
# Newton: teoria de bound/unbound
# ============================================================

def _energy_specific_newton(mu: float, state: List[float]) -> float:
    x, y, vx, vy = [float(v) for v in state]
    r = float(np.sqrt(x * x + y * y))
    v2 = float(vx * vx + vy * vy)
    return 0.5 * v2 - float(mu) / r


def _newton_status_theory(mu: float, state0: List[float]) -> str:
    e0 = _energy_specific_newton(mu, state0)
    return "BOUND" if e0 < 0.0 else "UNBOUND"


# ============================================================
# Eventos (Schwarzschild)
# ============================================================

def _extract_events(traj: Any) -> List[Dict[str, Any]]:
    if not hasattr(traj, "event_kind"):
        return []

    kind = list(getattr(traj, "event_kind"))
    tau = list(getattr(traj, "event_tau", []))
    tcoord = list(getattr(traj, "event_tcoord", []))
    vcoord = list(getattr(traj, "event_vcoord", [])) if hasattr(traj, "event_vcoord") else []
    r = list(getattr(traj, "event_r", []))
    phi = list(getattr(traj, "event_phi", []))
    pr = list(getattr(traj, "event_pr", []))

    if not vcoord:
        vcoord = [float("nan")] * len(tau)

    n = min(len(kind), len(tau), len(tcoord), len(vcoord), len(r), len(phi), len(pr))
    events: List[Dict[str, Any]] = []
    for i in range(n):
        events.append({
            "kind": str(kind[i]),
            "tau": float(tau[i]),
            "tcoord": float(tcoord[i]),
            "vcoord": float(vcoord[i]),
            "r": float(r[i]),
            "phi": float(phi[i]),
            "pr": float(pr[i]),
            "i": int(i),
        })

    events.sort(key=lambda e: float(e.get("tau", 0.0)))
    return events


def _events_compact_str(events: List[Dict[str, Any]], max_items: int = 6) -> str:
    if not events:
        return ""
    chunks = []
    for e in events[:max_items]:
        chunks.append(f"{e['kind']}@{float(e['tau']):.6g}")
    if len(events) > max_items:
        chunks.append(f"+{len(events) - max_items}")
    return "; ".join(chunks)


def _events_to_kind_map(events: List[Dict[str, Any]]) -> Dict[str, List[float]]:
    mp: Dict[str, List[float]] = defaultdict(list)
    for e in events:
        mp[str(e.get("kind", ""))].append(float(e.get("tau", 0.0)))
    return mp


def _check_event_criteria(events: List[Dict[str, Any]], crit: Dict[str, Any]) -> Tuple[bool, str]:
    kinds = [str(e.get("kind", "")) for e in events]
    kinds_set = set(kinds)

    min_events = int(crit.get("min_events", 0) or 0)
    if min_events > 0 and len(events) < min_events:
        return False, f"events: expected >= {min_events}, got {len(events)}"

    must_have = crit.get("must_have_events", None)
    if must_have is not None:
        must_have_list = [must_have] if isinstance(must_have, str) else list(must_have)
        missing = [k for k in must_have_list if str(k) not in kinds_set]
        if missing:
            return False, f"events: missing required kinds {missing}"

    must_not = crit.get("must_not_have_events", None)
    if must_not is not None:
        must_not_list = [must_not] if isinstance(must_not, str) else list(must_not)
        present = [k for k in must_not_list if str(k) in kinds_set]
        if present:
            return False, f"events: forbidden kinds present {present}"

    return True, ""


# ============================================================
# Validators
# ============================================================

def _validate_newton(case: Dict[str, Any], plotdir: Optional[str] = None) -> Dict[str, Any]:
    traj = _unwrap_traj(simulate_case(case, "newton"))

    E = list(traj.energy)
    h = list(traj.h)

    dE = _rel_drift(E)
    dh = _rel_drift(h)

    crit = case.get("criteria", {}) or {}
    dE_max = float(crit.get("energy_rel_drift_max", 2e-6))
    dh_max = float(crit.get("h_rel_drift_max", 2e-8))

    dt, n_steps = _get_solver_dt_nsteps(case)
    t0, tf = _get_span(case)
    params = case.get("params", {}) or {}
    mu = float(params.get("mu", case.get("mu", 1.0)))

    state0 = case.get("state0", None)
    energy0 = None
    theory_status = None
    if state0 is not None:
        try:
            energy0 = float(_energy_specific_newton(mu, list(state0)))
            theory_status = str(_newton_status_theory(mu, list(state0)))
        except Exception:
            energy0 = None
            theory_status = None

    status_str = str(getattr(traj, "status", ""))
    expected_status = crit.get("status", None)
    status_ok = True
    status_reason = ""
    if expected_status is not None:
        status_ok = _status_endswith(status_str, str(expected_status))
        if not status_ok:
            status_reason = f"status mismatch (got={status_str}, expected=*{expected_status})"
        if crit.get("status_must_match_theory", False) and theory_status is not None:
            if not _status_endswith(status_str, theory_status):
                status_ok = False
                status_reason = f"status != theory (got={status_str}, theory=*{theory_status})"

    passed = (dE <= dE_max) and (dh <= dh_max) and status_ok

    if plotdir is not None:
        _plot_newton(case["name"], traj, plotdir, dE_max, dh_max)

    msg = getattr(traj, "message", "") or ""
    if (not passed) and status_reason:
        msg = (msg + " | " + status_reason).strip(" |")

    y_end = None
    try:
        y_arr = np.array(traj.y, dtype=float)
        if y_arr.ndim == 2 and y_arr.shape[0] >= 1:
            y_end = [float(v) for v in y_arr[-1, :].tolist()]
    except Exception:
        y_end = None

    return {
        "name": case["name"],
        "passed": bool(passed),
        "status": status_str,
        "message": msg,
        "model": "newton",
        "mu": mu,
        "t0": t0,
        "tf": tf,
        "dt": float(dt),
        "n_steps": int(n_steps),
        "energy0": energy0,
        "status_theory": theory_status,
        "energy_rel_drift": float(dE),
        "h_rel_drift": float(dh),
        "y_end": y_end,
        "criteria": {
            "energy_rel_drift_max": dE_max,
            "h_rel_drift_max": dh_max,
            "status": expected_status,
            "status_must_match_theory": bool(crit.get("status_must_match_theory", False)),
        },
    }


def _validate_schw(
    case: Dict[str, Any],
    plotdir: Optional[str] = None,
    time_plotdir: Optional[str] = None,
) -> Dict[str, Any]:
    traj = _unwrap_traj(simulate_case(case, "schwarzschild"))

    tau = np.array(traj.tau, dtype=float)
    r = np.array(traj.r, dtype=float)

    r_min = float(np.min(r)) if r.size else None
    r_end = float(r[-1]) if r.size else None

    eps = np.array(traj.epsilon, dtype=float)
    eps_max = float(np.max(np.abs(eps))) if eps.size else None

    crit = case.get("criteria", {}) or {}
    eps_max_allowed = float(crit.get("constraint_abs_max", 1e-10))
    expected_status = str(crit.get("status", "BOUND"))

    status_str = str(getattr(traj, "status", ""))
    status_ok = _status_endswith(status_str, expected_status)

    events = _extract_events(traj)
    events_compact = _events_compact_str(events)
    events_ok, events_reason = _check_event_criteria(events, crit)

    # ----------------------------
    # Time: t(τ) e dt/dτ (e também v(τ))
    # ----------------------------
    params = case.get("params", {}) or {}
    M = float(params.get("M", case.get("M", 1.0)))
    Epar = float(params.get("E", case.get("E")))
    Lpar = float(params.get("L", case.get("L")))
    pr0 = _case_pr0(case)

    require_tcoord = bool(crit.get("require_tcoord", True))
    require_vcoord = bool(crit.get("require_vcoord", False))

    # limites para não avaliar “em cima” do horizonte (onde A->0 e t explode)
    A_min = float(crit.get("time_A_min", 1e-6))
    rel_err_max_allowed = float(crit.get("dt_dtau_rel_err_max", 1e-5))
    abs_err_max_allowed = float(crit.get("dt_dtau_abs_err_max", 1e-3))
    mono_tol = float(crit.get("tcoord_monotone_tol", 0.0))

    rel_err_v_max_allowed = float(crit.get("dv_dtau_rel_err_max", 1e-5))
    abs_err_v_max_allowed = float(crit.get("dv_dtau_abs_err_max", 1e-3))
    mono_v_tol = float(crit.get("vcoord_monotone_tol", 0.0))

    # A(r)
    A = 1.0 - (2.0 * M / np.maximum(r, 1e-300))
    mask_A = np.isfinite(A) & np.isfinite(r) & (A >= A_min) & np.isfinite(tau)
    mask_n = int(np.count_nonzero(mask_A))

    # tcoord (Schwarzschild)
    tcoord = None
    if hasattr(traj, "tcoord"):
        tcoord = np.array(getattr(traj, "tcoord"), dtype=float)
    elif hasattr(traj, "t"):
        tcoord = np.array(getattr(traj, "t"), dtype=float)
    tcoord_present = (tcoord is not None) and (tcoord.size == tau.size)

    # vcoord (EF ingoing)
    vcoord = None
    if hasattr(traj, "vcoord"):
        vcoord = np.array(getattr(traj, "vcoord"), dtype=float)
    elif hasattr(traj, "v"):
        vcoord = np.array(getattr(traj, "v"), dtype=float)
    vcoord_present = (vcoord is not None) and (vcoord.size == tau.size)

    # checks t
    tcoord_finite_ok = True
    tcoord_mono_ok = True
    dt_dtau_abs_max = None
    dt_dtau_rel_max = None

    # checks v
    vcoord_finite_ok = True
    vcoord_mono_ok = True
    dv_dtau_abs_max = None
    dv_dtau_rel_max = None

    time_reason = ""

    # ---------- tcoord ----------
    ut_num_for_plot = None
    ut_th_for_plot = None

    if require_tcoord and not tcoord_present:
        tcoord_finite_ok = False
        tcoord_mono_ok = False
        time_reason = "missing tcoord (required)"
    elif tcoord_present:
        if mask_n > 0:
            tcoord_finite_ok = bool(np.all(np.isfinite(tcoord[mask_A])))
            tcoord_mono_ok = _is_monotone_increasing(tcoord[mask_A], tol=mono_tol)
        else:
            tcoord_finite_ok = True
            tcoord_mono_ok = True

        ut_num = None
        if hasattr(traj, "ut_fd"):
            ut_fd = np.array(getattr(traj, "ut_fd"), dtype=float)
            if ut_fd.size == tau.size:
                ut_num = ut_fd
        if ut_num is None:
            ut_num = _finite_diff_first_derivative(tcoord, tau)

        ut_th = None
        if hasattr(traj, "ut_theory"):
            ut_theory = np.array(getattr(traj, "ut_theory"), dtype=float)
            if ut_theory.size == tau.size:
                ut_th = ut_theory
        if ut_th is None:
            ut_th = np.full_like(A, np.nan, dtype=float)
            ut_th[mask_A] = float(Epar) / A[mask_A]

        ut_num_for_plot = ut_num
        ut_th_for_plot = ut_th

        mask_ut = mask_A & np.isfinite(ut_num) & np.isfinite(ut_th)
        if np.any(mask_ut):
            abs_err = np.abs(ut_num[mask_ut] - ut_th[mask_ut])
            dt_dtau_abs_max = float(np.max(abs_err)) if abs_err.size else None

            with np.errstate(divide="ignore", invalid="ignore"):
                rel_err = abs_err / np.maximum(np.abs(ut_th[mask_ut]), 1e-300)
            dt_dtau_rel_max = float(np.max(rel_err)) if rel_err.size else None

            if (dt_dtau_rel_max is not None) and (dt_dtau_rel_max > rel_err_max_allowed):
                time_reason = (time_reason + " | " if time_reason else "") + f"dt/dtau rel_err_max>{rel_err_max_allowed:.1e}"
            if (dt_dtau_abs_max is not None) and (dt_dtau_abs_max > abs_err_max_allowed):
                time_reason = (time_reason + " | " if time_reason else "") + f"dt/dtau abs_err_max>{abs_err_max_allowed:.1e}"

        if not tcoord_finite_ok:
            time_reason = (time_reason + " | " if time_reason else "") + "tcoord non-finite (masked)"
        if not tcoord_mono_ok:
            time_reason = (time_reason + " | " if time_reason else "") + "tcoord not monotone (masked)"

        if time_plotdir is not None:
            try:
                _plot_schw_time(
                    case["name"],
                    tau=tau,
                    tcoord=tcoord,
                    vcoord=vcoord if vcoord_present else None,
                    ut_num=ut_num,
                    ut_th=ut_th,
                    vt_num=None,
                    vt_th=None,
                    outdir_time=time_plotdir,
                )
            except Exception:
                pass

    # ---------- vcoord ----------
    vt_num_for_plot = None
    vt_th_for_plot = None

    if require_vcoord and not vcoord_present:
        vcoord_finite_ok = False
        vcoord_mono_ok = False
        time_reason = (time_reason + " | " if time_reason else "") + "missing vcoord (required)"
    elif vcoord_present:
        # IMPORTANT: valide v majoritariamente no mesmo mask_A (onde A não está no abismo)
        if mask_n > 0:
            vcoord_finite_ok = bool(np.all(np.isfinite(vcoord[mask_A])))
            vcoord_mono_ok = _is_monotone_increasing(vcoord[mask_A], tol=mono_v_tol)
        else:
            vcoord_finite_ok = _is_finite_array(vcoord)
            vcoord_mono_ok = _is_monotone_increasing(vcoord, tol=mono_v_tol)

        vt_num = None
        if hasattr(traj, "vt_fd"):
            vt_fd = np.array(getattr(traj, "vt_fd"), dtype=float)
            if vt_fd.size == tau.size:
                vt_num = vt_fd
        if vt_num is None:
            vt_num = _finite_diff_first_derivative(vcoord, tau)

        vt_th = None
        if hasattr(traj, "vt_theory"):
            vt_theory = np.array(getattr(traj, "vt_theory"), dtype=float)
            if vt_theory.size == tau.size:
                vt_th = vt_theory
        if vt_th is None:
            pr = np.array(getattr(traj, "pr"), dtype=float)
            vt_th = np.full_like(A, np.nan, dtype=float)
            vt_th[mask_A] = _stable_vt_theory(M, Epar, Lpar, r[mask_A], pr[mask_A])

        vt_num_for_plot = vt_num
        vt_th_for_plot = vt_th

        mask_vt = mask_A & np.isfinite(vt_num) & np.isfinite(vt_th)
        if np.any(mask_vt):
            abs_err = np.abs(vt_num[mask_vt] - vt_th[mask_vt])
            dv_dtau_abs_max = float(np.max(abs_err)) if abs_err.size else None

            with np.errstate(divide="ignore", invalid="ignore"):
                rel_err = abs_err / np.maximum(np.abs(vt_th[mask_vt]), 1e-300)
            dv_dtau_rel_max = float(np.max(rel_err)) if rel_err.size else None

            if (dv_dtau_rel_max is not None) and (dv_dtau_rel_max > rel_err_v_max_allowed):
                time_reason = (time_reason + " | " if time_reason else "") + f"dv/dtau rel_err_max>{rel_err_v_max_allowed:.1e}"
            if (dv_dtau_abs_max is not None) and (dv_dtau_abs_max > abs_err_v_max_allowed):
                time_reason = (time_reason + " | " if time_reason else "") + f"dv/dtau abs_err_max>{abs_err_v_max_allowed:.1e}"

        if not vcoord_finite_ok:
            time_reason = (time_reason + " | " if time_reason else "") + "vcoord non-finite (masked)"
        if not vcoord_mono_ok:
            time_reason = (time_reason + " | " if time_reason else "") + "vcoord not monotone (masked)"

        if time_plotdir is not None:
            try:
                _plot_schw_time(
                    case["name"],
                    tau=tau,
                    tcoord=tcoord if tcoord_present else None,
                    vcoord=vcoord,
                    ut_num=ut_num_for_plot,
                    ut_th=ut_th_for_plot,
                    vt_num=vt_num,
                    vt_th=vt_th,
                    outdir_time=time_plotdir,
                )
            except Exception:
                pass

    # decisão time_ok
    time_ok = True
    if require_tcoord:
        time_ok = time_ok and bool(tcoord_present and tcoord_finite_ok and tcoord_mono_ok)
        if dt_dtau_rel_max is not None:
            time_ok = time_ok and (dt_dtau_rel_max <= rel_err_max_allowed)
        if dt_dtau_abs_max is not None:
            time_ok = time_ok and (dt_dtau_abs_max <= abs_err_max_allowed)

    if require_vcoord:
        time_ok = time_ok and bool(vcoord_present and vcoord_finite_ok and vcoord_mono_ok)
        if dv_dtau_rel_max is not None:
            time_ok = time_ok and (dv_dtau_rel_max <= rel_err_v_max_allowed)
        if dv_dtau_abs_max is not None:
            time_ok = time_ok and (dv_dtau_abs_max <= abs_err_v_max_allowed)

    # ----------------------------
    # norm_u: usar THEORY como juiz (FD é diagnóstico)
    # ----------------------------
    norm_u_fd_max = None
    norm_u_theory_max = None

    if hasattr(traj, "norm_u"):
        nu_fd = np.array(getattr(traj, "norm_u"), dtype=float)
        if nu_fd.size != tau.size:
            raise RuntimeError(
                f"norm_u shape mismatch no caso '{case.get('name')}': "
                f"len(norm_u)={nu_fd.size} vs len(tau)={tau.size}"
            )
        # FD explode perto do horizonte; ainda assim calculamos para depuração
        norm_u_fd_max = _nan_abs_max(nu_fd)

    if hasattr(traj, "norm_u_theory"):
        nu_th = np.array(getattr(traj, "norm_u_theory"), dtype=float)
        if nu_th.size == tau.size:
            if mask_n > 0:
                norm_u_theory_max = _nan_abs_max(nu_th[mask_A])
            else:
                norm_u_theory_max = _nan_abs_max(nu_th)

    # métrica principal publicada (para convergência): theory se existir; senão, cai pro FD
    norm_u_max_primary = norm_u_theory_max if (norm_u_theory_max is not None) else norm_u_fd_max

    passed = (
        (eps_max is not None and eps_max <= eps_max_allowed)
        and status_ok
        and events_ok
        and bool(time_ok)
    )

    dt, n_steps = _get_solver_dt_nsteps(case)
    tau0, tauf = _get_span(case)

    if plotdir is not None:
        _plot_schw(case["name"], traj, plotdir, eps_max_allowed)

    msg = getattr(traj, "message", "") or ""
    if (not passed) and (not events_ok) and events_reason:
        msg = (msg + " | " + events_reason).strip(" |")
    if (not passed) and time_reason:
        msg = (msg + " | " + time_reason).strip(" |")

    out = {
        "name": case["name"],
        "passed": bool(passed),
        "status": status_str,
        "message": msg,
        "model": "schwarzschild",
        "M": M,
        "E": Epar,
        "L": Lpar,
        "pr0": pr0,
        "tau0": tau0,
        "tauf": tauf,
        "dt": float(dt),
        "n_steps": int(n_steps),
        "r_min": r_min,
        "r_end": r_end,
        "constraint_abs_max": float(eps_max) if eps_max is not None else None,

        # norm_u (principal = theory, fallback = FD)
        "norm_u_abs_max": norm_u_max_primary,
        "norm_u_abs_max_theory": norm_u_theory_max,
        "norm_u_abs_max_fd": norm_u_fd_max,

        "events": events,
        "events_compact": events_compact,

        # --- t diagnostics ---
        "tcoord_present": bool(tcoord_present),
        "tcoord_finite_ok": bool(tcoord_finite_ok),
        "tcoord_monotone_ok": bool(tcoord_mono_ok),
        "dt_dtau_abs_max": dt_dtau_abs_max,
        "dt_dtau_rel_max": dt_dtau_rel_max,

        # --- v diagnostics ---
        "vcoord_present": bool(vcoord_present),
        "vcoord_finite_ok": bool(vcoord_finite_ok),
        "vcoord_monotone_ok": bool(vcoord_mono_ok),
        "dv_dtau_abs_max": dv_dtau_abs_max,
        "dv_dtau_rel_max": dv_dtau_rel_max,

        # mask info
        "time_mask_A_min": float(A_min),
        "time_mask_n": int(mask_n),

        "criteria": {
            "constraint_abs_max": eps_max_allowed,
            "status": expected_status,
            "min_events": int(crit.get("min_events", 0) or 0),
            "must_have_events": crit.get("must_have_events", None),
            "must_not_have_events": crit.get("must_not_have_events", None),

            # time criteria
            "require_tcoord": require_tcoord,
            "require_vcoord": require_vcoord,
            "time_A_min": A_min,

            "tcoord_monotone_tol": mono_tol,
            "dt_dtau_rel_err_max": rel_err_max_allowed,
            "dt_dtau_abs_err_max": abs_err_max_allowed,

            "vcoord_monotone_tol": mono_v_tol,
            "dv_dtau_rel_err_max": rel_err_v_max_allowed,
            "dv_dtau_abs_err_max": abs_err_v_max_allowed,
        },
    }
    return out


# ============================================================
# Convergência automática (Newton) — varrer dt, estimar ordem RK4
# ============================================================

def _clone_case_with_dt_consistent(case: Dict[str, Any], dt_target: float) -> Dict[str, Any]:
    c = copy.deepcopy(case)
    t0, tf = _get_span(c)
    T = float(tf - t0)
    if T <= 0.0:
        dt0, ns0 = _get_solver_dt_nsteps(c)
        _set_solver_dt_nsteps(c, float(dt0), int(ns0))
        return c

    n_steps = int(max(1, round(T / float(dt_target))))
    dt_eff = T / float(n_steps)
    _set_solver_dt_nsteps(c, float(dt_eff), int(n_steps))
    return c


def _traj_max_diff_coarse_vs_fine(tr_coarse: Any, tr_fine: Any, stride: int) -> float:
    yc = np.array(tr_coarse.y, dtype=float)
    yf = np.array(tr_fine.y, dtype=float)

    n = yc.shape[0]
    need = (n - 1) * stride + 1
    if yf.shape[0] < need:
        raise RuntimeError(
            f"traj_fine curto demais para stride={stride}: "
            f"len(y_fine)={yf.shape[0]} need>={need}"
        )

    deltas = yc - yf[0:need:stride, :]
    norms = np.linalg.norm(deltas, axis=1)
    return float(np.max(norms))


def _roundoff_floor_estimate(tr: Any) -> float:
    y = np.array(tr.y, dtype=float)
    N = max(1, y.shape[0])
    scale = max(1.0, float(np.max(np.linalg.norm(y, axis=1))))
    eps = float(np.finfo(float).eps)
    C = 200.0
    return float(C * eps * scale * math.sqrt(N))


def _run_convergence_newton_one_case(
    base_case: Dict[str, Any],
    plotdir: Optional[str],
    rigorous: bool = True,
) -> Dict[str, Any]:
    base_name = str(base_case.get("name", "<sem-nome>"))
    crit = base_case.get("criteria", {}) or {}

    min_order = float(crit.get("convergence_min_order", 3.6 if rigorous else 3.2))
    rel_err_max = float(crit.get("convergence_rel_err_max", 1e-6 if rigorous else 1e-5))
    abs_err_max = float(crit.get("convergence_abs_err_max", 1e-9 if rigorous else 1e-8))

    dt0, _ = _get_solver_dt_nsteps(base_case)
    dt_targets = [float(dt0), float(dt0) / 2.0, float(dt0) / 4.0]

    cases = [_clone_case_with_dt_consistent(base_case, dt_t) for dt_t in dt_targets]

    trajs: List[Any] = []
    runs: List[Dict[str, Any]] = []
    dts_eff: List[float] = []

    for i, c in enumerate(cases):
        dt_eff, ns_eff = _get_solver_dt_nsteps(c)
        dts_eff.append(float(dt_eff))
        c["name"] = f"{base_name}__conv{i}_dt{dt_eff:.3e}"

        r = _validate_newton(c, plotdir=None)
        runs.append(r)

        tr = _unwrap_traj(simulate_case(c, "newton"))
        trajs.append(tr)

    tr0, tr1, tr2 = trajs

    try:
        e_dt = _traj_max_diff_coarse_vs_fine(tr0, tr1, stride=2)
        e_dt2 = _traj_max_diff_coarse_vs_fine(tr1, tr2, stride=2)
        p_obs = _estimate_order_from_three(e_dt, e_dt2)
        ratio = (e_dt / e_dt2) if (e_dt2 is not None and e_dt2 > 0.0) else None
    except Exception as ex:
        return {
            "name": base_name,
            "passed": False,
            "inconclusive": True,
            "reason": f"trajectory alignment failed: {ex}",
            "dt_targets": dt_targets,
            "dt_effective": dts_eff,
            "e_dt": None,
            "e_dt2": None,
            "ratio_e": None,
            "p_obs": None,
            "abs_err_proxy": None,
            "rel_err_proxy": None,
            "criteria": {
                "convergence_min_order": min_order,
                "convergence_abs_err_max": abs_err_max,
                "convergence_rel_err_max": rel_err_max,
            },
            "runs": [
                {
                    "dt": r.get("dt"),
                    "n_steps": r.get("n_steps"),
                    "energy_rel_drift": r.get("energy_rel_drift"),
                    "h_rel_drift": r.get("h_rel_drift"),
                    "status": r.get("status"),
                    "passed_invariants": r.get("passed"),
                    "msg": _short_msg(str(r.get("message", ""))),
                }
                for r in runs
            ],
        }

    abs_err_proxy = float(e_dt2)

    y2 = np.array(tr2.y, dtype=float)
    scale = max(1.0, float(np.max(np.linalg.norm(y2, axis=1))))
    rel_err_proxy = float(abs_err_proxy / scale)

    floor2 = _roundoff_floor_estimate(tr2)
    saturated = bool(abs_err_proxy <= 5.0 * floor2)

    ok_monotone = bool(e_dt2 < e_dt)
    ok_abs = bool(abs_err_proxy <= abs_err_max)
    ok_rel = bool(rel_err_proxy <= rel_err_max)
    ok_order = bool((p_obs is not None) and (p_obs >= min_order))

    passed = bool(ok_monotone and ok_abs and ok_rel and (ok_order or saturated))

    reason_parts = []
    if not ok_monotone:
        reason_parts.append("error not decreasing")
    if not ok_abs:
        reason_parts.append(f"abs_err_proxy>{abs_err_max:.1e}")
    if not ok_rel:
        reason_parts.append(f"rel_err_proxy>{rel_err_max:.1e}")
    if not ok_order and not saturated:
        reason_parts.append(f"p_obs<{min_order:.2f}")
    if saturated and not ok_order:
        reason_parts.append("SATURATED(roundoff floor)")

    reason = "; ".join(reason_parts) if reason_parts else ""

    if plotdir is not None:
        os.makedirs(plotdir, exist_ok=True)
        try:
            err0 = _traj_max_diff_coarse_vs_fine(tr0, tr2, stride=4)
            err1 = _traj_max_diff_coarse_vs_fine(tr1, tr2, stride=2)
            dts_plot = [dts_eff[0], dts_eff[1]]
            errs_plot = [err0, err1]
            _plot_convergence_newton(base_name, dts_plot, errs_plot, plotdir, p_obs)

            labels = [f"dt={dts_eff[0]:.3e}", f"dt={dts_eff[1]:.3e}", f"dt={dts_eff[2]:.3e}"]
            _plot_convergence_overlay_newton(base_name, trajs, labels, plotdir)
        except Exception:
            pass

    return {
        "name": base_name,
        "passed": bool(passed),
        "inconclusive": False,
        "reason": reason,
        "dt_targets": dt_targets,
        "dt_effective": dts_eff,
        "e_dt": float(e_dt),
        "e_dt2": float(e_dt2),
        "ratio_e": float(ratio) if ratio is not None else None,
        "p_obs": float(p_obs) if p_obs is not None else None,
        "abs_err_proxy": float(abs_err_proxy),
        "rel_err_proxy": float(rel_err_proxy),
        "roundoff_floor_est": float(floor2),
        "saturated": bool(saturated),
        "criteria": {
            "convergence_min_order": min_order,
            "convergence_abs_err_max": abs_err_max,
            "convergence_rel_err_max": rel_err_max,
        },
        "runs": [
            {
                "dt": r.get("dt"),
                "n_steps": r.get("n_steps"),
                "energy_rel_drift": r.get("energy_rel_drift"),
                "h_rel_drift": r.get("h_rel_drift"),
                "status": r.get("status"),
                "passed_invariants": r.get("passed"),
                "msg": _short_msg(str(r.get("message", ""))),
            }
            for r in runs
        ],
    }


# ============================================================
# Convergência automática (Schwarzschild) — checks existentes
# ============================================================

def _schw_signature(case: Dict[str, Any]) -> str:
    params = case.get("params", {}) or {}
    state0 = case.get("state0", None)
    span = case.get("span", None)

    sig = {
        "model": "schwarzschild",
        "M": float(params.get("M", case.get("M", 1.0))),
        "E": float(params.get("E", case.get("E"))),
        "L": float(params.get("L", case.get("L"))),
        "state0": state0,
        "pr0": float(_case_pr0(case)),
        "span": span,
        "capture_r": float(params.get("capture_r", case.get("capture_r", 2.0))),
        "capture_eps": float(params.get("capture_eps", case.get("capture_eps", 1e-12))),
    }
    return json.dumps(sig, sort_keys=True, separators=(",", ":"))


def _conv_allows_increase(nu_big: float, nu_small: float, abs_tol: float, rel_tol: float) -> bool:
    return nu_small <= (nu_big + abs_tol + rel_tol * max(nu_big, 0.0))


def _check_convergence_schw(
    results: List[Dict[str, Any]],
    abs_tol: float = 1e-9,
    rel_tol: float = 0.25,
) -> List[Dict[str, Any]]:
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for r in results:
        key = r.get("_sig", None)
        if key is None:
            continue
        groups.setdefault(key, []).append(r)

    conv_reports: List[Dict[str, Any]] = []

    for _key, grp in groups.items():
        if len(grp) < 2:
            continue

        grp_sorted = sorted(grp, key=lambda x: float(x.get("dt", 0.0)), reverse=True)

        dts = [float(g["dt"]) for g in grp_sorted]
        nus_raw = [g.get("norm_u_abs_max", None) for g in grp_sorted]  # primary (theory preferred)
        names = [g.get("name", "") for g in grp_sorted]

        if any(v is None for v in nus_raw):
            conv_reports.append({
                "passed": False,
                "inconclusive": True,
                "reason": "missing norm_u_abs_max in at least one run",
                "cases": names,
                "dts": dts,
                "norm_u_abs_max": nus_raw,
                "violations": [],
                "abs_tol": abs_tol,
                "rel_tol": rel_tol,
            })
            continue

        nus = [float(v) for v in nus_raw]  # type: ignore[arg-type]

        ok = True
        violations: List[Dict[str, Any]] = []
        for i in range(len(nus) - 1):
            nu_big = nus[i]
            nu_small = nus[i + 1]
            if not _conv_allows_increase(nu_big, nu_small, abs_tol=abs_tol, rel_tol=rel_tol):
                ok = False
                violations.append({
                    "dt_big": dts[i],
                    "nu_big": nu_big,
                    "dt_small": dts[i + 1],
                    "nu_small": nu_small,
                    "ratio": (nu_small / nu_big) if nu_big != 0 else None,
                    "abs_tol": abs_tol,
                    "rel_tol": rel_tol,
                })

        conv_reports.append({
            "passed": bool(ok),
            "inconclusive": False,
            "cases": names,
            "dts": dts,
            "norm_u_abs_max": nus,
            "violations": violations,
            "abs_tol": abs_tol,
            "rel_tol": rel_tol,
        })

    return conv_reports


def _check_convergence_events_schw(
    results: List[Dict[str, Any]],
    abs_tol_factor: float = 2.0,
    rel_tol: float = 0.0,
) -> List[Dict[str, Any]]:
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for r in results:
        key = r.get("_sig", None)
        if key is None:
            continue
        groups.setdefault(key, []).append(r)

    conv_reports: List[Dict[str, Any]] = []

    for _key, grp in groups.items():
        if len(grp) < 2:
            continue

        grp_sorted = sorted(grp, key=lambda x: float(x.get("dt", 0.0)), reverse=True)

        dts = [float(g.get("dt", 0.0)) for g in grp_sorted]
        names = [g.get("name", "") for g in grp_sorted]

        ev_lists = [g.get("events", None) for g in grp_sorted]
        if any(v is None for v in ev_lists):
            conv_reports.append({
                "passed": False,
                "inconclusive": True,
                "reason": "missing events field in at least one run",
                "cases": names,
                "dts": dts,
                "violations": [],
                "mismatches": [],
                "abs_tol_factor": abs_tol_factor,
                "rel_tol": rel_tol,
            })
            continue

        counts = [len(list(v or [])) for v in ev_lists]
        has_any = any(c > 0 for c in counts)
        has_all = all(c > 0 for c in counts)

        if not has_any:
            conv_reports.append({
                "passed": True,
                "skipped": True,
                "inconclusive": False,
                "reason": "no events detected in any run for this group (N/A)",
                "cases": names,
                "dts": dts,
                "violations": [],
                "mismatches": [],
                "abs_tol_factor": abs_tol_factor,
                "rel_tol": rel_tol,
            })
            continue

        if has_any and not has_all:
            conv_reports.append({
                "passed": False,
                "skipped": False,
                "inconclusive": False,
                "reason": "events detected in some runs but not all (inconsistent event detection)",
                "cases": names,
                "dts": dts,
                "counts": counts,
                "violations": [],
                "mismatches": [{"kind": "*", "detail": "some runs have events, others do not"}],
                "abs_tol_factor": abs_tol_factor,
                "rel_tol": rel_tol,
            })
            continue

        ok = True
        violations: List[Dict[str, Any]] = []
        mismatches: List[Dict[str, Any]] = []

        for i in range(len(grp_sorted) - 1):
            g_big = grp_sorted[i]
            g_small = grp_sorted[i + 1]

            dt_big = float(g_big.get("dt", 0.0))
            dt_small = float(g_small.get("dt", 0.0))
            abs_tol = abs_tol_factor * (dt_small if dt_small > 0.0 else dt_big)

            ev_big = list(g_big.get("events", []) or [])
            ev_small = list(g_small.get("events", []) or [])

            mp_big = _events_to_kind_map(ev_big)
            mp_small = _events_to_kind_map(ev_small)

            kinds = sorted(set(mp_big.keys()) | set(mp_small.keys()))

            for kind in kinds:
                a = mp_big.get(kind, [])
                b = mp_small.get(kind, [])

                if len(a) != len(b):
                    ok = False
                    mismatches.append({
                        "kind": kind,
                        "dt_big": dt_big,
                        "dt_small": dt_small,
                        "count_big": len(a),
                        "count_small": len(b),
                        "tau_big": a,
                        "tau_small": b,
                    })

                mlen = min(len(a), len(b))
                for j in range(mlen):
                    tau_big = float(a[j])
                    tau_small = float(b[j])
                    err = abs(tau_small - tau_big)
                    allowed = abs_tol + rel_tol * max(abs(tau_big), 1.0)
                    if err > allowed:
                        ok = False
                        violations.append({
                            "kind": kind,
                            "occurrence": j,
                            "dt_big": dt_big,
                            "dt_small": dt_small,
                            "tau_big": tau_big,
                            "tau_small": tau_small,
                            "abs_err": err,
                            "allowed": allowed,
                            "abs_tol": abs_tol,
                            "rel_tol": rel_tol,
                        })

        conv_reports.append({
            "passed": bool(ok),
            "skipped": False,
            "inconclusive": False,
            "cases": names,
            "dts": dts,
            "abs_tol_factor": abs_tol_factor,
            "rel_tol": rel_tol,
            "violations": violations,
            "mismatches": mismatches,
        })

    return conv_reports


# ============================================================
# Output bonito (terminal)
# ============================================================

def _print_header(title: str) -> None:
    print("\n" + title)
    print("-" * len(title))


def _print_table(rows: List[List[str]], headers: List[str]) -> None:
    cols = len(headers)
    widths = [len(h) for h in headers]
    for r in rows:
        for i in range(cols):
            widths[i] = max(widths[i], len(r[i]))

    def fmt_row(r: List[str]) -> str:
        return "  ".join(r[i].ljust(widths[i]) for i in range(cols))

    print(fmt_row(headers))
    print("  ".join("-" * widths[i] for i in range(cols)))
    for r in rows:
        print(fmt_row(r))


# ============================================================
# Main
# ============================================================

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default=os.path.join(os.path.dirname(__file__), "cases.yaml"))
    ap.add_argument("--plots", action="store_true", help="Generate plots in <out>/plots and time plots in <out>/time_plots")
    ap.add_argument("--out", default="out", help="Output directory")

    ap.add_argument(
        "--convergence",
        action="store_true",
        help="Run automatic dt refinement (dt, dt/2, dt/4) for Newton cases and estimate observed order.",
    )
    ap.add_argument(
        "--conv-rigorous",
        action="store_true",
        help="Use stricter defaults for convergence criteria (can be overridden per-case in YAML criteria.*).",
    )

    args = ap.parse_args()

    print(engine_hello())

    cfg = load_cases_yaml(args.cases)
    outdir = args.out

    plotdir = os.path.join(outdir, "plots")
    time_plotdir = os.path.join(outdir, "time_plots")

    if args.plots or args.convergence:
        os.makedirs(plotdir, exist_ok=True)
    if args.plots:
        os.makedirs(time_plotdir, exist_ok=True)

    report: Dict[str, Any] = {"suites": []}

    # ----------------------------
    # Newton
    # ----------------------------
    newton_cases = cfg["suites"]["newton"]["cases"]
    newton_results: List[Dict[str, Any]] = []
    ok_newton = True

    for c in newton_cases:
        r = _validate_newton(c, plotdir if args.plots else None)
        newton_results.append(r)
        ok_newton = ok_newton and r["passed"]

    _print_header(f"Newton suite: ok={ok_newton} cases={len(newton_cases)}")
    n_rows: List[List[str]] = []
    for r in newton_results:
        n_rows.append([
            "PASS" if r["passed"] else "FAIL",
            r["name"],
            f"{r['dt']:.1e}",
            _fmt_e(r["energy_rel_drift"], width=12),
            _fmt_e(r["h_rel_drift"], width=12),
            str(r.get("status", "")),
            str(r.get("status_theory", "")) if r.get("status_theory") else "",
        ])
    _print_table(n_rows, headers=["ok", "case", "dt", "dE_rel", "dh_rel", "status", "theory"])

    newton_suite_block: Dict[str, Any] = {
        "suite": "newton",
        "ok": ok_newton,
        "n_cases": len(newton_cases),
        "results": newton_results,
    }

    conv_reports: List[Dict[str, Any]] = []
    conv_ok = True
    if args.convergence:
        for c in newton_cases:
            cr = _run_convergence_newton_one_case(
                c,
                plotdir if (args.plots or args.convergence) else None,
                rigorous=bool(args.conv_rigorous),
            )
            conv_reports.append(cr)
            conv_ok = conv_ok and bool(cr.get("passed", False))

        _print_header(f"Newton convergence: ok={conv_ok} groups={len(conv_reports)}")

        c_rows: List[List[str]] = []
        for g in conv_reports:
            tag = "PASS" if g["passed"] else ("INCONCLUSIVE" if g.get("inconclusive") else "FAIL")
            dts = g.get("dt_effective", [])
            dts_str = ", ".join([f"{float(dt):.2e}" for dt in dts]) if dts else "-"
            c_rows.append([
                tag,
                g["name"],
                dts_str,
                _fmt_e(g.get("e_dt"), width=12),
                _fmt_e(g.get("e_dt2"), width=12),
                _fmt_f(g.get("p_obs"), width=8, prec=3),
                _fmt_e(g.get("abs_err_proxy"), width=12),
                _fmt_e(g.get("rel_err_proxy"), width=12),
                _short_msg(str(g.get("reason", ""))),
            ])
        _print_table(
            c_rows,
            headers=["ok", "case", "dt_eff (dt,dt/2,dt/4)", "e_dt", "e_dt2", "p_obs", "abs_err", "rel_err", "reason"],
        )

        newton_suite_block["ok_convergence"] = bool(conv_ok)
        newton_suite_block["convergence"] = conv_reports
        newton_suite_block["ok_total"] = bool(ok_newton and conv_ok)

    report["suites"].append(newton_suite_block)

    # ----------------------------
    # Schwarzschild
    # ----------------------------
    schw_cases = cfg["suites"]["schwarzschild"]["cases"]
    schw_results: List[Dict[str, Any]] = []
    ok_schw_cases = True

    for c in schw_cases:
        rr = _validate_schw(
            c,
            plotdir if args.plots else None,
            time_plotdir if args.plots else None,
        )
        rr["_sig"] = _schw_signature(c)
        schw_results.append(rr)
        ok_schw_cases = ok_schw_cases and rr["passed"]

    conv = _check_convergence_schw(schw_results, abs_tol=1e-9, rel_tol=0.25)
    conv_ok_s = all(bool(x["passed"]) for x in conv) if conv else False

    events_conv = _check_convergence_events_schw(schw_results, abs_tol_factor=2.0, rel_tol=0.0)
    events_conv_ok = all(bool(x["passed"]) for x in events_conv) if events_conv else False

    ok_schw_total = bool(ok_schw_cases and conv_ok_s and events_conv_ok)

    _print_header(
        f"Schwarzschild suite: ok={ok_schw_total} "
        f"(cases_ok={ok_schw_cases}, conv_ok={conv_ok_s}, events_conv_ok={events_conv_ok}) cases={len(schw_cases)}"
    )

    s_rows: List[List[str]] = []
    for r in schw_results:
        s_rows.append([
            "PASS" if r["passed"] else "FAIL",
            r["name"],
            f"{r['dt']:.1e}",
            _fmt_f(r.get("r_min"), width=10, prec=6),
            _fmt_f(r.get("r_end"), width=10, prec=6),
            _fmt_e(r.get("constraint_abs_max"), width=12),
            _fmt_e(r.get("norm_u_abs_max"), width=12),        # primary (theory preferred)
            _fmt_e(r.get("norm_u_abs_max_fd"), width=12),     # diagnostic
            str(r.get("status", "")),
            r.get("events_compact", "") or "",
            _short_msg(str(r.get("message", ""))),
        ])
    _print_table(
        s_rows,
        headers=["ok", "case", "dt", "r_min", "r_end", "eps_max", "norm_u", "norm_u_fd", "status", "events", "msg"]
    )

    _print_header("Schwarzschild events (per run)")
    any_events = False
    for r in schw_results:
        evs = r.get("events", []) or []
        if evs:
            any_events = True
            print(f"{r['name']} (dt={r['dt']:.2e}): {_events_compact_str(evs)}")
    if not any_events:
        print("No events detected in these runs.")

    _print_header("Schwarzschild time-dilation checks (t(τ), v(τ), dt/dτ, dv/dτ)")
    td_rows: List[List[str]] = []
    for r in schw_results:
        td_rows.append([
            "OK" if r["passed"] else "WARN/FAIL",
            r["name"],
            "yes" if r.get("tcoord_present") else "no",
            "yes" if r.get("tcoord_finite_ok") else "no",
            "yes" if r.get("tcoord_monotone_ok") else "no",
            _fmt_e(r.get("dt_dtau_rel_max"), width=12),
            _fmt_e(r.get("dt_dtau_abs_max"), width=12),
            "yes" if r.get("vcoord_present") else "no",
            "yes" if r.get("vcoord_finite_ok") else "no",
            "yes" if r.get("vcoord_monotone_ok") else "no",
            _fmt_e(r.get("dv_dtau_rel_max"), width=12),
            _fmt_e(r.get("dv_dtau_abs_max"), width=12),
            str(r.get("time_mask_n", "")),
        ])
    _print_table(
        td_rows,
        headers=["ok", "case", "t", "t_finite", "t_mono", "dt_rel", "dt_abs", "v", "v_finite", "v_mono", "dv_rel", "dv_abs", "mask_n"],
    )

    _print_header("Schwarzschild convergence: norm_u_abs_max should not increase when dt decreases (with tolerance)")
    if not conv:
        print("No comparable groups found. Need >=2 cases with same physics and different dt.")
    else:
        c_rows2: List[List[str]] = []
        for g in conv:
            tag = "PASS" if g["passed"] else ("INCONCLUSIVE" if g.get("inconclusive") else "FAIL")
            dts = ", ".join([f"{dt:.2e}" for dt in g["dts"]])
            nus = ", ".join(["None" if v is None else f"{float(v):.3e}" for v in g["norm_u_abs_max"]])
            c_rows2.append([tag, dts, nus, ", ".join(g["cases"])])
        _print_table(c_rows2, headers=["ok", "dt (big->small)", "norm_u_abs_max", "cases"])

        for g in conv:
            if g.get("violations"):
                for v in g["violations"]:
                    print(
                        f"violation: dt {v['dt_big']:.2e}->{v['dt_small']:.2e} "
                        f"norm_u {v['nu_big']:.3e}->{v['nu_small']:.3e} "
                        f"(abs_tol={v['abs_tol']:.1e}, rel_tol={v['rel_tol']:.2f})"
                    )

    _print_header("Schwarzschild convergence: event times should change little when dt decreases")
    if not events_conv:
        print("No comparable groups found. Need >=2 cases with same physics and different dt.")
    else:
        e_rows: List[List[str]] = []
        for g in events_conv:
            tag = "PASS" if g["passed"] else (
                "SKIP" if g.get("skipped") else ("INCONCLUSIVE" if g.get("inconclusive") else "FAIL")
            )
            dts = ", ".join([f"{dt:.2e}" for dt in g["dts"]])
            reason = str(g.get("reason", "")) if (g.get("skipped") or g.get("inconclusive")) else ""
            e_rows.append([tag, dts, ", ".join(g["cases"]), reason])
        _print_table(e_rows, headers=["ok", "dt (big->small)", "cases", "reason"])

        for g in events_conv:
            for mm in g.get("mismatches", []) or []:
                print(
                    "mismatch: "
                    f"{mm.get('kind','?')} count dt {mm.get('dt_big',0.0):.2e}->{mm.get('dt_small',0.0):.2e} "
                    f"{mm.get('count_big','?')}->{mm.get('count_small','?')}"
                )
            for v in g.get("violations", []) or []:
                print(
                    f"violation: {v['kind']}[{v['occurrence']}] dt {v['dt_big']:.2e}->{v['dt_small']:.2e} "
                    f"tau {v['tau_big']:.6g}->{v['tau_small']:.6g} "
                    f"abs_err={v['abs_err']:.3e} allowed={v['allowed']:.3e} "
                    f"(abs_tol={v['abs_tol']:.3e}, rel_tol={v['rel_tol']:.2f})"
                )

    report["suites"].append({
        "suite": "schwarzschild",
        "ok": ok_schw_total,
        "ok_cases": ok_schw_cases,
        "ok_convergence": conv_ok_s,
        "ok_events_convergence": events_conv_ok,
        "n_cases": len(schw_cases),
        "results": schw_results,
        "convergence": conv,
        "events_convergence": events_conv,
    })

    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    if args.plots or args.convergence:
        print(f"\nPlots em: {plotdir}")
    if args.plots:
        print(f"Time plots em: {time_plotdir}")
    print(f"Relatório em: {os.path.join(outdir, 'report.json')}")


if __name__ == "__main__":
    main()
