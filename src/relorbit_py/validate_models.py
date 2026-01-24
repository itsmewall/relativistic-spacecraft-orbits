# src/relorbit_py/validate_models.py
from __future__ import annotations

import copy
import json
import math
import os
from typing import Any, Dict, List, Optional

import numpy as np

from .simulate import simulate_case
from .plots import (
    rel_drift as _rel_drift,
    plot_newton as _plot_newton,
    plot_schw as _plot_schw,
    plot_convergence_newton as _plot_convergence_newton,
    plot_convergence_overlay_newton as _plot_convergence_overlay_newton,
)
from .plots_time import plot_schw_time as _plot_schw_time

from .validate_helpers import (
    unwrap_traj as _unwrap_traj,
    get_solver_dt_nsteps as _get_solver_dt_nsteps,
    set_solver_dt_nsteps as _set_solver_dt_nsteps,
    get_span as _get_span,
    case_pr0 as _case_pr0,
    short_msg as _short_msg,
    status_endswith as _status_endswith,
    estimate_order_from_three as _estimate_order_from_three,
    is_finite_array as _is_finite_array,
    is_monotone_increasing as _is_monotone_increasing,
    finite_diff_first_derivative as _finite_diff_first_derivative,
    nan_abs_max as _nan_abs_max,
    extract_events as _extract_events,
    events_compact_str as _events_compact_str,
    events_to_kind_map as _events_to_kind_map,
    check_event_criteria as _check_event_criteria,
)


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
# Validators
# ============================================================

def validate_newton(case: Dict[str, Any], plotdir: Optional[str] = None) -> Dict[str, Any]:
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


def validate_schw(
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
            vt_th[mask_A] = (float(Epar) + pr[mask_A]) / A[mask_A]

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
            time_reason = (time_reason + " | " if time_reason else "") + "vcoord non-finite"
        if not vcoord_mono_ok:
            time_reason = (time_reason + " | " if time_reason else "") + "vcoord not monotone"

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
        norm_u_fd_max = _nan_abs_max(nu_fd)

    if hasattr(traj, "norm_u_theory"):
        nu_th = np.array(getattr(traj, "norm_u_theory"), dtype=float)
        if nu_th.size == tau.size:
            if mask_n > 0:
                norm_u_theory_max = _nan_abs_max(nu_th[mask_A])
            else:
                norm_u_theory_max = _nan_abs_max(nu_th)

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

        "norm_u_abs_max": norm_u_max_primary,
        "norm_u_abs_max_theory": norm_u_theory_max,
        "norm_u_abs_max_fd": norm_u_fd_max,

        "events": events,
        "events_compact": events_compact,

        "tcoord_present": bool(tcoord_present),
        "tcoord_finite_ok": bool(tcoord_finite_ok),
        "tcoord_monotone_ok": bool(tcoord_mono_ok),
        "dt_dtau_abs_max": dt_dtau_abs_max,
        "dt_dtau_rel_max": dt_dtau_rel_max,

        "vcoord_present": bool(vcoord_present),
        "vcoord_finite_ok": bool(vcoord_finite_ok),
        "vcoord_monotone_ok": bool(vcoord_mono_ok),
        "dv_dtau_abs_max": dv_dtau_abs_max,
        "dv_dtau_rel_max": dv_dtau_rel_max,

        "time_mask_A_min": float(A_min),
        "time_mask_n": int(mask_n),

        "criteria": {
            "constraint_abs_max": eps_max_allowed,
            "status": expected_status,
            "min_events": int(crit.get("min_events", 0) or 0),
            "must_have_events": crit.get("must_have_events", None),
            "must_not_have_events": crit.get("must_not_have_events", None),

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
# Convergência automática (Newton)
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


def run_convergence_newton_one_case(
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

        r = validate_newton(c, plotdir=None)
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
# Convergência automática (Schwarzschild)
# ============================================================

def schw_signature(case: Dict[str, Any]) -> str:
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


def check_convergence_schw(
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


def check_convergence_events_schw(
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
