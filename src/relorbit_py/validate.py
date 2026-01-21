# src/relorbit_py/validate.py
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

from . import engine_hello
from .simulate import load_cases_yaml, simulate_case


# ============================================================
# Helpers gerais
# ============================================================

def _rel_drift(series: List[float]) -> float:
    a0 = float(series[0])
    arr = np.array(series, dtype=float)
    if a0 == 0.0:
        return float(np.max(np.abs(arr)))
    return float(np.max(np.abs((arr - a0) / a0)))


def _savefig(path: str) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


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
    # status_str costuma vir como "OrbitStatus.BOUND"
    # expected_tail pode vir como "BOUND"
    return str(status_str).endswith(str(expected_tail))


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
    # convenção: e<0 bound, e>=0 unbound
    return "BOUND" if e0 < 0.0 else "UNBOUND"


# ============================================================
# Eventos (Schwarzschild)
# ============================================================

def _extract_events(traj: Any) -> List[Dict[str, Any]]:
    """Extrai eventos do objeto traj (se existirem) em formato estável para report."""
    if not hasattr(traj, "event_kind"):
        return []

    kind = list(getattr(traj, "event_kind"))
    tau = list(getattr(traj, "event_tau", []))
    tcoord = list(getattr(traj, "event_tcoord", []))
    r = list(getattr(traj, "event_r", []))
    phi = list(getattr(traj, "event_phi", []))
    pr = list(getattr(traj, "event_pr", []))

    n = min(len(kind), len(tau), len(tcoord), len(r), len(phi), len(pr))
    events: List[Dict[str, Any]] = []
    for i in range(n):
        events.append({
            "kind": str(kind[i]),
            "tau": float(tau[i]),
            "tcoord": float(tcoord[i]),
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
    """
    Critérios opcionais:
      - min_events: int
      - must_have_events: ["capture","periapse",...]
      - must_not_have_events: ["horizon",...]
    """
    kinds = [str(e.get("kind", "")) for e in events]
    kinds_set = set(kinds)

    min_events = int(crit.get("min_events", 0) or 0)
    if min_events > 0 and len(events) < min_events:
        return False, f"events: expected >= {min_events}, got {len(events)}"

    must_have = crit.get("must_have_events", None)
    if must_have is not None:
        if isinstance(must_have, str):
            must_have_list = [must_have]
        else:
            must_have_list = list(must_have)
        missing = [k for k in must_have_list if str(k) not in kinds_set]
        if missing:
            return False, f"events: missing required kinds {missing}"

    must_not = crit.get("must_not_have_events", None)
    if must_not is not None:
        if isinstance(must_not, str):
            must_not_list = [must_not]
        else:
            must_not_list = list(must_not)
        present = [k for k in must_not_list if str(k) in kinds_set]
        if present:
            return False, f"events: forbidden kinds present {present}"

    return True, ""


# ============================================================
# Plotting
# ============================================================

def _plot_newton(case_name: str, traj: Any, outdir: str) -> None:
    t = np.array(traj.t, dtype=float)
    y = np.array(traj.y, dtype=float)  # Nx4
    x, yy = y[:, 0], y[:, 1]
    E = np.array(traj.energy, dtype=float)
    h = np.array(traj.h, dtype=float)

    plt.figure()
    plt.plot(x, yy)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"Newton orbit: {case_name}")
    _savefig(os.path.join(outdir, f"{case_name}_orbit.png"))

    plt.figure()
    plt.plot(t, E, label="Energy (specific)")
    plt.plot(t, h, label="Angular momentum h_z (specific)")
    plt.xlabel("t")
    plt.ylabel("value")
    plt.title(f"Invariants vs time: {case_name}")
    plt.legend()
    _savefig(os.path.join(outdir, f"{case_name}_invariants.png"))

    dE = np.clip(np.abs(E - E[0]), 1e-300, None)
    dh = np.clip(np.abs(h - h[0]), 1e-300, None)

    plt.figure()
    plt.semilogy(t, dE, label="|E - E0|")
    plt.semilogy(t, dh, label="|h - h0|")
    plt.xlabel("t")
    plt.ylabel("absolute drift (log)")
    plt.title(f"Drift: {case_name} | rel_dE={_rel_drift(list(E)):.2e} rel_dh={_rel_drift(list(h)):.2e}")
    plt.legend()
    _savefig(os.path.join(outdir, f"{case_name}_drift.png"))


def _plot_schw(case_name: str, traj: Any, outdir: str) -> None:
    tau = np.array(traj.tau, dtype=float)
    r = np.array(traj.r, dtype=float)
    phi = np.array(traj.phi, dtype=float)

    eps = np.array(traj.epsilon, dtype=float) if hasattr(traj, "epsilon") else None
    if eps is None:
        raise AttributeError("Trajetória Schwarzschild não possui epsilon.")

    x = r * np.cos(phi)
    y = r * np.sin(phi)

    plt.figure()
    plt.plot(x, y)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"Schwarzschild orbit (equatorial): {case_name}")
    _savefig(os.path.join(outdir, f"{case_name}_orbit.png"))

    plt.figure()
    plt.plot(tau, r)
    plt.xlabel("tau")
    plt.ylabel("r")
    plt.title(f"r(tau): {case_name}")
    _savefig(os.path.join(outdir, f"{case_name}_r_tau.png"))

    plt.figure()
    plt.plot(tau, eps, label="epsilon(tau)")
    plt.xlabel("tau")
    plt.ylabel("epsilon")
    plt.title(f"Constraint (signed): {case_name}")
    plt.legend()
    _savefig(os.path.join(outdir, f"{case_name}_constraint.png"))

    abs_eps = np.clip(np.abs(eps), 1e-300, None)
    plt.figure()
    plt.semilogy(tau, abs_eps, label="|epsilon(tau)|")
    plt.xlabel("tau")
    plt.ylabel("|epsilon| (log)")
    plt.title(f"Constraint drift (log): {case_name}")
    plt.legend()
    _savefig(os.path.join(outdir, f"{case_name}_constraint_log.png"))

    if hasattr(traj, "norm_u"):
        nu = np.array(traj.norm_u, dtype=float)
        if len(nu) != len(tau):
            raise RuntimeError(
                f"norm_u shape mismatch no caso '{case_name}': "
                f"len(norm_u)={len(nu)} vs len(tau)={len(tau)}"
            )
        if len(nu) > 0:
            plt.figure()
            plt.plot(tau, nu, label="norm_u = g(u,u)+1")
            plt.xlabel("tau")
            plt.ylabel("norm_u")
            plt.title(f"4-velocity normalization drift: {case_name}")
            plt.legend()
            _savefig(os.path.join(outdir, f"{case_name}_norm_u.png"))

            plt.figure()
            plt.semilogy(tau, np.clip(np.abs(nu), 1e-300, None), label="|norm_u|")
            plt.xlabel("tau")
            plt.ylabel("|norm_u| (log)")
            plt.title(f"4-velocity normalization drift (log): {case_name}")
            plt.legend()
            _savefig(os.path.join(outdir, f"{case_name}_norm_u_log.png"))


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

    # teoria (por energia inicial do state0)
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

    # Checagem opcional de status
    status_str = str(getattr(traj, "status", ""))
    expected_status = crit.get("status", None)  # se não existir, não exige
    status_ok = True
    status_reason = ""
    if expected_status is not None:
        status_ok = _status_endswith(status_str, str(expected_status))
        if not status_ok:
            status_reason = f"status mismatch (got={status_str}, expected=*{expected_status})"
        # também pode exigir que bata com teoria (se theory_status disponível)
        if crit.get("status_must_match_theory", False) and theory_status is not None:
            if not _status_endswith(status_str, theory_status):
                status_ok = False
                status_reason = f"status != theory (got={status_str}, theory=*{theory_status})"

    passed = (dE <= dE_max) and (dh <= dh_max) and status_ok

    if plotdir is not None:
        _plot_newton(case["name"], traj, plotdir)

    msg = getattr(traj, "message", "") or ""
    if (not passed) and status_reason:
        msg = (msg + " | " + status_reason).strip(" |")

    return {
        "name": case["name"],
        "passed": bool(passed),
        "status": status_str,
        "message": msg,
        "model": "newton",
        "mu": mu,
        "t0": t0,
        "tf": tf,
        "dt": dt,
        "n_steps": int(n_steps),
        "energy0": energy0,
        "status_theory": theory_status,
        "energy_rel_drift": float(dE),
        "h_rel_drift": float(dh),
        "criteria": {
            "energy_rel_drift_max": dE_max,
            "h_rel_drift_max": dh_max,
            "status": expected_status,
            "status_must_match_theory": bool(crit.get("status_must_match_theory", False)),
        },
    }


def _validate_schw(case: Dict[str, Any], plotdir: Optional[str] = None) -> Dict[str, Any]:
    traj = _unwrap_traj(simulate_case(case, "schwarzschild"))

    tau = np.array(traj.tau, dtype=float)
    r = np.array(traj.r, dtype=float)

    r_min = float(np.min(r)) if r.size else None
    r_end = float(r[-1]) if r.size else None

    eps = np.array(traj.epsilon, dtype=float)
    eps_max = float(np.max(np.abs(eps))) if eps.size else None

    norm_u_max = None
    if hasattr(traj, "norm_u"):
        nu = np.array(traj.norm_u, dtype=float)
        if nu.size != tau.size:
            raise RuntimeError(
                f"norm_u shape mismatch no caso '{case.get('name')}': "
                f"len(norm_u)={nu.size} vs len(tau)={tau.size}"
            )
        if nu.size > 0:
            norm_u_max = float(np.nanmax(np.abs(nu)))

    crit = case.get("criteria", {}) or {}
    eps_max_allowed = float(crit.get("constraint_abs_max", 1e-10))
    expected_status = str(crit.get("status", "BOUND"))

    status_str = str(getattr(traj, "status", ""))
    status_ok = _status_endswith(status_str, expected_status)

    events = _extract_events(traj)
    events_compact = _events_compact_str(events)

    events_ok, events_reason = _check_event_criteria(events, crit)

    passed = (
        (eps_max is not None and eps_max <= eps_max_allowed)
        and status_ok
        and events_ok
    )

    dt, n_steps = _get_solver_dt_nsteps(case)
    tau0, tauf = _get_span(case)

    params = case.get("params", {}) or {}
    M = float(params.get("M", case.get("M", 1.0)))
    Epar = float(params.get("E", case.get("E")))
    Lpar = float(params.get("L", case.get("L")))
    pr0 = _case_pr0(case)

    if plotdir is not None:
        _plot_schw(case["name"], traj, plotdir)

    msg = getattr(traj, "message", "") or ""
    if (not passed) and (not events_ok) and events_reason:
        msg = (msg + " | " + events_reason).strip(" |")

    return {
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
        "dt": dt,
        "n_steps": int(n_steps),
        "r_min": r_min,
        "r_end": r_end,
        "constraint_abs_max": float(eps_max) if eps_max is not None else None,
        "norm_u_abs_max": norm_u_max,
        "events": events,
        "events_compact": events_compact,
        "criteria": {
            "constraint_abs_max": eps_max_allowed,
            "status": expected_status,
            "min_events": int(crit.get("min_events", 0) or 0),
            "must_have_events": crit.get("must_have_events", None),
            "must_not_have_events": crit.get("must_not_have_events", None),
        },
    }


# ============================================================
# Convergência automática (Schwarzschild)
# ============================================================

def _schw_signature(case: Dict[str, Any]) -> str:
    """
    Assinatura física (o que deve permanecer igual entre dt diferentes):
      (M,E,L,r0,phi0,pr0,span,capture_r,capture_eps)
    Exclui solver e criteria.
    """
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
        nus_raw = [g.get("norm_u_abs_max", None) for g in grp_sorted]
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
    ap.add_argument("--plots", action="store_true", help="Generate plots in <out>/plots")
    ap.add_argument("--out", default="out", help="Output directory")
    args = ap.parse_args()

    print(engine_hello())

    cfg = load_cases_yaml(args.cases)
    outdir = args.out
    plotdir = os.path.join(outdir, "plots")
    if args.plots:
        os.makedirs(plotdir, exist_ok=True)

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

    report["suites"].append({
        "suite": "newton",
        "ok": ok_newton,
        "n_cases": len(newton_cases),
        "results": newton_results
    })

    # ----------------------------
    # Schwarzschild
    # ----------------------------
    schw_cases = cfg["suites"]["schwarzschild"]["cases"]
    schw_results: List[Dict[str, Any]] = []
    ok_schw_cases = True

    for c in schw_cases:
        rr = _validate_schw(c, plotdir if args.plots else None)
        rr["_sig"] = _schw_signature(c)
        schw_results.append(rr)
        ok_schw_cases = ok_schw_cases and rr["passed"]

    conv = _check_convergence_schw(schw_results, abs_tol=1e-9, rel_tol=0.25)
    conv_ok = all(bool(x["passed"]) for x in conv) if conv else False

    events_conv = _check_convergence_events_schw(schw_results, abs_tol_factor=2.0, rel_tol=0.0)
    events_conv_ok = all(bool(x["passed"]) for x in events_conv) if events_conv else False

    ok_schw_total = bool(ok_schw_cases and conv_ok and events_conv_ok)

    _print_header(
        f"Schwarzschild suite: ok={ok_schw_total} "
        f"(cases_ok={ok_schw_cases}, conv_ok={conv_ok}, events_conv_ok={events_conv_ok}) cases={len(schw_cases)}"
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
            _fmt_e(r.get("norm_u_abs_max"), width=12),
            str(r.get("status", "")),
            r.get("events_compact", "") or "",
            _short_msg(str(r.get("message", ""))),
        ])
    _print_table(
        s_rows,
        headers=["ok", "case", "dt", "r_min", "r_end", "eps_max", "norm_u", "status", "events", "msg"]
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

    _print_header("Schwarzschild convergence: norm_u_abs_max should not increase when dt decreases (with tolerance)")
    if not conv:
        print("No comparable groups found. Need >=2 cases with same physics and different dt.")
    else:
        c_rows: List[List[str]] = []
        for g in conv:
            tag = "PASS" if g["passed"] else ("INCONCLUSIVE" if g.get("inconclusive") else "FAIL")
            dts = ", ".join([f"{dt:.2e}" for dt in g["dts"]])
            nus = ", ".join(["None" if v is None else f"{float(v):.3e}" for v in g["norm_u_abs_max"]])
            c_rows.append([tag, dts, nus, ", ".join(g["cases"])])
        _print_table(c_rows, headers=["ok", "dt (big->small)", "norm_u_abs_max", "cases"])

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
            tag = "PASS" if g["passed"] else ("SKIP" if g.get("skipped") else ("INCONCLUSIVE" if g.get("inconclusive") else "FAIL"))
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
        "ok_convergence": conv_ok,
        "ok_events_convergence": events_conv_ok,
        "n_cases": len(schw_cases),
        "results": schw_results,
        "convergence": conv,
        "events_convergence": events_conv,
    })

    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    if args.plots:
        print(f"\nPlots em: {plotdir}")
    print(f"Relatório em: {os.path.join(outdir, 'report.json')}")


if __name__ == "__main__":
    main()
