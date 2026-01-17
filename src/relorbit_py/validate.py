# src/relorbit_py/validate.py
from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

from . import engine_hello
from .simulate import load_cases_yaml, simulate_case


# ----------------------------
# Helpers
# ----------------------------

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
    """
    Compatibilidade:
      - novo: engine retorna TrajectoryNewton/TrajectorySchwarzschildEq diretamente
      - antigo: dict com 'traj'
      - antigo: objeto com .data['traj']
    """
    if res is None:
        raise TypeError("simulate_case retornou None (bug).")

    # Newton (pybind)
    if hasattr(res, "t") and hasattr(res, "y"):
        return res

    # Schwarzschild eq (pybind) - pode NÃO ter tau/t
    if hasattr(res, "r") and hasattr(res, "phi"):
        return res

    if isinstance(res, dict) and "traj" in res:
        return res["traj"]

    if hasattr(res, "data") and isinstance(getattr(res, "data"), dict) and "traj" in res.data:
        return res.data["traj"]

    raise TypeError(f"Formato inesperado do retorno de simulate_case: {type(res)}")


def _get_solver_dt_nsteps(case: Dict[str, Any]) -> Tuple[float, int]:
    # schema novo: case['solver'] = {dt, n_steps}
    if isinstance(case.get("solver"), dict):
        s = case["solver"]
        if "dt" in s:
            dt = float(s["dt"])
            n_steps = int(s.get("n_steps", 0) or 0)
            return dt, n_steps

    # legado
    if "dt" in case:
        dt = float(case["dt"])
        n_steps = int(case.get("n_steps", 0) or 0)
        return dt, n_steps

    raise KeyError(
        f"dt ausente no caso '{case.get('name','<sem-nome>')}'. "
        "Esperado em case.solver.dt (novo) ou case.dt (legado)."
    )


def _get_span(case: Dict[str, Any]) -> Tuple[float, float]:
    # schema novo
    if "span" in case:
        a, b = case["span"]
        return float(a), float(b)

    # legado
    if "t0" in case and "tf" in case:
        return float(case["t0"]), float(case["tf"])
    if "tau0" in case and "tauf" in case:
        return float(case["tau0"]), float(case["tauf"])

    raise KeyError(
        f"span ausente no caso '{case.get('name','<sem-nome>')}'. "
        "Esperado case.span=[a,b] (novo) ou (t0,tf)/(tau0,tauf) (legado)."
    )


def _get_time_axis_or_index(traj: Any, dt: float, t0: float = 0.0) -> np.ndarray:
    """
    Para Schwarzschild: algumas estruturas não expõem tau/t.
    Se existir traj.tau ou traj.t, usa. Se não, cria eixo artificial por índice.
    """
    if hasattr(traj, "tau"):
        return np.array(traj.tau, dtype=float)
    if hasattr(traj, "t"):
        return np.array(traj.t, dtype=float)

    # fallback: índice
    n = len(traj.r) if hasattr(traj, "r") else 0
    if n <= 0:
        return np.array([], dtype=float)
    return t0 + dt * np.arange(n, dtype=float)


# ----------------------------
# Plots
# ----------------------------

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

    dE = np.abs(E - E[0])
    dh = np.abs(h - h[0])
    dE = np.clip(dE, 1e-300, None)
    dh = np.clip(dh, 1e-300, None)

    plt.figure()
    plt.semilogy(t, dE, label="|E - E0|")
    plt.semilogy(t, dh, label="|h - h0|")
    plt.xlabel("t")
    plt.ylabel("absolute drift (log)")
    plt.title(f"Drift: {case_name} | rel_dE={_rel_drift(list(E)):.2e} rel_dh={_rel_drift(list(h)):.2e}")
    plt.legend()
    _savefig(os.path.join(outdir, f"{case_name}_drift.png"))


def _plot_schw(case_name: str, traj: Any, outdir: str, dt: float, tau0: float) -> None:
    tau = _get_time_axis_or_index(traj, dt=dt, t0=tau0)
    r = np.array(traj.r, dtype=float)
    phi = np.array(traj.phi, dtype=float)

    # epsilon/constraint
    if hasattr(traj, "epsilon"):
        eps = np.array(traj.epsilon, dtype=float)
    elif hasattr(traj, "eps"):
        eps = np.array(traj.eps, dtype=float)
    elif hasattr(traj, "constraint"):
        eps = np.array(traj.constraint, dtype=float)
    else:
        raise AttributeError("Trajetória Schwarzschild não possui epsilon/eps/constraint para plotar constraint.")

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
    plt.xlabel("tau (ou índice*dt)")
    plt.ylabel("r")
    plt.title(f"r(tau): {case_name}")
    _savefig(os.path.join(outdir, f"{case_name}_r_tau.png"))

    plt.figure()
    plt.plot(tau, eps, label="epsilon(tau)")
    plt.xlabel("tau (ou índice*dt)")
    plt.ylabel("epsilon")
    plt.title(f"Constraint (signed): {case_name}")
    plt.legend()
    _savefig(os.path.join(outdir, f"{case_name}_constraint.png"))

    abs_eps = np.abs(eps)
    abs_eps = np.clip(abs_eps, 1e-300, None)

    plt.figure()
    plt.semilogy(tau, abs_eps, label="|epsilon(tau)|")
    plt.xlabel("tau (ou índice*dt)")
    plt.ylabel("|epsilon| (log)")
    plt.title(f"Constraint drift (log): {case_name}")
    plt.legend()
    _savefig(os.path.join(outdir, f"{case_name}_constraint_log.png"))


# ----------------------------
# Validation
# ----------------------------

def _validate_newton(case: Dict[str, Any], plotdir: Optional[str] = None) -> Dict[str, Any]:
    res = simulate_case(case, "newton")
    traj = _unwrap_traj(res)

    E = list(traj.energy)
    h = list(traj.h)

    dE = _rel_drift(E)
    dh = _rel_drift(h)

    crit = case.get("criteria", {}) or {}
    dE_max = float(crit.get("energy_rel_drift_max", 2e-6))
    dh_max = float(crit.get("h_rel_drift_max", 2e-8))

    passed = (dE <= dE_max) and (dh <= dh_max)

    dt, n_steps = _get_solver_dt_nsteps(case)
    t0, tf = _get_span(case)
    params = case.get("params", {}) or {}
    mu = float(params.get("mu", case.get("mu", 1.0)))

    if plotdir is not None:
        _plot_newton(case["name"], traj, plotdir)

    return {
        "name": case["name"],
        "passed": bool(passed),
        "status": str(getattr(traj, "status", "")),
        "message": getattr(traj, "message", ""),
        "model": "newton",
        "mu": mu,
        "t0": t0,
        "tf": tf,
        "dt": dt,
        "n_steps": int(n_steps),
        "energy_rel_drift": float(dE),
        "h_rel_drift": float(dh),
        "criteria": {
            "energy_rel_drift_max": dE_max,
            "h_rel_drift_max": dh_max,
        },
    }


def _validate_schw(case: Dict[str, Any], plotdir: Optional[str] = None) -> Dict[str, Any]:
    res = simulate_case(case, "schwarzschild")
    traj = _unwrap_traj(res)

    if hasattr(traj, "epsilon"):
        eps = np.array(traj.epsilon, dtype=float)
    elif hasattr(traj, "eps"):
        eps = np.array(traj.eps, dtype=float)
    elif hasattr(traj, "constraint"):
        eps = np.array(traj.constraint, dtype=float)
    else:
        raise AttributeError("Trajetória Schwarzschild não possui epsilon/eps/constraint.")

    eps_max = float(np.max(np.abs(eps)))

    crit = case.get("criteria", {}) or {}
    eps_max_allowed = float(crit.get("constraint_abs_max", 1e-10))

    expected_status = str(crit.get("status", "BOUND"))
    status_str = str(getattr(traj, "status", ""))
    status_ok = status_str.endswith(expected_status)

    r_max_dev = None
    r_ok = True
    if crit.get("r_max_dev", None) is not None:
        r = np.array(traj.r, dtype=float)
        r0 = float(case["state0"][0])
        r_max_dev = float(np.max(np.abs(r - r0)))
        r_ok = r_max_dev <= float(crit["r_max_dev"])

    passed = (eps_max <= eps_max_allowed) and status_ok and r_ok

    dt, n_steps = _get_solver_dt_nsteps(case)
    tau0, tauf = _get_span(case)
    params = case.get("params", {}) or {}
    M = float(params.get("M", case.get("M", 1.0)))
    Epar = params.get("E", case.get("E", None))
    Lpar = params.get("L", case.get("L", None))

    if plotdir is not None:
        _plot_schw(case["name"], traj, plotdir, dt=dt, tau0=tau0)

    return {
        "name": case["name"],
        "passed": bool(passed),
        "status": status_str,
        "message": getattr(traj, "message", ""),
        "model": "schwarzschild",
        "M": M,
        "E": Epar,
        "L": Lpar,
        "tau0": tau0,
        "tauf": tauf,
        "dt": dt,
        "n_steps": int(n_steps),
        "constraint_abs_max": float(eps_max),
        "r_max_dev": r_max_dev,
        "criteria": {
            "constraint_abs_max": eps_max_allowed,
            "r_max_dev": crit.get("r_max_dev", None),
            "status": expected_status,
        },
    }


# ----------------------------
# Main
# ----------------------------

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

    # Newton
    newton_cases = cfg["suites"]["newton"]["cases"]
    newton_results: List[Dict[str, Any]] = []
    ok_newton = True
    for c in newton_cases:
        r = _validate_newton(c, plotdir if args.plots else None)
        newton_results.append(r)
        ok_newton = ok_newton and r["passed"]

    print(f"Newton suite: ok={ok_newton} cases={len(newton_cases)}")
    for r in newton_results:
        tag = "PASS" if r["passed"] else "FAIL"
        print(f"[{tag}] {r['name']} | dt={r['dt']:.1e} | dE={r['energy_rel_drift']:.3e} | dh={r['h_rel_drift']:.3e}")

    report["suites"].append(
        {"suite": "newton", "ok": ok_newton, "n_cases": len(newton_cases), "results": newton_results}
    )

    # Schwarzschild
    schw_cases = cfg["suites"]["schwarzschild"]["cases"]
    schw_results: List[Dict[str, Any]] = []
    ok_schw = True
    for c in schw_cases:
        r = _validate_schw(c, plotdir if args.plots else None)
        schw_results.append(r)
        ok_schw = ok_schw and r["passed"]

    print(f"Schwarzschild suite: ok={ok_schw} cases={len(schw_cases)}")
    for r in schw_results:
        tag = "PASS" if r["passed"] else "FAIL"
        print(f"[{tag}] {r['name']} | dt={r['dt']:.1e} | eps_max={r['constraint_abs_max']:.3e} | status={r['status']}")

    report["suites"].append(
        {"suite": "schwarzschild", "ok": ok_schw, "n_cases": len(schw_cases), "results": schw_results}
    )

    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    if args.plots:
        print(f"Plots em: {plotdir}")
    print(f"Relatório em: {os.path.join(outdir, 'report.json')}")


if __name__ == "__main__":
    main()
