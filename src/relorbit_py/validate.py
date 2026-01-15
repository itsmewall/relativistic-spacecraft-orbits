# src/relorbit_py/validate.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from . import engine_hello, get_engine
from .cases import load_cases


def _rel_drift(series: np.ndarray) -> float:
    # drift relativo robusto: (max-min) / max(1, |valor inicial|)
    s0 = float(series[0])
    denom = max(1.0, abs(s0))
    return float((series.max() - series.min()) / denom)


def _write_json(obj: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def run_newton_cases(cases_path: str = "validation_cases.yaml") -> Dict[str, Any]:
    eng = get_engine()
    cases = [c for c in load_cases(cases_path) if c.model == "newton"]

    results: List[Dict[str, Any]] = []
    ok_all = True

    for c in cases:
        cfg = eng.SolverCfg()
        cfg.dt = float(c.solver.dt)
        cfg.n_steps = int(c.solver.n_steps)

        mu = float(c.params["mu"])
        t0, tf = c.span
        state0 = list(c.state0)

        traj = eng.simulate_newton_rk4(mu=mu, state0=state0, t0=t0, tf=tf, cfg=cfg)

        energy = np.asarray(traj.energy, dtype=float)
        h = np.asarray(traj.h, dtype=float)

        e_drift = _rel_drift(energy)
        h_drift = _rel_drift(h)

        exp = dict(c.expected)
        e_max = float(exp.get("energy_rel_drift_max", 1e-6))
        h_max = float(exp.get("h_rel_drift_max", 1e-8))

        status_is_error = (traj.status == eng.OrbitStatus.ERROR)
        passed = (not status_is_error) and (e_drift <= e_max) and (h_drift <= h_max)
        ok_all = ok_all and passed

        results.append(
            {
                "name": c.name,
                "passed": passed,
                "status": str(traj.status),
                "message": getattr(traj, "message", ""),
                "model": c.model,
                "mu": mu,
                "t0": t0,
                "tf": tf,
                "dt": float(c.solver.dt),
                "n_steps": int(c.solver.n_steps),
                "energy_rel_drift": e_drift,
                "h_rel_drift": h_drift,
                "criteria": {
                    "energy_rel_drift_max": e_max,
                    "h_rel_drift_max": h_max,
                },
            }
        )

    return {"suite": "newton", "ok": ok_all, "n_cases": len(results), "results": results}


def main() -> None:
    print(engine_hello())

    report = run_newton_cases("validation_cases.yaml")

    print(f"Newton suite: ok={report['ok']} cases={report['n_cases']}")
    for r in report["results"]:
        tag = "PASS" if r["passed"] else "FAIL"
        print(
            f"[{tag}] {r['name']} | dt={r['dt']:.1e} | "
            f"dE={r['energy_rel_drift']:.3e} (<= {r['criteria']['energy_rel_drift_max']:.1e}) | "
            f"dh={r['h_rel_drift']:.3e} (<= {r['criteria']['h_rel_drift_max']:.1e})"
        )

    _write_json(report, Path("docs/reports/newton_validation.json"))


if __name__ == "__main__":
    main()
