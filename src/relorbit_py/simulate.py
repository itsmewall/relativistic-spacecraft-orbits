from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np

from . import get_engine
from .cases import ValidationCase


@dataclass(frozen=True)
class SimResult:
    t: np.ndarray           # (N,)
    y: np.ndarray           # (N,4) -> [x,y,vx,vy]
    energy: np.ndarray      # (N,)
    h: np.ndarray           # (N,)
    status: Any
    message: str
    meta: Dict[str, Any]


def simulate_case(case: ValidationCase) -> SimResult:
    if case.model != "newton":
        raise ValueError(f"simulate_case ainda só suporta model='newton'. Recebido: {case.model}")

    eng = get_engine()

    cfg = eng.SolverCfg()
    cfg.dt = float(case.solver.dt)
    cfg.n_steps = int(case.solver.n_steps)

    mu = float(case.params["mu"])
    t0, tf = case.span
    state0 = list(case.state0)

    traj = eng.simulate_newton_rk4(mu=mu, state0=state0, t0=t0, tf=tf, cfg=cfg)

    t = np.asarray(traj.t, dtype=float)
    y = np.asarray(traj.y, dtype=float)
    energy = np.asarray(traj.energy, dtype=float)
    h = np.asarray(traj.h, dtype=float)

    meta = {
        "name": case.name,
        "model": case.model,
        "mu": mu,
        "t0": t0,
        "tf": tf,
        "dt": float(case.solver.dt),
        "n_steps": int(case.solver.n_steps),
        "expected": dict(case.expected),
    }

    return SimResult(
        t=t,
        y=y,
        energy=energy,
        h=h,
        status=traj.status,
        message=getattr(traj, "message", ""),
        meta=meta,
    )
