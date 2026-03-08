# src/relorbit_py/validate_helpers.py
from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ============================================================
# Unwrap/compat helpers
# ============================================================

def unwrap_traj(res: Any) -> Any:
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


def get_solver_dt_nsteps(case: Dict[str, Any]) -> Tuple[float, int]:
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


def set_solver_dt_nsteps(case: Dict[str, Any], dt: float, n_steps: int) -> None:
    """Atualiza (in-place) dt e n_steps, suportando formato novo (case.solver) e legado."""
    if isinstance(case.get("solver"), dict):
        case["solver"]["dt"] = float(dt)
        case["solver"]["n_steps"] = int(n_steps)
        return
    # legado
    case["dt"] = float(dt)
    case["n_steps"] = int(n_steps)


def get_span(case: Dict[str, Any]) -> Tuple[float, float]:
    if "span" in case:
        a, b = case["span"]
        return float(a), float(b)
    if "t0" in case and "tf" in case:
        return float(case["t0"]), float(case["tf"])
    if "tau0" in case and "tauf" in case:
        return float(case["tau0"]), float(case["tauf"])
    raise KeyError(f"span ausente no caso '{case.get('name','<sem-nome>')}'. Esperado case.span=[a,b].")


def case_pr0(case: Dict[str, Any]) -> float:
    if "pr0" in case:
        return float(case["pr0"])
    params = case.get("params", {}) or {}
    if "pr0" in params:
        return float(params["pr0"])
    return 0.0


# ============================================================
# Formatting
# ============================================================

def fmt_e(x: Any, width: int = 12, prec: int = 3) -> str:
    if x is None:
        return " " * (width - 4) + "None"
    try:
        return f"{float(x):>{width}.{prec}e}"
    except Exception:
        s = str(x)
        if len(s) > width:
            return (s[: width - 3] + "...").rjust(width)
        return s.rjust(width)


def fmt_f(x: Any, width: int = 10, prec: int = 6) -> str:
    if x is None:
        return " " * (width - 4) + "None"
    try:
        return f"{float(x):>{width}.{prec}f}"
    except Exception:
        s = str(x)
        if len(s) > width:
            return (s[: width - 3] + "...").rjust(width)
        return s.rjust(width)


def short_msg(msg: str, maxlen: int = 76) -> str:
    if not msg:
        return ""
    msg = msg.replace("\n", " ").strip()
    if len(msg) <= maxlen:
        return msg
    return msg[: maxlen - 3] + "..."


# ============================================================
# Math utilities
# ============================================================

def status_endswith(status_str: str, expected_tail: str) -> bool:
    return str(status_str).endswith(str(expected_tail))


def estimate_order_from_three(e_dt: float, e_dt2: float) -> Optional[float]:
    if not (np.isfinite(e_dt) and np.isfinite(e_dt2)):
        return None
    if e_dt <= 0.0 or e_dt2 <= 0.0:
        return None
    return float(math.log(e_dt / e_dt2, 2.0))


def is_finite_array(x: np.ndarray) -> bool:
    return bool(np.all(np.isfinite(np.asarray(x, dtype=float))))


def is_monotone_increasing(x: np.ndarray, tol: float = 0.0) -> bool:
    x = np.asarray(x, dtype=float).reshape(-1)
    if x.size < 2:
        return True
    dx = np.diff(x)
    return bool(np.all(dx >= -float(tol)))


def finite_diff_first_derivative(y: np.ndarray, x: np.ndarray) -> np.ndarray:
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


def nan_abs_max(x: Any) -> Optional[float]:
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


# ============================================================
# Events helpers
# ============================================================

def extract_events(traj: Any) -> List[Dict[str, Any]]:
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


def events_compact_str(events: List[Dict[str, Any]], max_items: int = 6) -> str:
    if not events:
        return ""
    chunks = []
    for e in events[:max_items]:
        chunks.append(f"{e['kind']}@{float(e['tau']):.6g}")
    if len(events) > max_items:
        chunks.append(f"+{len(events) - max_items}")
    return "; ".join(chunks)


def events_to_kind_map(events: List[Dict[str, Any]]) -> Dict[str, List[float]]:
    mp: Dict[str, List[float]] = defaultdict(list)
    for e in events:
        mp[str(e.get("kind", ""))].append(float(e.get("tau", 0.0)))
    return mp


def check_event_criteria(events: List[Dict[str, Any]], crit: Dict[str, Any]) -> Tuple[bool, str]:
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
