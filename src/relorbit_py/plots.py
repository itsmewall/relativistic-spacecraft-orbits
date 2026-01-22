# src/relorbit_py/plots.py
from __future__ import annotations

import os
from typing import Any, List, Optional

import numpy as np
import matplotlib.pyplot as plt


def rel_drift(series: List[float]) -> float:
    a0 = float(series[0])
    arr = np.array(series, dtype=float)
    if a0 == 0.0:
        return float(np.max(np.abs(arr)))
    return float(np.max(np.abs((arr - a0) / a0)))


def savefig(path: str) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_newton(
    case_name: str,
    traj: Any,
    outdir: str,
    energy_rel_drift_max: float,
    h_rel_drift_max: float,
    eps_floor: float = 1e-16,
) -> None:
    """
    Plots Newton com escala log interpretável (sem "paredões" artificiais).

    eps_floor:
      Piso VISUAL para evitar log(0) e evitar gráficos enganosos. Não é precisão física.
      Escolhido para ficar bem abaixo dos thresholds típicos (1e-6, 1e-8), mas sem ser
      ridículo (1e-300).
    """
    t = np.array(traj.t, dtype=float)
    y = np.array(traj.y, dtype=float)  # Nx4
    x, yy = y[:, 0], y[:, 1]
    E = np.array(traj.energy, dtype=float)
    h = np.array(traj.h, dtype=float)

    # Órbita
    plt.figure()
    plt.plot(x, yy)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"Newton orbit: {case_name}")
    savefig(os.path.join(outdir, f"{case_name}_orbit.png"))

    # Invariantes
    plt.figure()
    plt.plot(t, E, label="Energy (specific)")
    plt.plot(t, h, label="Angular momentum h_z (specific)")
    plt.xlabel("t")
    plt.ylabel("value")
    plt.title(f"Invariants vs time: {case_name}")
    plt.legend(loc="upper right")
    savefig(os.path.join(outdir, f"{case_name}_invariants.png"))

    # Drift RELATIVO (coerente com os critérios do YAML)
    E0 = float(E[0]) if len(E) else 0.0
    h0 = float(h[0]) if len(h) else 0.0

    dE_rel = np.abs(E - E0) / max(abs(E0), 1.0)
    dh_rel = np.abs(h - h0) / max(abs(h0), 1.0)

    dE_rel_plot = np.maximum(dE_rel, eps_floor)
    dh_rel_plot = np.maximum(dh_rel, eps_floor)

    plt.figure()
    plt.semilogy(t, dE_rel_plot, label="|E - E0| / max(|E0|,1)")
    plt.semilogy(t, dh_rel_plot, label="|h - h0| / max(|h0|,1)")
    plt.axhline(
        float(energy_rel_drift_max),
        linestyle="--",
        label=f"crit energy_rel_drift_max={energy_rel_drift_max:.1e}",
    )
    plt.axhline(
        float(h_rel_drift_max),
        linestyle="--",
        label=f"crit h_rel_drift_max={h_rel_drift_max:.1e}",
    )
    plt.xlabel("t")
    plt.ylabel("relative drift (log)")
    plt.title(
        f"Drift (relative): {case_name} | rel_dE={rel_drift(list(E)):.2e} rel_dh={rel_drift(list(h)):.2e}"
    )
    plt.legend(loc="upper right")
    savefig(os.path.join(outdir, f"{case_name}_drift_rel.png"))


def plot_schw(
    case_name: str,
    traj: Any,
    outdir: str,
    constraint_abs_max: float,
    eps_floor: float = 1e-16,
) -> None:
    """
    Plots Schwarzschild com log interpretável + linhas do critério.

    eps_floor:
      Piso VISUAL para evitar log(0). Mantém o gráfico honesto e legível.
    """
    tau = np.array(traj.tau, dtype=float)
    r = np.array(traj.r, dtype=float)
    phi = np.array(traj.phi, dtype=float)

    eps = np.array(traj.epsilon, dtype=float) if hasattr(traj, "epsilon") else None
    if eps is None:
        raise AttributeError("Trajetória Schwarzschild não possui epsilon.")

    x = r * np.cos(phi)
    y = r * np.sin(phi)

    # Órbita
    plt.figure()
    plt.plot(x, y)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"Schwarzschild orbit (equatorial): {case_name}")
    savefig(os.path.join(outdir, f"{case_name}_orbit.png"))

    # r(tau)
    plt.figure()
    plt.plot(tau, r)
    plt.xlabel("tau")
    plt.ylabel("r")
    plt.title(f"r(tau): {case_name}")
    savefig(os.path.join(outdir, f"{case_name}_r_tau.png"))

    # epsilon(tau) assinado + linhas do critério
    plt.figure()
    plt.plot(tau, eps, label="epsilon(tau)")
    plt.axhline(+float(constraint_abs_max), linestyle="--", label=f"+crit {constraint_abs_max:.1e}")
    plt.axhline(-float(constraint_abs_max), linestyle="--", label=f"-crit {constraint_abs_max:.1e}")
    plt.xlabel("tau")
    plt.ylabel("epsilon")
    plt.title(f"Constraint (signed): {case_name}")
    plt.legend(loc="upper right")
    savefig(os.path.join(outdir, f"{case_name}_constraint.png"))

    # |epsilon| em log + linha do critério
    abs_eps = np.abs(eps)
    abs_eps_plot = np.maximum(abs_eps, eps_floor)

    plt.figure()
    plt.semilogy(tau, abs_eps_plot, label="|epsilon(tau)|")
    plt.axhline(float(constraint_abs_max), linestyle="--", label=f"crit constraint_abs_max={constraint_abs_max:.1e}")
    plt.xlabel("tau")
    plt.ylabel("|epsilon| (log)")
    plt.title(f"Constraint drift (log): {case_name}")
    plt.legend(loc="upper right")
    savefig(os.path.join(outdir, f"{case_name}_constraint_log.png"))

    # norm_u (se existir)
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
            plt.legend(loc="upper right")
            savefig(os.path.join(outdir, f"{case_name}_norm_u.png"))

            abs_nu_plot = np.maximum(np.abs(nu), eps_floor)
            plt.figure()
            plt.semilogy(tau, abs_nu_plot, label="|norm_u|")
            plt.xlabel("tau")
            plt.ylabel("|norm_u| (log)")
            plt.title(f"4-velocity normalization drift (log): {case_name}")
            plt.legend(loc="upper right")
            savefig(os.path.join(outdir, f"{case_name}_norm_u_log.png"))


def plot_convergence_newton(
    base_name: str,
    dts: List[float],
    errs: List[float],
    outdir: str,
    p_obs: Optional[float],
) -> None:
    plt.figure()
    plt.loglog(dts, errs, marker="o")
    plt.xlabel("dt")
    plt.ylabel("||y_dt(tf) - y_ref(tf)|| (proxy)")
    title = f"Newton convergence: {base_name}"
    if p_obs is not None:
        title += f" | p_obs≈{p_obs:.3f}"
    plt.title(title)
    savefig(os.path.join(outdir, f"{base_name}_convergence_loglog.png"))


def plot_convergence_overlay_newton(
    base_name: str,
    trajs: List[Any],
    labels: List[str],
    outdir: str,
) -> None:
    plt.figure()
    for tr, lab in zip(trajs, labels):
        y = np.array(tr.y, dtype=float)
        plt.plot(y[:, 0], y[:, 1], label=lab)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"Newton overlay (dt refinement): {base_name}")
    plt.legend(loc="upper right")
    savefig(os.path.join(outdir, f"{base_name}_overlay_orbit.png"))
