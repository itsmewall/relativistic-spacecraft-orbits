# src/relorbit_py/plots.py
from __future__ import annotations

import os
from pathlib import Path
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


def _sibling_time_dir_from_plots_dir(outdir_plots: str) -> str:
    """
    outdir_plots tipicamente: out/plots
    retorna: out/time_plots
    """
    p = Path(outdir_plots)
    parent = p.parent  # out/
    return str(parent / "time_plots")


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
      Piso VISUAL para evitar log(0) e evitar gráficos enganosos.
      Não é precisão física; é só para legibilidade.
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


def _plot_schw_time_pack(
    case_name: str,
    traj: Any,
    plots_outdir: str,
    eps_floor: float = 1e-16,
) -> None:
    """
    Plots de TEMPO (dilatação temporal) para Schwarzschild.
    Salva em out/time_plots/ (separado de out/plots/).

    Conteúdo:
      - t(τ)
      - dt/dτ (FD) e dt/dτ (teórico = E/(1-2M/r)), quando possível
      - (t - τ) como “descolamento” (intuitivo p/ leigo)
      - dt/dτ em semilogy (útil perto do horizonte)
    """
    if not hasattr(traj, "tcoord"):
        return

    tau = np.array(traj.tau, dtype=float)
    t = np.array(traj.tcoord, dtype=float)
    r = np.array(traj.r, dtype=float)

    if tau.size == 0 or t.size != tau.size or r.size != tau.size:
        return

    time_dir = _sibling_time_dir_from_plots_dir(plots_outdir)
    os.makedirs(time_dir, exist_ok=True)

    # t(tau)
    plt.figure()
    plt.plot(tau, t)
    plt.xlabel("tau (proper time)")
    plt.ylabel("t (Schwarzschild coordinate time)")
    plt.title(f"Coordinate time vs proper time: {case_name}")
    savefig(os.path.join(time_dir, f"{case_name}_tcoord_vs_tau.png"))

    # t - tau (intuitivo: “quanto o relógio coordenado esticou”)
    plt.figure()
    plt.plot(tau, (t - tau))
    plt.xlabel("tau")
    plt.ylabel("t - tau")
    plt.title(f"Time offset (t - tau): {case_name}")
    savefig(os.path.join(time_dir, f"{case_name}_t_minus_tau.png"))

    # dt/dtau via FD (se existir) + teórico (se E e M existirem)
    dt_dtau_fd = None
    if hasattr(traj, "ut_fd"):
        u = np.array(traj.ut_fd, dtype=float)
        if u.size == tau.size:
            dt_dtau_fd = u

    dt_dtau_theory = None
    if hasattr(traj, "E") and hasattr(traj, "M"):
        E = float(getattr(traj, "E"))
        M = float(getattr(traj, "M"))
        A = 1.0 - 2.0 * M / r
        A_safe = np.where(np.abs(A) < 1e-300, np.sign(A) * 1e-300, A)
        dt_dtau_theory = E / A_safe

    if dt_dtau_fd is not None or dt_dtau_theory is not None:
        plt.figure()
        if dt_dtau_fd is not None:
            plt.plot(tau, dt_dtau_fd, label="dt/dtau (FD)")
        if dt_dtau_theory is not None:
            plt.plot(tau, dt_dtau_theory, label="dt/dtau (theory: E/A)")
        plt.xlabel("tau")
        plt.ylabel("dt/dtau")
        plt.title(f"Time dilation factor dt/dtau: {case_name}")
        plt.legend(loc="upper right")
        savefig(os.path.join(time_dir, f"{case_name}_dt_dtau.png"))

        # versão log (para mergulho perto do horizonte)
        plt.figure()
        if dt_dtau_fd is not None:
            plt.semilogy(tau, np.maximum(np.abs(dt_dtau_fd), eps_floor), label="|dt/dtau| (FD)")
        if dt_dtau_theory is not None:
            plt.semilogy(tau, np.maximum(np.abs(dt_dtau_theory), eps_floor), label="|dt/dtau| (theory)")
        plt.xlabel("tau")
        plt.ylabel("|dt/dtau| (log)")
        plt.title(f"Time dilation factor |dt/dtau| (log): {case_name}")
        plt.legend(loc="upper right")
        savefig(os.path.join(time_dir, f"{case_name}_dt_dtau_log.png"))


def plot_schw(
    case_name: str,
    traj: Any,
    outdir: str,
    constraint_abs_max: float,
    eps_floor: float = 1e-16,
) -> None:
    """
    Plots Schwarzschild (out/plots) + pacote de tempo (out/time_plots) se tcoord existir.
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

    # =========================
    # NOVO: pacote de tempo
    # =========================
    _plot_schw_time_pack(case_name, traj, outdir, eps_floor=eps_floor)


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
