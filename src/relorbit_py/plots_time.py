# src/relorbit_py/plots_time.py
from __future__ import annotations

import os
from typing import Optional

import numpy as np


def plot_schw_time(
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
