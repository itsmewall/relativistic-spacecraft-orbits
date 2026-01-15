# scripts/plot_newton_suite.py
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from relorbit_py.cases import load_cases
from relorbit_py.simulate import simulate_case


def rel_drift(series: np.ndarray) -> float:
    s0 = float(series[0])
    denom = max(1.0, abs(s0))
    return float((series.max() - series.min()) / denom)


def main() -> None:
    cases = [c for c in load_cases("validation_cases.yaml") if c.model == "newton"]

    out_dir = Path("docs/figures/newton")
    out_dir.mkdir(parents=True, exist_ok=True)

    eps = np.finfo(float).tiny  # menor float positivo normalizado (robusto para log)

    for c in cases:
        res = simulate_case(c)

        t = res.t
        x = res.y[:, 0]
        y = res.y[:, 1]
        E = res.energy
        H = res.h

        # 1) Órbita x-y
        plt.figure()
        plt.plot(x, y)
        plt.gca().set_aspect("equal", adjustable="box")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title(f"Newton orbit: {c.name}")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(out_dir / f"{c.name}_orbit.png", dpi=200)
        plt.close()

        # 2) Invariantes (delta) vs tempo: ΔE e Δh
        dE_signed = E - E[0]
        dH_signed = H - H[0]

        plt.figure()
        plt.plot(t, dE_signed, label="ΔE = E - E0")
        plt.plot(t, dH_signed, label="Δh = h - h0")
        plt.xlabel("t")
        plt.ylabel("delta")
        plt.title(f"Invariants (delta) vs time: {c.name}")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"{c.name}_invariants_delta.png", dpi=200)
        plt.close()

        # 3) Drift absoluto em escala log (sem chão artificial)
        dE_abs = np.abs(dE_signed)
        dH_abs = np.abs(dH_signed)

        plt.figure()
        plt.semilogy(t, np.maximum(dE_abs, eps), label="|E - E0|")
        plt.semilogy(t, np.maximum(dH_abs, eps), label="|h - h0|")
        plt.xlabel("t")
        plt.ylabel("absolute drift (log)")
        plt.title(
            f"Drift: {c.name} | rel_dE={rel_drift(E):.2e} rel_dh={rel_drift(H):.2e}"
        )
        plt.grid(True, which="both")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"{c.name}_drift.png", dpi=200)
        plt.close()

    print(f"Figuras geradas em: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
