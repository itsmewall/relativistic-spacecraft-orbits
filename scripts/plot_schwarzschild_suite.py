# scripts/plot_schwarzschild_suite.py
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from relorbit_py.cases import load_cases
from relorbit_py.simulate import simulate_case


def main() -> None:
    cases = [c for c in load_cases("validation_cases.yaml") if c.model == "schwarzschild"]

    out_dir = Path("docs/figures/schwarzschild")
    out_dir.mkdir(parents=True, exist_ok=True)

    eps = np.finfo(float).tiny

    for c in cases:
        res = simulate_case(c)

        tau = np.asarray(res.t, dtype=float)
        y = np.asarray(res.y, dtype=float)

        r = y[:, 0]
        pr = y[:, 1]
        phi = y[:, 2]
        tcoord = y[:, 3]

        x = r * np.cos(phi)
        yxy = r * np.sin(phi)

        # 1) órbita no plano equatorial (x,y)
        plt.figure()
        plt.plot(x, yxy)
        plt.gca().set_aspect("equal", adjustable="box")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title(f"Schwarzschild orbit (equatorial): {c.name}")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(out_dir / f"{c.name}_orbit.png", dpi=200)
        plt.close()

        # 2) r(tau)
        plt.figure()
        plt.plot(tau, r)
        plt.xlabel("tau")
        plt.ylabel("r")
        plt.title(f"r(tau): {c.name}")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(out_dir / f"{c.name}_r_tau.png", dpi=200)
        plt.close()

        # 3) constraint epsilon (signed) e log(|epsilon|)
        eps_series = np.asarray(res.constraint_eps, dtype=float)

        plt.figure()
        plt.plot(tau, eps_series, label="epsilon(tau)")
        plt.xlabel("tau")
        plt.ylabel("epsilon")
        plt.title(f"Constraint (signed): {c.name}")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"{c.name}_constraint.png", dpi=200)
        plt.close()

        plt.figure()
        plt.semilogy(tau, np.maximum(np.abs(eps_series), eps), label="|epsilon(tau)|")
        plt.xlabel("tau")
        plt.ylabel("|epsilon| (log)")
        plt.title(f"Constraint drift (log): {c.name}")
        plt.grid(True, which="both")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"{c.name}_constraint_log.png", dpi=200)
        plt.close()

    print(f"Figuras geradas em: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
