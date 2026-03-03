import os
import math
import matplotlib.pyplot as plt


def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _savefig(path: str) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_telemetry(outdir: str, name: str, traj, tele) -> None:
    _ensure_dir(outdir)

    tau = list(traj.tau)
    r = list(traj.r)

    # 1) Latency vs tau
    plt.figure()
    plt.plot(tau, tele.latency)
    plt.xlabel("tau")
    plt.ylabel("tcoord - tau")
    plt.title(f"{name} | Communication Latency vs tau")
    _savefig(os.path.join(outdir, f"{name}_communication_latency_vs_tau.png"))

    # 2) Latency vs r
    plt.figure()
    plt.plot(r, tele.latency)
    plt.xlabel("r")
    plt.ylabel("tcoord - tau")
    plt.title(f"{name} | Communication Latency vs r")
    _savefig(os.path.join(outdir, f"{name}_communication_latency_vs_r.png"))

    # 3) dt/dtau vs r (mask de valores não finitos)
    x = []
    y = []
    for i in range(len(r)):
        v = tele.dt_dtau[i]
        if math.isfinite(v):
            x.append(r[i])
            y.append(v)
    plt.figure()
    plt.plot(x, y)
    plt.xlabel("r")
    plt.ylabel("dt/dtau")
    plt.title(f"{name} | dt/dtau vs r")
    _savefig(os.path.join(outdir, f"{name}_dt_dtau_vs_r.png"))

    # 4) freq ratio vs r (proxy)
    x = []
    y = []
    for i in range(len(r)):
        v = tele.freq_ratio[i]
        if math.isfinite(v):
            x.append(r[i])
            y.append(v)
    plt.figure()
    plt.plot(x, y)
    plt.xlabel("r")
    plt.ylabel("freq_ratio ~ 1/(dt/dtau)")
    plt.title(f"{name} | Redshift/Doppler proxy vs r")
    _savefig(os.path.join(outdir, f"{name}_freq_ratio_vs_r.png"))

    # 5) Visibility vs tau (0/1)
    vv = [1.0 if b else 0.0 for b in tele.visible]
    plt.figure()
    plt.plot(tau, vv)
    plt.xlabel("tau")
    plt.ylabel("visible (1=yes, 0=no)")
    plt.title(f"{name} | Visibility vs tau")
    _savefig(os.path.join(outdir, f"{name}_visibility_vs_tau.png"))