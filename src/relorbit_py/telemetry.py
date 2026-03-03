import math
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class Telemetry:
    dt_dtau: List[float]     # dt/dtau ao longo da trajetória
    latency: List[float]     # tcoord - tau
    freq_ratio: List[float]  # proxy de redshift/doppler ~ 1/(dt/dtau)
    visible: List[bool]      # visível para observador distante (oclusão 2D)


def _central_diff(x: List[float], t: List[float]) -> List[float]:
    n = len(x)
    out = [float("nan")] * n
    if n < 2:
        return out
    for i in range(n):
        if i == 0:
            dt = t[1] - t[0]
            dx = x[1] - x[0]
        elif i == n - 1:
            dt = t[-1] - t[-2]
            dx = x[-1] - x[-2]
        else:
            dt = t[i + 1] - t[i - 1]
            dx = x[i + 1] - x[i - 1]
        out[i] = float("nan") if dt == 0 else dx / dt
    return out


def _visibility_2d_occlusion(
    r: float,
    phi: float,
    r_occ: float,
    observer_dir: Tuple[float, float] = (1.0, 0.0),
) -> bool:
    # Posição no plano equatorial
    x = r * math.cos(phi)
    y = r * math.sin(phi)

    nx, ny = observer_dir
    nn = math.hypot(nx, ny)
    if nn == 0:
        nx, ny = 1.0, 0.0
    else:
        nx, ny = nx / nn, ny / nn

    # Linha de visada: p(s) = p + s*n, s>=0
    # Se a distância mínima dessa semi-reta até a origem for < r_occ => ocluído.
    pdotn = x * nx + y * ny
    s_star = -pdotn

    # Se o ponto mais próximo cai “atrás” do próprio ponto (s<=0), não cruza o disco à frente.
    if s_star <= 0.0:
        return True

    x2 = x + s_star * nx
    y2 = y + s_star * ny
    dmin = math.hypot(x2, y2)

    return dmin >= r_occ


def compute_telemetry(
    traj,
    *,
    r_occ: float,
    prefer_theory: bool = True,
    observer_dir: Tuple[float, float] = (1.0, 0.0),
) -> Telemetry:
    tau = list(traj.tau)
    r = list(traj.r)
    phi = list(traj.phi)

    # tcoord: em Schwarzschild/Kerr você expõe como tcoord, e no pybind às vezes tem alias .t
    if hasattr(traj, "tcoord"):
        tcoord = list(traj.tcoord)
    elif hasattr(traj, "t"):
        tcoord = list(traj.t)
    else:
        raise AttributeError("traj não possui tcoord/t")

    # dt/dtau: preferir série teórica (mais limpa), senão FD, senão derivada numérica de tcoord(tau)
    dt_dtau = None
    if prefer_theory and hasattr(traj, "ut_theory") and len(traj.ut_theory) == len(tau):
        dt_dtau = list(traj.ut_theory)
    elif hasattr(traj, "ut_fd") and len(traj.ut_fd) == len(tau):
        dt_dtau = list(traj.ut_fd)
    else:
        dt_dtau = _central_diff(tcoord, tau)

    latency = [tcoord[i] - tau[i] for i in range(len(tau))]

    freq_ratio = []
    for v in dt_dtau:
        if (not math.isfinite(v)) or v == 0.0:
            freq_ratio.append(float("nan"))
        else:
            freq_ratio.append(1.0 / v)

    visible = [
        _visibility_2d_occlusion(r[i], phi[i], r_occ, observer_dir=observer_dir)
        for i in range(len(tau))
    ]

    return Telemetry(dt_dtau=dt_dtau, latency=latency, freq_ratio=freq_ratio, visible=visible)