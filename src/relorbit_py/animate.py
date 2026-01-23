# src/relorbit_py/animate.py
from __future__ import annotations

import argparse
import copy
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from .simulate import load_cases_yaml, simulate_case


# ============================================================
# Utils / IO
# ============================================================

def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _normalize_suites(root: Dict[str, Any]) -> Dict[str, Any]:
    # aceita {"suites": {...}} ou {...}
    if isinstance(root, dict) and "suites" in root and isinstance(root["suites"], dict):
        return root["suites"]
    if isinstance(root, dict):
        return root
    raise RuntimeError("cases.yaml inválido: root não é dict.")


def _find_case(suites: Dict[str, Any], suite_name: str, case_name: str) -> Tuple[str, Dict[str, Any]]:
    if suite_name not in suites:
        raise RuntimeError(f"Suite '{suite_name}' não encontrada. Disponíveis: {list(suites.keys())}")
    suite = suites[suite_name] or {}
    cases = suite.get("cases", []) or []
    for c in cases:
        if str(c.get("name", "")).strip() == case_name:
            return suite_name, c
    avail = [str(c.get("name", "")).strip() for c in cases]
    raise RuntimeError(f"Case '{case_name}' não encontrado em '{suite_name}'. Disponíveis: {avail}")


def _compute_stride(n: int, stride: int, max_frames: int) -> int:
    if n <= 0:
        return 1
    if stride and stride > 0:
        return stride
    return max(1, int(math.ceil(n / max(1, max_frames))))


def _safe_float(x: Any, default: float) -> float:
    try:
        return float(x)
    except Exception:
        return default


# ============================================================
# Extract trajectory from engine
# ============================================================

@dataclass
class Traj2D:
    s: np.ndarray      # tau (schw) ou t (newton)
    x: np.ndarray
    y: np.ndarray
    r: np.ndarray
    phi: Optional[np.ndarray] = None
    M: Optional[float] = None
    dt_dtau: Optional[np.ndarray] = None


def _extract_traj(res: Any, case: Dict[str, Any]) -> Traj2D:
    # Newton
    if hasattr(res, "t") and hasattr(res, "y"):
        t = np.array(res.t, dtype=float)
        y = np.array(res.y, dtype=float)
        x = y[:, 0]
        yy = y[:, 1]
        r = np.sqrt(x * x + yy * yy)
        return Traj2D(s=t, x=x, y=yy, r=r)

    # Schwarzschild equatorial
    if hasattr(res, "tau") and hasattr(res, "r") and hasattr(res, "phi"):
        tau = np.array(res.tau, dtype=float)
        r = np.array(res.r, dtype=float)
        phi = np.array(res.phi, dtype=float)
        x = r * np.cos(phi)
        y = r * np.sin(phi)

        params = case.get("params", {}) or {}
        M = params.get("M", None)
        if hasattr(res, "M"):
            M = getattr(res, "M")
        M = None if M is None else float(M)

        dt_dtau = None
        if hasattr(res, "ut_fd"):
            try:
                u = np.array(res.ut_fd, dtype=float)
                if u.size == tau.size:
                    dt_dtau = u
            except Exception:
                pass

        if dt_dtau is None and M is not None:
            # fallback teórico se tiver E
            E = params.get("E", None)
            if hasattr(res, "E"):
                E = getattr(res, "E")
            if E is not None:
                E = float(E)
                A = 1.0 - 2.0 * M / r
                A = np.where(np.abs(A) < 1e-300, np.sign(A) * 1e-300, A)
                dt_dtau = E / A

        return Traj2D(s=tau, x=x, y=y, r=r, phi=phi, M=M, dt_dtau=dt_dtau)

    raise RuntimeError(f"Trajetória não reconhecida (res={type(res)}).")


# ============================================================
# Physics-consistent pr0 for Schwarzschild (approx; matches your validation constraint form)
#   Timelike: pr^2 = E^2 - A*(1 + L^2/r^2)  with A = 1 - 2M/r
# ============================================================

def _compute_pr_from_constraint(M: float, E: float, L: float, r: float, sign_hint: float) -> float:
    A = 1.0 - 2.0 * M / r
    rhs = E * E - A * (1.0 + (L * L) / (r * r))
    if rhs < 0.0:
        # sem energia suficiente para esse deslocamento local -> zera (visual), evita NaN
        return 0.0
    pr = math.sqrt(rhs)
    return -pr if sign_hint < 0 else pr


# ============================================================
# Embedding / Grid
# ============================================================

def z_of_r_flamm(M: float, r: np.ndarray) -> np.ndarray:
    """
    Embedding “exato” do espaço de Schwarzschild (slice t=const):
      z(r) = -sqrt(8M (r - 2M)), r >= 2M
    (mesmo do script que tu mandou)
    """
    r = np.asarray(r, dtype=float)
    val = 8.0 * M * (r - 2.0 * M)
    val = np.maximum(val, 0.0)
    return -np.sqrt(val)


def z_of_r_well(M: float, r: np.ndarray, scale: float, softening: float) -> np.ndarray:
    """
    Poço visual “plano no infinito”:
      z(r) = -scale*M / sqrt(r^2 + softening^2)
    """
    r = np.asarray(r, dtype=float)
    return -scale * M / np.sqrt(r * r + softening * softening)


def _grid_r_theta(Rs: float, limit: float, n_near: int, n_far: int, n_theta: int) -> Tuple[np.ndarray, np.ndarray]:
    # radial mais denso perto do horizonte, mais esparso longe
    r1 = np.linspace(Rs * 1.0001, Rs * 2.0, max(5, n_near))
    r2 = np.linspace(Rs * 2.0, limit * 1.5, max(5, n_far))
    r_grid = np.unique(np.concatenate([r1, r2]))
    th_grid = np.linspace(0.0, 2.0 * np.pi, max(24, n_theta))
    return r_grid, th_grid


def _add_event_horizon_disk(ax: Any, M: float, z_func, bh_eps: float, n: int = 240) -> None:
    """
    Horizonte como disco negro “tampando” o buraco (sem esfera amarela).
    """
    Rs = 2.0 * M
    r = Rs * (1.0 + float(bh_eps))
    th = np.linspace(0.0, 2.0 * np.pi, n)
    xs = r * np.cos(th)
    ys = r * np.sin(th)
    rs = np.full_like(th, r)
    zs = z_func(rs)

    verts = [list(zip(xs, ys, zs))]
    poly = Poly3DCollection(verts, facecolor=(0, 0, 0, 1), edgecolor=(0, 0, 0, 1))
    ax.add_collection3d(poly)


# ============================================================
# Main animation: Schwarzschild "spaghettification style"
# ============================================================

def animate_schwarzschild_spaghetti(
    suite_name: str,
    case: Dict[str, Any],
    out_base: str,
    fmt: str,
    fps: int,
    duration_sec: float,
    max_frames: int,
    stride: int,
    limit: float,
    grid_theta: int,
    grid_near: int,
    grid_far: int,
    embedding: str,
    well_scale: float,
    well_softening: float,
    n_particles: int,
    body_size: float,
    trail_len: int,
    rotate: bool,
    rotate_speed: float,
    show_dt_dtau: bool,
    ghost_orbit: bool,
    bh_eps: float,
) -> Tuple[bool, str]:
    case_name = str(case.get("name", "case")).strip()
    params = case.get("params", {}) or {}
    M = float(params.get("M", 1.0))
    E = float(params.get("E"))
    L = float(params.get("L"))
    pr0_base = _safe_float(case.get("pr0", 0.0), 0.0)
    sign_hint = -1.0 if pr0_base < 0 else +1.0

    state0 = case.get("state0", None)
    if not state0 or len(state0) < 2:
        raise RuntimeError("Case Schwarzschild precisa state0=[r0, phi0].")
    r0_cm = float(state0[0])
    phi0_cm = float(state0[1])

    # escolher embedding (estilo do script vs plano no infinito)
    if embedding.lower() == "flamm":
        z_func = lambda rr: z_of_r_flamm(M, rr)
    elif embedding.lower() == "well":
        z_func = lambda rr: z_of_r_well(M, rr, scale=well_scale, softening=well_softening)
    else:
        raise RuntimeError("embedding inválido. Use 'flamm' ou 'well'.")

    # frames efetivos
    frames = int(max(30, min(max_frames, fps * max(1.0, float(duration_sec)))))

    # cria casos das partículas (anel)
    # offsets em coordenadas locais aproximadas (ok longe; perto do horizonte vira “efeito dramático” mesmo)
    particle_cases: List[Dict[str, Any]] = []
    for i in range(max(1, int(n_particles))):
        ang = 2.0 * np.pi * i / max(1, int(n_particles))
        dx = float(body_size) * math.cos(ang)
        dy = float(body_size) * math.sin(ang)

        # converte pra (r,phi) global aproximado no plano
        # parte de (r0_cm,phi0_cm) -> cartesiano -> soma offset -> volta pra polar
        x0 = r0_cm * math.cos(phi0_cm)
        y0 = r0_cm * math.sin(phi0_cm)
        xp = x0 + dx
        yp = y0 + dy
        r_p = math.sqrt(xp * xp + yp * yp)
        phi_p = math.atan2(yp, xp)

        # pr0 coerente com constraint (mantém fidelidade numérica ao teu modelo)
        pr_p = _compute_pr_from_constraint(M=M, E=E, L=L, r=r_p, sign_hint=sign_hint)

        c = copy.deepcopy(case)
        c["state0"] = [r_p, phi_p]
        c["pr0"] = pr_p
        particle_cases.append(c)

    # roda engine para cada partícula (fiel ao teu solver)
    trajs: List[Traj2D] = []
    for c in particle_cases:
        res = simulate_case(c, suite_name)
        tr = _extract_traj(res, c)
        trajs.append(tr)

    # stride (reduz travamento)
    # usa o menor tamanho disponível entre as trajetórias
    nmin = min(int(tr.s.size) for tr in trajs)
    step = _compute_stride(nmin, stride=stride, max_frames=frames)
    idx = np.arange(0, nmin, step, dtype=int)
    if idx.size < 2:
        idx = np.arange(0, nmin, dtype=int)

    # ============ FIG
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(10, 8), dpi=120)
    ax = fig.add_subplot(111, projection="3d")

    ax.set_facecolor("black")
    try:
        ax.set_axis_off()
        ax.grid(False)
    except Exception:
        pass

    Rs = 2.0 * M

    # ============ GRID
    r_grid, th_grid = _grid_r_theta(Rs=Rs, limit=float(limit), n_near=int(grid_near), n_far=int(grid_far), n_theta=int(grid_theta))
    R, TH = np.meshgrid(r_grid, th_grid)
    Xg = R * np.cos(TH)
    Yg = R * np.sin(TH)
    Zg = z_func(R)

    ax.plot_wireframe(Xg, Yg, Zg, color="#00FFFF", alpha=0.18, linewidth=0.55)

    # ============ BH (disco)
    _add_event_horizon_disk(ax, M=M, z_func=z_func, bh_eps=bh_eps)

    # ============ elementos (linhas + pontos)
    line_refs = []
    for _ in range(max(1, int(n_particles))):
        ln, = ax.plot([], [], [], color="#FFD700", lw=1.5, alpha=0.85)
        line_refs.append(ln)

    scat = ax.scatter([], [], [], color="white", s=10)

    status_text = ax.text2D(
        0.05, 0.95, "",
        transform=ax.transAxes,
        color="white",
        family="monospace",
    )

    # ============ limites / câmera
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)

    zmin = float(np.nanmin(Zg))
    zmax = float(np.nanmax(Zg))
    ax.set_zlim(zmin * 1.05, max(2.0, zmax * 1.15))

    elev0, azim0 = 30.0, -60.0
    ax.view_init(elev=elev0, azim=azim0)

    # ============ órbita fantasma do CM (opcional)
    if ghost_orbit:
        # usa partícula 0 como proxy do CM (já que offsets são pequenos)
        tr0 = trajs[0]
        n0 = min(tr0.r.size, nmin)
        x0 = tr0.x[:n0]
        y0 = tr0.y[:n0]
        z0 = z_func(tr0.r[:n0])
        ax.plot(x0, y0, z0, color=(1.0, 1.0, 1.0, 0.15), lw=1.0)

    # ============ dt/dtau (se disponível)
    dt_dtau = None
    if show_dt_dtau and trajs[0].dt_dtau is not None and trajs[0].dt_dtau.size >= nmin:
        dt_dtau = np.asarray(trajs[0].dt_dtau[:nmin], dtype=float)[idx]

    # ============ atualização
    rest_len = float(body_size) * 2.0 * math.sin(math.pi / max(3, int(n_particles)))  # distância “ideal” entre vizinhos

    def update(frame_i: int):
        i = int(frame_i)
        if rotate:
            ax.view_init(elev=elev0, azim=azim0 + i * float(rotate_speed))

        x_data, y_data, z_data = [], [], []
        r_center_sum = 0.0
        valid_p = 0

        # coleta pontos
        for p in range(len(trajs)):
            tr = trajs[p]
            k = idx[min(i, idx.size - 1)]
            rr = float(tr.r[k])

            # trava na borda (visual)
            if rr <= Rs * (1.0 + float(bh_eps)):
                rr = Rs * (1.0 + float(bh_eps))

            ph = float(tr.phi[k]) if tr.phi is not None else math.atan2(tr.y[k], tr.x[k])
            xx = rr * math.cos(ph)
            yy = rr * math.sin(ph)
            zz = float(z_func(np.array([rr]))[0])

            x_data.append(xx)
            y_data.append(yy)
            z_data.append(zz)

            if rr > Rs * (1.0 + 5.0 * float(bh_eps)):
                r_center_sum += rr
                valid_p += 1

        # desenha conexões (cor por tensão)
        for p in range(len(trajs)):
            q = (p + 1) % len(trajs)
            dx = x_data[p] - x_data[q]
            dy = y_data[p] - y_data[q]
            dz = z_data[p] - z_data[q]
            dist = math.sqrt(dx * dx + dy * dy + dz * dz)

            # mesmas regras do script (tensão -> laranja -> vermelho)
            color = "#FFD700"
            if dist > rest_len * 2.0:
                color = "#FF0000"
            elif dist > rest_len * 1.2:
                color = "#FF4500"

            line_refs[p].set_data([x_data[p], x_data[q]], [y_data[p], y_data[q]])
            line_refs[p].set_3d_properties([z_data[p], z_data[q]])
            line_refs[p].set_color(color)

            width = max(0.6, 2.0 / (dist / max(1e-9, rest_len) + 0.1))
            line_refs[p].set_linewidth(width)

        scat._offsets3d = (x_data, y_data, z_data)

        # texto de status
        k = idx[min(i, idx.size - 1)]
        tau = float(trajs[0].s[k])
        dist_avg = (r_center_sum / max(1, valid_p)) if valid_p > 0 else Rs

        msg = f"TEMPO PRÓPRIO: {tau:.2f}\nDISTÂNCIA: {dist_avg:.3f} M"
        if dist_avg < 6.0 * M and dist_avg > Rs:
            msg += "\nALERTA: REGIÃO < ISCO (instável)"
        elif dist_avg <= Rs * (1.0 + 2.0 * float(bh_eps)):
            msg += "\nSTATUS: HORIZONTE CRUZADO"

        if dt_dtau is not None:
            msg += f"\ndt/dτ≈{float(dt_dtau[min(i, dt_dtau.size-1)]):.3g}"

        status_text.set_text(msg)

        # trilha (opcional simples: “cauda” no texto já é suficiente; mas aqui não desenhamos trail extra)
        return line_refs + [scat, status_text]

    # frames reais da animação
    n_frames = min(int(idx.size), int(frames))

    ani = animation.FuncAnimation(fig, update, frames=n_frames, interval=int(1000 / max(1, fps)), blit=False)

    outdir = os.path.join(out_base, "animations")
    _ensure_dir(outdir)
    outpath = os.path.join(outdir, f"{suite_name}__{case_name}__spaghetti.{fmt.lower()}")

    # salvar (mp4 precisa ffmpeg; gif usa pillow)
    try:
        if fmt.lower() == "mp4":
            writer = animation.FFMpegWriter(fps=fps, bitrate=2400)
            ani.save(outpath, writer=writer, dpi=120)
        else:
            writer = animation.PillowWriter(fps=fps)
            ani.save(outpath, writer=writer, dpi=120)
        plt.close(fig)
        return True, outpath
    except FileNotFoundError:
        # ffmpeg ausente -> cai pra gif
        try:
            outpath2 = os.path.splitext(outpath)[0] + ".gif"
            writer = animation.PillowWriter(fps=fps)
            ani.save(outpath2, writer=writer, dpi=120)
            plt.close(fig)
            return True, outpath2
        except Exception as e2:
            plt.close(fig)
            return False, f"Falhou mp4 (ffmpeg ausente) e falhou gif: {e2!r}"
    except Exception as e:
        plt.close(fig)
        return False, f"Falha ao salvar: {e!r}"


# ============================================================
# Newton (fallback simples: ponto + órbita fantasma 3D “plana”)
# ============================================================

def animate_newton_simple(
    suite_name: str,
    case: Dict[str, Any],
    out_base: str,
    fmt: str,
    fps: int,
    duration_sec: float,
    max_frames: int,
    stride: int,
    limit: float,
    rotate: bool,
    rotate_speed: float,
    ghost_orbit: bool,
) -> Tuple[bool, str]:
    case_name = str(case.get("name", "case")).strip()
    res = simulate_case(case, suite_name)
    tr = _extract_traj(res, case)

    frames = int(max(30, min(max_frames, fps * max(1.0, float(duration_sec)))))
    n = int(tr.s.size)
    step = _compute_stride(n, stride=stride, max_frames=frames)
    idx = np.arange(0, n, step, dtype=int)
    if idx.size < 2:
        idx = np.arange(0, n, dtype=int)

    x = tr.x[idx]
    y = tr.y[idx]
    z = np.zeros_like(x)

    plt.style.use("dark_background")
    fig = plt.figure(figsize=(10, 8), dpi=120)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("black")
    try:
        ax.set_axis_off()
    except Exception:
        pass

    if ghost_orbit:
        ax.plot(tr.x, tr.y, np.zeros_like(tr.x), color=(1, 1, 1, 0.18), lw=1.0)

    (point,) = ax.plot([], [], [], "o", color="white", markersize=6)
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_zlim(-1, 1)

    elev0, azim0 = 35.0, -60.0
    ax.view_init(elev=elev0, azim=azim0)

    def update(i: int):
        if rotate:
            ax.view_init(elev=elev0, azim=azim0 + i * float(rotate_speed))
        point.set_data([x[i]], [y[i]])
        point.set_3d_properties([z[i]])
        ax.set_title(f"NEWTON | {suite_name}/{case_name} | t={float(tr.s[idx[i]]):.3f}", color="white", pad=8)
        return (point,)

    ani = animation.FuncAnimation(fig, update, frames=min(int(idx.size), frames), interval=int(1000 / max(1, fps)), blit=False)

    outdir = os.path.join(out_base, "animations")
    _ensure_dir(outdir)
    outpath = os.path.join(outdir, f"{suite_name}__{case_name}__simple.{fmt.lower()}")

    try:
        if fmt.lower() == "mp4":
            writer = animation.FFMpegWriter(fps=fps, bitrate=2400)
            ani.save(outpath, writer=writer, dpi=120)
        else:
            writer = animation.PillowWriter(fps=fps)
            ani.save(outpath, writer=writer, dpi=120)
        plt.close(fig)
        return True, outpath
    except FileNotFoundError:
        try:
            outpath2 = os.path.splitext(outpath)[0] + ".gif"
            writer = animation.PillowWriter(fps=fps)
            ani.save(outpath2, writer=writer, dpi=120)
            plt.close(fig)
            return True, outpath2
        except Exception as e2:
            plt.close(fig)
            return False, f"Falhou mp4 (ffmpeg ausente) e falhou gif: {e2!r}"
    except Exception as e:
        plt.close(fig)
        return False, f"Falha ao salvar: {e!r}"


# ============================================================
# CLI
# ============================================================

def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m relorbit_py.animate")

    ap.add_argument("--cases", required=True, help="Caminho para cases.yaml")
    ap.add_argument("--out", default="out", help="Diretório base de saída (default: out)")
    ap.add_argument("--format", default="gif", choices=["gif", "mp4"], help="Formato (gif/mp4)")

    # um caso só (não travar)
    ap.add_argument("--suite", required=True, help="Nome da suite (ex: schwarzschild)")
    ap.add_argument("--case", dest="case_name", required=True, help="Nome do case")

    # tempo / performance
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--duration-sec", type=float, default=20.0)
    ap.add_argument("--max-frames", type=int, default=700)
    ap.add_argument("--stride", type=int, default=0)

    # mundo
    ap.add_argument("--limit", type=float, default=40.0, help="limite XY (default: 40)")

    # grid (estilo do script)
    ap.add_argument("--grid-theta", type=int, default=60)
    ap.add_argument("--grid-near", type=int, default=30)
    ap.add_argument("--grid-far", type=int, default=20)

    # embedding
    ap.add_argument("--embedding", choices=["flamm", "well"], default="well",
                    help="flamm = z=-sqrt(8M(r-2M)) (script), well = plano no infinito (default)")
    ap.add_argument("--well-scale", type=float, default=18.0)
    ap.add_argument("--well-softening", type=float, default=1.2)

    # “corpo” (spaghetti)
    ap.add_argument("--n-particles", type=int, default=16)
    ap.add_argument("--body-size", type=float, default=0.6)
    ap.add_argument("--trail-len", type=int, default=60)  # mantido por compat, mas não desenha trail extra aqui

    # extras
    ap.add_argument("--rotate", action="store_true", help="Rotaciona câmera (default: off)")
    ap.add_argument("--rotate-speed", type=float, default=0.10)
    ap.add_argument("--show-dt-dtau", action="store_true")
    ap.add_argument("--ghost-orbit", action="store_true")
    ap.add_argument("--bh-eps", type=float, default=1e-3)

    args = ap.parse_args(argv)

    root = load_cases_yaml(str(args.cases))
    suites = _normalize_suites(root)
    suite_name, case = _find_case(suites, args.suite, args.case_name)

    model = str(case.get("model", "")).strip().lower()

    try:
        if model == "schwarzschild":
            ok, msg = animate_schwarzschild_spaghetti(
                suite_name=suite_name,
                case=case,
                out_base=str(args.out),
                fmt=str(args.format),
                fps=int(args.fps),
                duration_sec=float(args.duration_sec),
                max_frames=int(args.max_frames),
                stride=int(args.stride),
                limit=float(args.limit),
                grid_theta=int(args.grid_theta),
                grid_near=int(args.grid_near),
                grid_far=int(args.grid_far),
                embedding=str(args.embedding),
                well_scale=float(args.well_scale),
                well_softening=float(args.well_softening),
                n_particles=int(args.n_particles),
                body_size=float(args.body_size),
                trail_len=int(args.trail_len),
                rotate=bool(args.rotate),
                rotate_speed=float(args.rotate_speed),
                show_dt_dtau=bool(args.show_dt_dtau),
                ghost_orbit=bool(args.ghost_orbit),
                bh_eps=float(args.bh_eps),
            )
        else:
            ok, msg = animate_newton_simple(
                suite_name=suite_name,
                case=case,
                out_base=str(args.out),
                fmt=str(args.format),
                fps=int(args.fps),
                duration_sec=float(args.duration_sec),
                max_frames=int(args.max_frames),
                stride=int(args.stride),
                limit=float(args.limit),
                rotate=bool(args.rotate),
                rotate_speed=float(args.rotate_speed),
                ghost_orbit=bool(args.ghost_orbit),
            )

        if ok:
            print(f"==> OK: {msg}")
            return 0
        print(f"==> FAIL: {msg}")
        return 2

    except Exception as e:
        print(f"==> FAIL: {e}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
