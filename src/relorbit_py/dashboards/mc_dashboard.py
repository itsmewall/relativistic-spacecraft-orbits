# src/relorbit_py/mc_dashboard.py
"""
Monte Carlo Dashboard — Análise de Dispersão Relativística Kerr 6-DOF
======================================================================

CHANGELOG vs versão anterior
──────────────────────────────
A) Coordenador robusto
   • Loop wait(FIRST_COMPLETED) — drena todos os N jobs sem perder futures.
   • stop_flag respeitado sem travar a UI (timeout=2 s no wait).
   • Timeout por simulação (SIM_TIMEOUT_S=120): job lento → MCResult ERROR.

B) Auto-save streaming + export confiável
   • StreamingCSV: abre ficheiro com timestamp ao INICIAR; escreve cada linha
     à medida que chega; flush a cada FLUSH_EVERY linhas; fecha ao terminar.
   • Log mostra o caminho e total de linhas ao fechar.
   • Botão "EXPORTAR CÓPIA" faz shutil.copy2 em thread — não bloqueia Tk.
   • all_results removido — sem acumulação de 100k objectos em memória.

C) 6-DOF configurável
   • Checkboxes TIDAL (modelo: WEAK_N / DIAG_EIJ / RIEMANN_FD) e THRUST.
   • Worker passa tidal/engine correctamente para run_kerr_6dof_mission.
   • CSV com colunas extras: T_rot0, T_rot_abs_drift, T_rot_rel_drift,
     tidal_norm_max, work_tidal, delta_T_rot, work_energy_err,
     angular_balance_err, align_angle_final, mass_consumed.
   • Leitura defensiva de atributos do traj (_safe_*) — nunca quebra.

D) UI sem travamentos
   • Log limitado a 4000 linhas; taxa de log = max(1, N//200).
   • _redraw_plots() sem tight_layout() (caro); usa draw_idle().
   • Refresh drena máx 80 mensagens de log por ciclo.

E) Memória
   • MCState sem all_results. Amostras para plot limitadas a 20 000.

Uso:
    py -3 src/relorbit_py/mc_dashboard.py
    py -3 src/relorbit_py/mc_dashboard.py --n 500 --workers 4
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import queue
import random
import shutil
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

# ── Paleta ────────────────────────────────────────────────────────────────────
BG     = "#0a0f14"
BG2    = "#0f1820"
BG3    = "#141e28"
BORDER = "#1e3040"
GREEN  = "#00e5a0"
CYAN   = "#00c8ff"
AMBER  = "#ffaa00"
RED    = "#ff4444"
WHITE  = "#ddeeff"
MUTED  = "#3a5060"

# ── Constantes de comportamento ───────────────────────────────────────────────
AUTOSAVE_DIR    = Path("out/mc")
SIM_TIMEOUT_S   = 120.0    # timeout por simulação individual
PIPELINE_FACTOR = 6        # max_inflight = workers × PIPELINE_FACTOR

plt.rcParams.update({
    "figure.facecolor": BG2,  "axes.facecolor": BG3,
    "axes.edgecolor": BORDER, "axes.labelcolor": MUTED,
    "xtick.color": MUTED,     "ytick.color": MUTED,
    "text.color": WHITE,      "grid.color": BORDER,
    "grid.alpha": 0.4,        "font.family": "monospace",
    "font.size": 9,
})


# ═══════════════════════════════════════════════════════════════════════════════
# ESTRUTURAS DE DADOS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class NominalParams:
    M:            float = 1.0
    a:            float = 0.5
    E:            float = 0.95
    L:            float = 3.8
    r0:           float = 10.0
    pr0:          float = 0.0
    mass0_kg:     float = 1000.0
    dry_mass_kg:  float = 300.0
    tau_final:    float = 500.0
    dt:           float = 0.005
    record_every: int   = 50
    capture_r:    float = 2.0
    wz:           float = 0.02
    Ixx:          float = 100.0
    Iyy:          float = 200.0
    Izz:          float = 150.0
    # 6-DOF extras
    tidal_enabled:  bool  = False
    tidal_model:    str   = "DIAG_EIJ"   # WEAK_N | DIAG_EIJ | RIEMANN_FD
    thrust_enabled: bool  = False
    F_newton:       float = 10.0
    isp_s:          float = 3000.0


@dataclass
class DispersionSource:
    name:     str
    param:    str
    sigma:    float
    enabled:  bool = True
    absolute: bool = False   # True → sigma é absoluto (não relativo)


DEFAULT_DISPERSIONS: List[DispersionSource] = [
    DispersionSource("Energia E",         "E",        0.001),
    DispersionSource("Mom. angular L",    "L",        0.001),
    DispersionSource("Raio inicial r₀",   "r0",       0.005),
    DispersionSource("Massa da nave",     "mass0_kg", 0.005),
    DispersionSource("Mom. radial pr₀",   "pr0",      0.010, enabled=False, absolute=True),
    DispersionSource("Spin ωz",           "wz",       0.050, enabled=False),
    DispersionSource("Spin Kerr a",       "a",        0.010, enabled=False),
]

# ── Colunas CSV ───────────────────────────────────────────────────────────────
CSV_ALL_COLS = [
    "index", "status",
    "r_final", "r_min", "r_max",
    "eps_rms", "qnorm_err",
    "mass_consumed",
    "T_rot0", "T_rot_abs_drift", "T_rot_rel_drift",
    "tidal_norm_max", "work_tidal", "delta_T_rot",
    "work_energy_err", "angular_balance_err", "align_angle_final",
]


@dataclass
class MCResult:
    index:        int
    status:       str    # "OK" | "CAPTURE" | "ERROR"
    r_final:      float
    r_min:        float
    r_max:        float
    eps_rms:      float
    qnorm_err:    float
    mass_consumed: float
    T_rot0:             float = 0.0
    T_rot_abs_drift:    float = float("nan")
    T_rot_rel_drift:    float = float("nan")
    tidal_norm_max:     float = float("nan")
    work_tidal:         float = float("nan")
    delta_T_rot:        float = float("nan")
    work_energy_err:    float = float("nan")
    angular_balance_err:float = float("nan")
    align_angle_final:  float = float("nan")
    error_msg: str = ""

    def to_row(self) -> list:
        def _fmt(v):
            if isinstance(v, float) and not math.isfinite(v):
                return ""
            return v
        return [
            self.index, self.status,
            _fmt(self.r_final), _fmt(self.r_min), _fmt(self.r_max),
            _fmt(self.eps_rms), _fmt(self.qnorm_err),
            _fmt(self.mass_consumed),
            _fmt(self.T_rot0), _fmt(self.T_rot_abs_drift), _fmt(self.T_rot_rel_drift),
            _fmt(self.tidal_norm_max), _fmt(self.work_tidal), _fmt(self.delta_T_rot),
            _fmt(self.work_energy_err), _fmt(self.angular_balance_err),
            _fmt(self.align_angle_final),
        ]


# ═══════════════════════════════════════════════════════════════════════════════
# WORKER  (spawn-safe: sem imports pesados no topo do módulo)
# ═══════════════════════════════════════════════════════════════════════════════

def _run_one(args: Tuple[int, Dict[str, Any]]) -> MCResult:
    """Corre numa simulação Kerr 6-DOF. Chamado em ProcessPoolExecutor."""
    idx, p = args
    try:
        from relorbit_py.core.simulate_kerr_6dof import run_kerr_6dof_mission  # lazy

        tidal_en    = bool(p.get("tidal_enabled", False))
        tidal_model = str(p.get("tidal_model", "DIAG_EIJ"))
        thrust_en   = bool(p.get("thrust_enabled", False))
        tau_final   = float(p.get("tau_final", 500.0))

        m_cfg: Dict[str, Any] = {
            "name":  f"mc_{idx:06d}",
            "model": "kerr_6dof",
            "params": {
                "M":           p["M"],
                "a":           p["a"],
                "E":           p["E"],
                "L":           p["L"],
                "capture_r":   p.get("capture_r",  2.0),
                "capture_eps": 1e-12,
            },
            "state0": [p["r0"], 0.0],
            "pr0":    p.get("pr0", 0.0),
            "attitude0": {
                "q0": 1.0, "q1": 0.0, "q2": 0.0, "q3": 0.0,
                "wx": 0.0, "wy": 0.0, "wz": p.get("wz", 0.02),
            },
            "inertia": {
                "Ixx": p.get("Ixx", 100.0),
                "Iyy": p.get("Iyy", 200.0),
                "Izz": p.get("Izz", 150.0),
            },
            "engine": {
                "F_newton":    float(p.get("F_newton", 10.0)) if thrust_en else 0.0,
                "isp_s":       float(p.get("isp_s", 3000.0)),
                "mass0_kg":    p.get("mass0_kg",    1000.0),
                "dry_mass_kg": p.get("dry_mass_kg",  300.0),
                "nozzle_body": [0.0, 0.0, 1.0],
                "tau_on":  0.0,
                "tau_off": tau_final if thrust_en else 0.0,
            },
            "ext_torque": {"tx": 0.0, "ty": 0.0, "tz": 0.0},
            "tidal": {
                "enabled":         tidal_en,
                "model":           tidal_model if tidal_en else "NONE",
                "fd_eps_r":        1e-5,
                "Q_from_inertia":  True,
                "spin_correction": False,
            },
            "span":   [0.0, tau_final],
            "solver": {
                "dt":           p.get("dt", 0.005),
                "record_every": int(p.get("record_every", 50)),
            },
        }

        result = run_kerr_6dof_mission(m_cfg)
        traj   = result.traj

        if traj is None or not list(traj.tau):
            return MCResult(
                index=idx, status="ERROR",
                r_final=float("nan"), r_min=float("nan"), r_max=float("nan"),
                eps_rms=float("nan"), qnorm_err=float("nan"), mass_consumed=0.0,
                error_msg="trajectória vazia")

        # ── Extracção robusta de arrays ───────────────────────────────────────
        def _arr(attr: str):
            try:
                return np.asarray(getattr(traj, attr), dtype=float)
            except Exception:
                return np.array([], dtype=float)

        def _scalar(attr: str):
            try:
                v = float(getattr(traj, attr))
                return v if math.isfinite(v) else float("nan")
            except Exception:
                return float("nan")

        r_arr = _arr("r"); eps  = _arr("epsilon")
        qnorm = _arr("qnorm"); T_rot = _arr("T_rot")
        mass  = _arr("mass")
        tn    = _arr("tidal_norm");   al = _arr("align_angle_rad")

        status = "CAPTURE" if "CAPTURE" in str(traj.status) else "OK"

        eps_rms   = float(np.sqrt(np.mean(eps**2))) if len(eps)   else float("nan")
        qnorm_err = float(np.max(np.abs(qnorm-1.))) if len(qnorm) else float("nan")

        T0        = float(T_rot[0]) if len(T_rot) else 0.0
        T_abs     = float(np.max(T_rot) - np.min(T_rot)) if len(T_rot) else float("nan")
        # rel_drift: NaN quando T0 ≈ 0 (sem spin inicial); grava "" no CSV
        T_rel = T_abs / abs(T0) if (math.isfinite(T_abs) and abs(T0) > max(1e-30, 1e-12*abs(T0))) \
                else float("nan")

        mass_consumed = float(mass[0] - mass[-1]) if len(mass) > 1 else 0.0
        tidal_nm = float(np.max(tn)) if len(tn) else float("nan")
        align_f  = float(np.degrees(al[-1])) if len(al) else float("nan")

        # work_tidal / delta_T_rot / work_energy_err (só quando tidal activo)
        work_tidal = delta_Trot = work_e_err = ang_bal = float("nan")
        if tidal_en and len(T_rot) > 1:
            try:
                tau_a = _arr("tau")
                wx_ = _arr("wx"); wy_ = _arr("wy"); wz_ = _arr("wz")
                tx_ = _arr("tidal_tau_x")
                ty_ = _arr("tidal_tau_y")
                tz_ = _arr("tidal_tau_z")
                if len(tx_) == len(tau_a) and len(tau_a) > 1:
                    odt = wx_*tx_ + wy_*ty_ + wz_*tz_
                    work_tidal = float(np.trapz(odt, tau_a))
                    delta_Trot = float(T_rot[-1] - T_rot[0])
                    denom      = max(abs(delta_Trot), abs(work_tidal), 1e-30)
                    work_e_err = abs(delta_Trot - work_tidal) / denom

                    Ixx = p.get("Ixx", 100.0); Iyy = p.get("Iyy", 200.0)
                    Izz = p.get("Izz", 150.0)
                    L   = np.vstack([Ixx*wx_, Iyy*wy_, Izz*wz_]).T
                    om  = np.vstack([wx_, wy_, wz_]).T
                    rhs = np.vstack([tx_, ty_, tz_]).T - np.cross(om, L)
                    int_rhs = np.array([float(np.trapz(rhs[:,i], tau_a)) for i in range(3)])
                    dL      = L[-1] - L[0]
                    den2    = max(np.linalg.norm(dL), np.linalg.norm(int_rhs), 1e-30)
                    ang_bal = float(np.linalg.norm(dL - int_rhs) / den2)
            except Exception:
                pass

        return MCResult(
            index=idx, status=status,
            r_final=float(r_arr[-1]) if len(r_arr) else float("nan"),
            r_min=float(np.min(r_arr)) if len(r_arr) else float("nan"),
            r_max=float(np.max(r_arr)) if len(r_arr) else float("nan"),
            eps_rms=eps_rms, qnorm_err=qnorm_err, mass_consumed=mass_consumed,
            T_rot0=T0, T_rot_abs_drift=T_abs, T_rot_rel_drift=T_rel,
            tidal_norm_max=tidal_nm, work_tidal=work_tidal,
            delta_T_rot=delta_Trot, work_energy_err=work_e_err,
            angular_balance_err=ang_bal, align_angle_final=align_f,
        )

    except Exception as ex:
        return MCResult(
            index=idx, status="ERROR",
            r_final=float("nan"), r_min=float("nan"), r_max=float("nan"),
            eps_rms=float("nan"), qnorm_err=float("nan"), mass_consumed=0.0,
            error_msg=str(ex)[:250])


# ═══════════════════════════════════════════════════════════════════════════════
# PERTURBAÇÃO
# ═══════════════════════════════════════════════════════════════════════════════

def perturb(nominal: NominalParams,
            dispersions: List[DispersionSource],
            rng: random.Random) -> Dict[str, Any]:
    p: Dict[str, Any] = {
        "M": nominal.M, "a": nominal.a, "E": nominal.E, "L": nominal.L,
        "r0": nominal.r0, "pr0": nominal.pr0,
        "mass0_kg": nominal.mass0_kg, "dry_mass_kg": nominal.dry_mass_kg,
        "tau_final": nominal.tau_final, "dt": nominal.dt,
        "record_every": nominal.record_every, "capture_r": nominal.capture_r,
        "wz": nominal.wz, "Ixx": nominal.Ixx, "Iyy": nominal.Iyy, "Izz": nominal.Izz,
        "tidal_enabled":  nominal.tidal_enabled,
        "tidal_model":    nominal.tidal_model,
        "thrust_enabled": nominal.thrust_enabled,
        "F_newton":       nominal.F_newton,
        "isp_s":          nominal.isp_s,
    }
    for d in dispersions:
        if not d.enabled or d.sigma <= 0 or d.param not in p:
            continue
        noise = rng.gauss(0, 1) * d.sigma
        p[d.param] = p[d.param] + noise if d.absolute else p[d.param] * (1.0 + noise)

    p["a"]  = max(0.0, min(0.9999 * p["M"], p["a"]))
    p["r0"] = max(p["capture_r"] * 1.5, p["r0"])
    p["dt"] = max(1e-4, p["dt"])
    return p


# ═══════════════════════════════════════════════════════════════════════════════
# ESTATÍSTICAS ONLINE (Welford — O(1) memória)
# ═══════════════════════════════════════════════════════════════════════════════

class WelfordStats:
    __slots__ = ("n", "_mu", "_M2")
    def __init__(self):           self.n=0; self._mu=0.; self._M2=0.
    def update(self, x: float):
        if not math.isfinite(x): return
        self.n += 1
        d = x - self._mu; self._mu += d/self.n; self._M2 += d*(x-self._mu)
    @property
    def mean(self): return self._mu
    @property
    def std(self):  return math.sqrt(max(0., self._M2/(self.n-1))) if self.n>1 else 0.


# ═══════════════════════════════════════════════════════════════════════════════
# STREAMING CSV  (escreve cada linha à medida que chega — thread-safe)
# ═══════════════════════════════════════════════════════════════════════════════

class StreamingCSV:
    """Abre ficheiro, escreve header, permite write() de MCResult, flush e close."""
    FLUSH_EVERY = 200

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._path = path
        self._f    = open(path, "w", newline="", encoding="utf-8")
        self._w    = csv.writer(self._f)
        self._lock = threading.Lock()
        self._n    = 0
        self._w.writerow(CSV_ALL_COLS)
        self._f.flush()
        self._closed = False

    def write(self, r: MCResult):
        if self._closed: return
        with self._lock:
            self._w.writerow(r.to_row())
            self._n += 1
            if self._n % self.FLUSH_EVERY == 0:
                self._f.flush()

    def close(self):
        with self._lock:
            if not self._closed:
                self._f.flush(); self._f.close()
                self._closed = True

    @property
    def path(self):         return self._path
    @property
    def rows_written(self): return self._n


# ═══════════════════════════════════════════════════════════════════════════════
# ESTADO GLOBAL DA CORRIDA MC
# ═══════════════════════════════════════════════════════════════════════════════

class MCState:
    def __init__(self):
        self.lock       = threading.Lock()
        self.running    = False
        self.stop_flag  = threading.Event()
        self.done       = 0
        self.total      = 0
        self.n_capture  = 0
        self.n_error    = 0
        self.start_time = 0.0

        self.stat_r    = WelfordStats()
        self.stat_eps  = WelfordStats()
        self.stat_qerr = WelfordStats()

        # Amostras para plot (máx 20 000 — sem guardar all_results)
        self.r_finals:    List[float] = []
        self.eps_vals:    List[float] = []
        self.conv_n:      List[int]   = []
        self.conv_mu_r:   List[float] = []
        self.conv_mu_eps: List[float] = []

        self.log_queue:  queue.Queue             = queue.Queue()
        self.csv_writer: Optional[StreamingCSV]  = None

    def reset(self):
        with self.lock:
            self.done = self.n_capture = self.n_error = 0
            self.stat_r = WelfordStats(); self.stat_eps = WelfordStats()
            self.stat_qerr = WelfordStats()
            self.r_finals.clear(); self.eps_vals.clear()
            self.conv_n.clear(); self.conv_mu_r.clear(); self.conv_mu_eps.clear()
        while not self.log_queue.empty():
            try: self.log_queue.get_nowait()
            except: pass

    def ingest(self, r: MCResult):
        with self.lock:
            self.done += 1
            # escrever CSV imediatamente (streaming)
            if self.csv_writer:
                self.csv_writer.write(r)
            if r.status == "CAPTURE":
                self.n_capture += 1
            elif r.status == "ERROR":
                self.n_error += 1
            else:
                self.stat_r.update(r.r_final)
                self.stat_eps.update(r.eps_rms)
                self.stat_qerr.update(r.qnorm_err)
                if len(self.r_finals) < 20_000:
                    self.r_finals.append(r.r_final)
                    self.eps_vals.append(r.eps_rms)

            interval = max(1, self.total // 200)
            if self.done % interval == 0:
                self.conv_n.append(self.done)
                self.conv_mu_r.append(self.stat_r.mean)
                self.conv_mu_eps.append(self.stat_eps.mean)


# ═══════════════════════════════════════════════════════════════════════════════
# THREAD COORDENADORA
# ═══════════════════════════════════════════════════════════════════════════════

def coordinator_thread(
    state:       MCState,
    nominal:     NominalParams,
    dispersions: List[DispersionSource],
    n_total:     int,
    n_workers:   int,
    seed:        int,
):
    """Corre num thread daemon. Usa wait(FIRST_COMPLETED) para drenar todos os N jobs."""
    rng  = random.Random(seed)
    jobs = [perturb(nominal, dispersions, rng) for _ in range(n_total)]

    # taxa de log: ~200 mensagens para toda a corrida
    log_every = max(1, n_total // 200)

    state.log_queue.put(("info",
        f"MC iniciado  N={n_total:,}  workers={n_workers}  seed={seed}"))
    active = [d for d in dispersions if d.enabled]
    state.log_queue.put(("info",
        "Dispersões: " + (
            "  ".join(f"{d.name}(σ={d.sigma})" for d in active)
            if active else "nenhuma")))
    state.log_queue.put(("info",
        ("Tidal: ON  modelo=" + nominal.tidal_model)
        if nominal.tidal_enabled else "Tidal: OFF"
        + "     "
        + (f"Thrust: ON  F={nominal.F_newton} N  Isp={nominal.isp_s} s"
           if nominal.thrust_enabled else "Thrust: OFF")))
    if state.csv_writer:
        state.log_queue.put(("info", f"CSV → {state.csv_writer.path}"))

    MAX_INFLIGHT = n_workers * PIPELINE_FACTOR

    try:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            pending: Dict[Any, int] = {}   # future → job_index
            submitted = 0

            # ── encher pipeline inicial ───────────────────────────────────────
            while submitted < n_total and len(pending) < MAX_INFLIGHT:
                f = pool.submit(_run_one, (submitted, jobs[submitted]))
                pending[f] = submitted
                submitted += 1

            # ── loop principal — drena TODOS os N jobs ────────────────────────
            while pending:
                if state.stop_flag.is_set():
                    for f in list(pending):
                        f.cancel()
                    state.log_queue.put(("warn",
                        f"Interrompido em {state.done:,}/{n_total:,}"))
                    break

                # timeout=2 s para checar stop_flag regularmente
                done_set, _ = wait(
                    list(pending.keys()), timeout=2.0,
                    return_when=FIRST_COMPLETED)

                for f in done_set:
                    job_idx = pending.pop(f)
                    try:
                        r = f.result(timeout=SIM_TIMEOUT_S)
                    except Exception as ex:
                        r = MCResult(
                            index=job_idx, status="ERROR",
                            r_final=float("nan"), r_min=float("nan"),
                            r_max=float("nan"), eps_rms=float("nan"),
                            qnorm_err=float("nan"), mass_consumed=0.0,
                            error_msg=f"timeout/ex: {str(ex)[:180]}")

                    state.ingest(r)

                    # ── log selectivo ─────────────────────────────────────────
                    if r.status == "ERROR":
                        state.log_queue.put(("error",
                            f"#{r.index:06d}  ERRO: {r.error_msg[:80]}"))
                    elif r.index % log_every == 0:
                        cls = "capture" if r.status == "CAPTURE" else "pass"
                        state.log_queue.put((cls,
                            f"#{r.index:06d}  {r.status:<8}"
                            f"  r={r.r_final:8.4f}M"
                            f"  ε={r.eps_rms:.2e}"
                            f"  ‖q‖={r.qnorm_err:.2e}"))

                    # ── repor pipeline ────────────────────────────────────────
                    while submitted < n_total and len(pending) < MAX_INFLIGHT:
                        nf = pool.submit(_run_one, (submitted, jobs[submitted]))
                        pending[nf] = submitted
                        submitted += 1

    except Exception as ex:
        state.log_queue.put(("error", f"Erro fatal no coordinator: {ex}"))

    finally:
        # ── fechar CSV + log resumo ───────────────────────────────────────────
        if state.csv_writer:
            try:
                state.csv_writer.close()
                state.log_queue.put(("done",
                    f"CSV fechado: {state.csv_writer.path}"
                    f"  ({state.csv_writer.rows_written:,} linhas)"))
            except Exception as ex:
                state.log_queue.put(("warn", f"Erro ao fechar CSV: {ex}"))

        state.running = False
        elapsed = time.time() - state.start_time
        sp = state.done / max(elapsed, 1e-6)
        state.log_queue.put(("sep", "─" * 66))
        state.log_queue.put(("done",
            f"CONCLUÍDO  {_fmt_time(elapsed)}"
            f"  ({state.done:,}/{n_total:,})"))
        state.log_queue.put(("info",
            f"PASS={state.stat_r.n:,}  "
            f"CAPTURA={state.n_capture:,}  "
            f"ERRO={state.n_error:,}"))
        state.log_queue.put(("info",
            f"r_final  μ={state.stat_r.mean:.5f} M"
            f"  σ={state.stat_r.std:.5f} M"))
        state.log_queue.put(("info",
            f"ε_rms    μ={state.stat_eps.mean:.3e}"
            f"  σ={state.stat_eps.std:.3e}"))
        state.log_queue.put(("info", f"Velocidade: {sp:.1f} sim/s"))
        state.log_queue.put(("sep", "─" * 66))


def _fmt_time(s: float) -> str:
    if not math.isfinite(s) or s < 0: return "—"
    h = int(s//3600); m = int((s%3600)//60); sc = int(s%60)
    return f"{h:02d}:{m:02d}:{sc:02d}"


# ═══════════════════════════════════════════════════════════════════════════════
# JANELA PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════════

class MCDashboard(tk.Tk):
    REFRESH_MS = 300    # actualização UI
    PLOT_MS    = 1500   # redesenho gráficos

    def __init__(self, default_n: int = 1000, default_workers: int = 4):
        super().__init__()
        self.title("KERR-MC  //  Monte Carlo · Kerr 6-DOF")
        self.configure(bg=BG)
        self.geometry("1540x940")
        self.minsize(1100, 700)

        self._state     = MCState()
        self._coord_th: Optional[threading.Thread] = None
        self._last_plot_t = 0.0
        self._current_tab = "hist_r"
        self._plot_pending = False

        self._build_ui(default_n, default_workers)
        self._schedule_refresh()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ── Detecção de fonte ─────────────────────────────────────────────────────

    def _best_mono(self) -> str:
        try:
            import tkinter.font as tkf
            for fam in ("JetBrains Mono","Consolas","Cascadia Code",
                        "Lucida Console","Courier New"):
                f = tkf.Font(family=fam, size=9)
                if "courier" not in f.actual("family").lower():
                    return fam
        except Exception:
            pass
        return "Courier New"

    # ── Construção da UI ──────────────────────────────────────────────────────

    def _build_ui(self, default_n: int, default_workers: int):
        fm = self._best_mono()
        F_T  = (fm, 7)            # secções / cabeçalhos
        F_B  = (fm, 9)            # body
        F_Bd = (fm, 9, "bold")    # valores
        F_L  = (fm,15, "bold")    # logo
        F_S  = (fm, 8)            # subtítulo
        F_Bt = (fm,10, "bold")    # botões
        F_Ml = (fm,12, "bold")    # métricas grandes
        F_Tb = (fm, 8, "bold")    # tabs

        self.columnconfigure(0, weight=0)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        # ── Painel esquerdo scrollável ────────────────────────────────────────
        outer = tk.Frame(self, bg=BG, width=336)
        outer.grid(row=0, column=0, sticky="nsew", padx=(8,0), pady=8)
        outer.grid_propagate(False)
        outer.rowconfigure(0, weight=1); outer.columnconfigure(0, weight=1)

        lc = tk.Canvas(outer, bg=BG, bd=0, highlightthickness=0, width=320)
        lc.grid(row=0, column=0, sticky="nsew")
        lsb = ttk.Scrollbar(outer, orient="vertical", command=lc.yview)
        lsb.grid(row=0, column=1, sticky="ns")
        lc.configure(yscrollcommand=lsb.set)

        left = tk.Frame(lc, bg=BG)
        _lw  = lc.create_window((0,0), window=left, anchor="nw")
        left.bind("<Configure>",
                  lambda e: lc.configure(scrollregion=lc.bbox("all")))
        lc.bind("<Configure>",
                lambda e: lc.itemconfig(_lw, width=e.width))
        lc.bind_all("<MouseWheel>",
                    lambda e: lc.yview_scroll(int(-e.delta/120), "units"))
        left.columnconfigure(0, weight=1)

        # ── helpers de layout ─────────────────────────────────────────────────
        def sep(r):
            tk.Frame(left, bg=BORDER, height=1).grid(
                row=r, column=0, sticky="ew", padx=12, pady=3)
            return r+1

        def sec(r, txt):
            tk.Label(left, text=txt, bg=BG, fg="#2a4050", font=F_T).grid(
                row=r, column=0, sticky="w", padx=12, pady=(6,2))
            return r+1

        def entry_row(frame, ri, label, key, default, fg=GREEN):
            tk.Label(frame, text=label, bg=BG, fg="#4a6878",
                     font=F_B, width=14, anchor="w").grid(
                row=ri, column=0, padx=(0,6), pady=2, sticky="w")
            var = tk.StringVar(value=default)
            self._pv[key] = var
            e = tk.Entry(frame, textvariable=var, bg="#0d1c26", fg=fg,
                         font=F_Bd, insertbackground=fg,
                         relief="flat", bd=0, width=10,
                         highlightthickness=1,
                         highlightcolor=fg, highlightbackground=BORDER)
            e.grid(row=ri, column=1, sticky="ew", pady=2, ipady=3)
            return ri+1

        self._pv: Dict[str, tk.Variable] = {}   # _param_vars

        row = 0

        # Logo
        tk.Label(left, text="KERR–MC", bg=BG, fg=CYAN, font=F_L).grid(
            row=row, column=0, sticky="w", padx=12, pady=(10,0)); row+=1
        tk.Label(left, text="Monte Carlo  ·  Kerr 6-DOF",
                 bg=BG, fg=MUTED, font=F_S).grid(
            row=row, column=0, sticky="w", padx=12, pady=(0,6)); row+=1

        # ── Parâmetros nominais ───────────────────────────────────────────────
        row = sep(row); row = sec(row, "PARÂMETROS NOMINAIS")
        pf = tk.Frame(left, bg=BG)
        pf.grid(row=row, column=0, sticky="ew", padx=12); row+=1
        pf.columnconfigure(1, weight=1)
        ri = 0
        for lbl, key, val in [
            ("M  [geom]",   "M",         "1.0"),
            ("a  [M]",      "a",         "0.5"),
            ("E  [mc²]",    "E",         "0.95"),
            ("L  [mM]",     "L",         "3.8"),
            ("r₀ [M]",      "r0",        "10.0"),
            ("τ_final [M]", "tau_final", "500.0"),
            ("dt [M]",      "dt",        "0.005"),
            ("massa [kg]",  "mass0_kg",  "1000.0"),
        ]:
            ri = entry_row(pf, ri, lbl, key, val)

        # ── Tidal ─────────────────────────────────────────────────────────────
        row = sep(row); row = sec(row, "TIDAL  ( maré gravitacional )")
        tf = tk.Frame(left, bg=BG)
        tf.grid(row=row, column=0, sticky="ew", padx=12); row+=1
        tf.columnconfigure(1, weight=1)

        self._tidal_en = tk.BooleanVar(value=False)
        tk.Checkbutton(tf, text="TIDAL  enabled",
                       variable=self._tidal_en,
                       bg=BG, fg=WHITE, selectcolor="#0d1c26",
                       activebackground=BG, activeforeground=GREEN,
                       font=F_B).grid(row=0, column=0, columnspan=2, sticky="w")

        tk.Label(tf, text="Modelo", bg=BG, fg="#4a6878",
                 font=F_B, width=14, anchor="w").grid(
            row=1, column=0, sticky="w", pady=2)
        self._tidal_model = tk.StringVar(value="DIAG_EIJ")
        ttk.Combobox(tf, textvariable=self._tidal_model,
                     values=["WEAK_N","DIAG_EIJ","RIEMANN_FD"],
                     state="readonly", width=13, font=F_B).grid(
            row=1, column=1, sticky="ew", pady=2)

        # ── Thrust ────────────────────────────────────────────────────────────
        row = sep(row); row = sec(row, "THRUST  ( propulsão )")
        thf = tk.Frame(left, bg=BG)
        thf.grid(row=row, column=0, sticky="ew", padx=12); row+=1
        thf.columnconfigure(1, weight=1)

        self._thrust_en = tk.BooleanVar(value=False)
        tk.Checkbutton(thf, text="THRUST  enabled",
                       variable=self._thrust_en,
                       bg=BG, fg=WHITE, selectcolor="#0d1c26",
                       activebackground=BG, activeforeground=GREEN,
                       font=F_B).grid(row=0, column=0, columnspan=2, sticky="w")
        ri2 = 1
        for lbl, key, val in [
            ("F_newton [N]", "F_newton", "10.0"),
            ("Isp [s]",      "isp_s",    "3000.0"),
        ]:
            ri2 = entry_row(thf, ri2, lbl, key, val, fg=AMBER)

        # ── Dispersões ────────────────────────────────────────────────────────
        row = sep(row); row = sec(row, "FONTES DE INCERTEZA  ( σ relativo )")
        df = tk.Frame(left, bg=BG)
        df.grid(row=row, column=0, sticky="ew", padx=12); row+=1
        df.columnconfigure(1, weight=1)

        self._disp_en:    List[tk.BooleanVar] = []
        self._disp_sigma: List[tk.StringVar]  = []
        for i, d in enumerate(DEFAULT_DISPERSIONS):
            ev = tk.BooleanVar(value=d.enabled)
            sv = tk.StringVar(value=str(d.sigma))
            self._disp_en.append(ev); self._disp_sigma.append(sv)
            tk.Checkbutton(df, variable=ev, text=d.name,
                           bg=BG, fg="#c0d8e8", selectcolor="#0d1c26",
                           activebackground=BG, activeforeground=GREEN,
                           font=F_B, anchor="w", pady=1).grid(
                row=i, column=0, sticky="w")
            tk.Entry(df, textvariable=sv, bg="#0d1c26", fg=AMBER,
                     font=F_Bd, width=7, relief="flat", bd=0,
                     highlightthickness=1,
                     highlightcolor=AMBER, highlightbackground=BORDER).grid(
                row=i, column=1, sticky="e", padx=(4,0), ipady=2)

        # ── Configuração da corrida ────────────────────────────────────────────
        row = sep(row); row = sec(row, "CONFIGURAÇÃO DA CORRIDA")
        cf = tk.Frame(left, bg=BG)
        cf.grid(row=row, column=0, sticky="ew", padx=12); row+=1
        cf.columnconfigure(1, weight=1)
        ri3 = 0
        for lbl, key, val in [
            ("N simulações",  "n_sim",    str(default_n)),
            ("Workers (CPU)", "n_workers",str(default_workers)),
            ("Seed RNG",      "seed",     "42"),
        ]:
            ri3 = entry_row(cf, ri3, lbl, key, val, fg=CYAN)

        nf = tk.Frame(left, bg=BG)
        nf.grid(row=row, column=0, sticky="ew", padx=12, pady=(4,0)); row+=1
        for nv in [100, 1_000, 10_000, 100_000]:
            tk.Button(nf, text=f"{nv:,}", bg="#0d1c26", fg="#4a6878",
                      font=F_T, relief="flat", bd=0,
                      activebackground=BORDER, activeforeground=CYAN,
                      command=lambda v=nv: self._pv["n_sim"].set(str(v)),
                      padx=8, pady=4).pack(side="left", padx=2)

        row = sep(row)

        # Botões acção
        bf = tk.Frame(left, bg=BG)
        bf.grid(row=row, column=0, sticky="ew", padx=12); row+=1
        bf.columnconfigure(0, weight=1); bf.columnconfigure(1, weight=1)

        self._run_btn = tk.Button(bf, text="▶   INICIAR",
            bg="#002d1f", fg=GREEN, font=F_Bt, relief="flat", bd=0,
            activebackground="#004030", activeforeground=GREEN,
            command=self._toggle_run, pady=11)
        self._run_btn.grid(row=0, column=0, sticky="ew", padx=(0,4))

        self._stop_btn = tk.Button(bf, text="■   PARAR",
            bg="#2a0000", fg=RED, font=F_Bt, relief="flat", bd=0,
            activebackground="#460000", activeforeground=RED,
            command=self._stop, state="disabled", pady=11)
        self._stop_btn.grid(row=0, column=1, sticky="ew")

        self._export_btn = tk.Button(left, text="↓   EXPORTAR CÓPIA CSV",
            bg="#0d1c26", fg="#3a5060", font=F_B,
            relief="flat", bd=0, command=self._export_copy_async, pady=7)
        self._export_btn.grid(row=row, column=0, sticky="ew",
                              padx=12, pady=(6,0)); row+=1

        self._status_lbl = tk.Label(left, text="●  IDLE",
            bg=BG, fg=MUTED, font=F_B, anchor="w")
        self._status_lbl.grid(row=row, column=0, sticky="w",
                              padx=12, pady=(4,10)); row+=1

        # ── Painel direito ────────────────────────────────────────────────────
        right = tk.Frame(self, bg=BG)
        right.grid(row=0, column=1, sticky="nsew", padx=8, pady=8)
        right.columnconfigure(0, weight=1)
        right.rowconfigure(2, weight=1)

        # Barra de progresso
        prf = tk.Frame(right, bg="#0c1620", pady=10)
        prf.grid(row=0, column=0, sticky="ew"); prf.columnconfigure(1, weight=1)
        tk.Label(prf, text="PROGRESSO", bg="#0c1620", fg="#2a4050",
                 font=F_T, width=11).grid(row=0, column=0, padx=10)
        sty = ttk.Style(); sty.theme_use("default")
        sty.configure("MC.Horizontal.TProgressbar",
                       troughcolor="#0d1c26", background=CYAN,
                       borderwidth=0, lightcolor=CYAN, darkcolor=CYAN)
        self._pbar = ttk.Progressbar(prf, orient="horizontal",
                                     mode="determinate",
                                     style="MC.Horizontal.TProgressbar")
        self._pbar.grid(row=0, column=1, sticky="ew", padx=8)
        self._pct_lbl = tk.Label(prf, text="0.00%",
            bg="#0c1620", fg=CYAN, font=F_Ml, width=8)
        self._pct_lbl.grid(row=0, column=2, padx=10)

        # Métricas
        mf = tk.Frame(right, bg=BG)
        mf.grid(row=1, column=0, sticky="ew", pady=(6,4))
        self._mv: Dict[str, tk.StringVar] = {}
        for ci, (lbl, key, col) in enumerate([
            ("Concluídas", "done",    CYAN),
            ("PASS",       "pass",    GREEN),
            ("CAPTURA",    "cap",     RED),
            ("Velocidade", "speed",   AMBER),
            ("Decorrido",  "elapsed", WHITE),
            ("ETA",        "eta",     AMBER),
            ("r_final  μ", "rfmu",    CYAN),
            ("r_final  σ", "rfsg",    AMBER),
            ("ε_rms  μ",   "epsmu",   CYAN),
            ("‖q‖ err  μ", "qmu",     AMBER),
        ]):
            cell = tk.Frame(mf, bg="#0c1620", padx=2)
            cell.grid(row=0, column=ci, padx=2, sticky="nsew")
            mf.columnconfigure(ci, weight=1)
            tk.Label(cell, text=lbl, bg="#0c1620", fg="#2a4050",
                     font=F_T, anchor="w").pack(anchor="w", padx=8, pady=(5,0))
            var = tk.StringVar(value="—")
            self._mv[key] = var
            tk.Label(cell, textvariable=var, bg="#0c1620", fg=col,
                     font=F_Ml, anchor="w").pack(anchor="w", padx=8, pady=(1,5))

        # Bottom: log + plots
        bot = tk.Frame(right, bg=BG)
        bot.grid(row=2, column=0, sticky="nsew")
        bot.columnconfigure(0, weight=0); bot.columnconfigure(1, weight=1)
        bot.rowconfigure(0, weight=1)

        # Log
        lf = tk.Frame(bot, bg=BG, width=390)
        lf.grid(row=0, column=0, sticky="nsew", padx=(0,6))
        lf.grid_propagate(False)
        lf.rowconfigure(1, weight=1); lf.columnconfigure(0, weight=1)
        tk.Label(lf, text="LOG  EM  TEMPO  REAL",
                 bg=BG, fg="#2a4050", font=F_T).grid(
            row=0, column=0, sticky="w", pady=(0,2))
        self._log = tk.Text(lf, bg="#0c1620", fg=WHITE,
            font=F_B, relief="flat", bd=0,
            state="disabled", wrap="none",
            selectbackground=BORDER, spacing1=1, spacing3=1)
        self._log.grid(row=1, column=0, sticky="nsew")
        lsb2 = ttk.Scrollbar(lf, command=self._log.yview)
        lsb2.grid(row=1, column=1, sticky="ns")
        self._log["yscrollcommand"] = lsb2.set
        for tag, col in [("info","#2a4050"),("pass",GREEN),("capture",RED),
                          ("error",RED),("warn",AMBER),("done",CYAN),
                          ("sep","#1a2a36"),("default",WHITE)]:
            self._log.tag_config(tag, foreground=col)
        self._log_lines = 0

        # Plots
        plotf = tk.Frame(bot, bg=BG)
        plotf.grid(row=0, column=1, sticky="nsew")
        plotf.rowconfigure(1, weight=1); plotf.columnconfigure(0, weight=1)

        tbar = tk.Frame(plotf, bg="#0c1620")
        tbar.grid(row=0, column=0, sticky="ew")
        self._tab_btns: Dict[str, tk.Button] = {}
        for t_key, t_lbl in [
            ("hist_r",   "HIST  r_final"),
            ("hist_eps", "HIST  ε_rms"),
            ("scatter",  "DISPERSÃO"),
            ("conv",     "CONVERGÊNCIA"),
        ]:
            btn = tk.Button(tbar, text=t_lbl, bg="#0c1620", fg="#2a4050",
                            font=F_Tb, relief="flat", bd=0, padx=14, pady=6,
                            activebackground=BORDER, activeforeground=CYAN,
                            command=lambda k=t_key: self._switch_tab(k))
            btn.pack(side="left", padx=1)
            self._tab_btns[t_key] = btn

        self._fig = plt.figure(figsize=(8,5), dpi=96)
        self._ax  = self._fig.add_subplot(111)
        self._canvas_mpl = FigureCanvasTkAgg(self._fig, master=plotf)
        self._canvas_mpl.get_tk_widget().grid(row=1, column=0, sticky="nsew")

        self._switch_tab("hist_r")
        self._draw_empty()

    # ── Helpers UI ────────────────────────────────────────────────────────────

    def _switch_tab(self, key: str):
        self._current_tab = key
        for k, b in self._tab_btns.items():
            b.config(fg=CYAN if k==key else MUTED,
                     bg=BG2 if k==key else BG3)
        self._plot_pending = True

    def _log_write(self, tag: str, msg: str):
        self._log.config(state="normal")
        ts = time.strftime("%H:%M:%S")
        self._log.insert("end", f"[{ts}]  {msg}\n", tag)
        self._log_lines += 1
        if self._log_lines > 4000:
            self._log.delete("1.0", "250.0")
            self._log_lines -= 250
        self._log.see("end")
        self._log.config(state="disabled")

    def _draw_empty(self):
        self._ax.clear()
        self._ax.text(0.5, 0.5, "AGUARDANDO  DADOS",
                      transform=self._ax.transAxes,
                      ha="center", va="center",
                      color=MUTED, fontsize=13, fontfamily="monospace")
        self._ax.set_xticks([]); self._ax.set_yticks([])
        self._canvas_mpl.draw_idle()

    def _redraw_plots(self):
        """Redesenha o gráfico activo. Sem tight_layout() (caro)."""
        self._plot_pending = False
        st = self._state
        if not st.r_finals and not st.conv_n:
            self._draw_empty(); return
        try:
            self._ax.clear()
            tab = self._current_tab

            if tab == "hist_r" and st.r_finals:
                data = np.array(st.r_finals)
                nb   = min(80, max(20, int(np.sqrt(len(data)))))
                self._ax.hist(data, bins=nb, color=CYAN, alpha=0.75,
                              edgecolor="none", label=f"n={len(data):,}")
                self._ax.axvline(st.stat_r.mean, color=AMBER, lw=1.5,
                                 linestyle="--",
                                 label=f"μ={st.stat_r.mean:.4f} M")
                self._ax.axvspan(st.stat_r.mean - st.stat_r.std,
                                 st.stat_r.mean + st.stat_r.std,
                                 alpha=0.10, color=AMBER)
                self._ax.set_xlabel("r_final [M]")
                self._ax.set_ylabel("frequência")
                self._ax.set_title(
                    f"r_final  μ={st.stat_r.mean:.4f}  σ={st.stat_r.std:.4f}  n={st.stat_r.n:,}",
                    color=WHITE)
                self._ax.legend(fontsize=8)

            elif tab == "hist_eps" and st.eps_vals:
                data = np.array(st.eps_vals)
                data = data[np.isfinite(data) & (data > 0)]
                if len(data) > 1:
                    nb = min(80, max(20, int(np.sqrt(len(data)))))
                    self._ax.hist(np.log10(data), bins=nb,
                                  color=GREEN, alpha=0.75, edgecolor="none")
                    self._ax.set_xlabel("log₁₀(ε_rms)")
                    self._ax.set_ylabel("frequência")
                    self._ax.set_title(
                        f"ε_rms  μ={st.stat_eps.mean:.3e}"
                        f"  σ={st.stat_eps.std:.3e}",
                        color=WHITE)

            elif tab == "scatter" and len(st.r_finals) > 1:
                n  = min(len(st.r_finals), len(st.eps_vals))
                ra = np.array(st.r_finals[:n])
                ea = np.array(st.eps_vals[:n])
                ok = np.isfinite(ra) & np.isfinite(ea) & (ea > 0)
                alpha = float(np.clip(300 / max(ok.sum(), 1), 0.02, 0.55))
                self._ax.scatter(ra[ok], ea[ok], s=2, alpha=alpha,
                                 color=CYAN, linewidths=0)
                self._ax.set_xlabel("r_final [M]")
                self._ax.set_ylabel("ε_rms")
                self._ax.set_yscale("log")
                self._ax.set_title("Dispersão: r_final vs ε_rms", color=WHITE)

            elif tab == "conv" and st.conv_n:
                self._ax.plot(st.conv_n, st.conv_mu_r,
                              color=GREEN, lw=1.5, label="μ(r_final)")
                if any(v > 0 for v in st.conv_mu_eps):
                    ax2 = self._ax.twinx()
                    ax2.plot(st.conv_n, st.conv_mu_eps,
                             color=AMBER, lw=1.0, linestyle="--",
                             label="μ(ε_rms)")
                    ax2.set_ylabel("μ(ε_rms)", color=AMBER)
                    ax2.tick_params(colors=AMBER)
                    ax2.set_yscale("log")
                    l1, lb1 = self._ax.get_legend_handles_labels()
                    l2, lb2 = ax2.get_legend_handles_labels()
                    self._ax.legend(l1+l2, lb1+lb2, fontsize=8)
                self._ax.set_xlabel("n simulações")
                self._ax.set_ylabel("μ(r_final) [M]", color=GREEN)
                self._ax.set_title("Convergência das médias", color=WHITE)
            else:
                self._draw_empty(); return

            self._ax.grid(True, alpha=0.3)
            self._canvas_mpl.draw_idle()   # não usa tight_layout — mais rápido
        except Exception:
            pass  # nunca quebrar a UI por causa de plots

    # ── Leitura de parâmetros ─────────────────────────────────────────────────

    def _read_nominal(self) -> NominalParams:
        def f(k, d=0.0):
            try: return float(self._pv[k].get())
            except: return d
        return NominalParams(
            M=f("M",1.), a=f("a",.5), E=f("E",.95), L=f("L",3.8),
            r0=f("r0",10.), pr0=0.,
            mass0_kg=f("mass0_kg",1000.), dry_mass_kg=300.,
            tau_final=f("tau_final",500.), dt=f("dt",.005),
            record_every=50, capture_r=2., wz=.02,
            Ixx=100., Iyy=200., Izz=150.,
            tidal_enabled=self._tidal_en.get(),
            tidal_model=self._tidal_model.get(),
            thrust_enabled=self._thrust_en.get(),
            F_newton=f("F_newton",10.),
            isp_s=f("isp_s",3000.),
        )

    def _read_dispersions(self) -> List[DispersionSource]:
        out = []
        for i, d in enumerate(DEFAULT_DISPERSIONS):
            try: sg = float(self._disp_sigma[i].get())
            except: sg = d.sigma
            out.append(DispersionSource(
                name=d.name, param=d.param, sigma=sg,
                enabled=self._disp_en[i].get(),
                absolute=d.absolute))
        return out

    # ── Controlo de corrida ───────────────────────────────────────────────────

    def _toggle_run(self):
        if self._state.running: self._stop()
        else:                   self._start()

    def _start(self):
        if self._state.running: return
        try:
            n_sim     = int(self._pv["n_sim"].get())
            n_workers = int(self._pv["n_workers"].get())
            seed      = int(self._pv["seed"].get())
        except ValueError:
            messagebox.showerror("Erro",
                "N simulações, workers e seed devem ser inteiros.")
            return

        nominal     = self._read_nominal()
        dispersions = self._read_dispersions()

        self._state.reset()
        self._state.total      = n_sim
        self._state.running    = True
        self._state.start_time = time.time()
        self._state.stop_flag.clear()

        # ── abrir CSV com timestamp ───────────────────────────────────────────
        ts_str = time.strftime("%Y%m%d_%H%M%S")
        csv_path = AUTOSAVE_DIR / f"mc_{ts_str}_N{n_sim}.csv"
        self._state.csv_writer = StreamingCSV(csv_path)

        # limpar log
        self._log.config(state="normal")
        self._log.delete("1.0", "end")
        self._log.config(state="disabled")
        self._log_lines = 0

        self._run_btn.config(state="disabled")
        self._stop_btn.config(state="normal")
        self._status_lbl.config(text="●  RUNNING", fg=GREEN)

        self._coord_th = threading.Thread(
            target=coordinator_thread,
            args=(self._state, nominal, dispersions,
                  n_sim, n_workers, seed),
            daemon=True)
        self._coord_th.start()

    def _stop(self):
        self._state.stop_flag.set()
        self._state.running = False
        self._run_btn.config(state="normal")
        self._stop_btn.config(state="disabled")
        self._status_lbl.config(text="●  STOPPED", fg=AMBER)

    # ── Refresh periódico (thread Tk) ─────────────────────────────────────────

    def _schedule_refresh(self):
        self.after(self.REFRESH_MS, self._refresh)

    def _refresh(self):
        st = self._state

        # drenar fila de log (máx 80 msg/ciclo — não trava UI)
        for _ in range(80):
            try:
                tag, msg = st.log_queue.get_nowait()
                self._log_write(tag, msg)
            except queue.Empty:
                break

        # barra de progresso
        if st.total > 0:
            pct = st.done / st.total
            self._pbar["value"] = pct * 100
            self._pct_lbl.config(text=f"{pct*100:.2f}%")

        elapsed = time.time() - st.start_time if st.start_time else 0.
        speed   = st.done / max(elapsed, 1e-6)
        eta     = (st.total - st.done) / max(speed, 1e-6) if st.done > 0 else float("inf")

        mv = self._mv
        mv["done"].set(f"{st.done:,} / {st.total:,}")
        mv["pass"].set(f"{st.stat_r.n:,}")
        mv["cap"].set(f"{st.n_capture:,}  ({st.n_capture/max(st.done,1)*100:.1f}%)")
        mv["speed"].set(f"{speed:.1f} sim/s")
        mv["elapsed"].set(_fmt_time(elapsed))
        mv["eta"].set(_fmt_time(eta) if st.running else "—")
        mv["rfmu"].set(f"{st.stat_r.mean:.4f} M" if st.stat_r.n  else "—")
        mv["rfsg"].set(f"{st.stat_r.std:.4f} M"  if st.stat_r.n>1 else "—")
        mv["epsmu"].set(f"{st.stat_eps.mean:.3e}" if st.stat_eps.n else "—")
        mv["qmu"].set(f"{st.stat_qerr.mean:.3e}"  if st.stat_qerr.n else "—")

        # reactivar botão quando thread acabar
        if not st.running and self._run_btn["state"] == "disabled":
            self._run_btn.config(state="normal")
            self._stop_btn.config(state="disabled")
            self._status_lbl.config(text="●  DONE", fg=CYAN)

        # redesenho de plots (limitado por tempo + flag)
        now = time.time()
        if self._plot_pending or (now - self._last_plot_t > self.PLOT_MS/1000.):
            self._redraw_plots()
            self._last_plot_t = now

        self._schedule_refresh()

    # ── Exportar cópia CSV (não bloqueia Tk) ─────────────────────────────────

    def _export_copy_async(self):
        """Copia o CSV streaming para um destino à escolha — em thread separada."""
        cw = self._state.csv_writer
        if cw is None or not cw.path.exists():
            messagebox.showinfo(
                "Exportar",
                "Nenhum CSV disponível ainda.\n"
                "Inicia uma simulação — o ficheiro é criado automaticamente.")
            return

        dest = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV","*.csv"),("Todos","*.*")],
            initialfile="mc_kerr_copy.csv")
        if not dest:
            return

        src = cw.path

        def _copy():
            try:
                shutil.copy2(src, dest)
                self._state.log_queue.put(("done",
                    f"Cópia exportada → {dest}"))
            except Exception as ex:
                self._state.log_queue.put(("error",
                    f"Erro na cópia: {ex}"))

        threading.Thread(target=_copy, daemon=True).start()

    # ── Fechar ────────────────────────────────────────────────────────────────

    def _on_close(self):
        self._state.stop_flag.set()
        self._state.running = False
        if self._state.csv_writer:
            try: self._state.csv_writer.close()
            except: pass
        self.destroy()


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Monte Carlo Dashboard — Kerr 6-DOF")
    parser.add_argument("--n",       type=int, default=1000,
                        help="N simulações por defeito")
    parser.add_argument("--workers", type=int, default=4,
                        help="Workers paralelos")
    args = parser.parse_args()

    # Windows spawn-safety
    from multiprocessing import freeze_support
    freeze_support()

    app = MCDashboard(default_n=args.n, default_workers=args.workers)
    app.mainloop()


if __name__ == "__main__":
    main()
