# src/relorbit_py/target_escape.py
import numpy as np
import relorbit_py as rp
from relorbit_py.simulate import simulate_case
import matplotlib.pyplot as plt
import os

def find_minimum_escape_thrust(
    model="kerr_lowthrust",
    a=0.9,
    r0=10.0,
    target_r=50.0,
    tol=1e-6,
    max_iter=15
):
    print(f"\n==> Iniciando Busca de Empuxo Mínimo (Targeting)")
    print(f"    Modelo: {model} | a={a} | r0={r0}M -> alvo={target_r}M")

    # Limites da busca binária para F_phi [unidades geométricas]
    low = 0.0
    high = 1e-3  # Um valor alto o suficiente para garantir escape
    best_f = high

    history = []

    for i in range(max_iter):
        mid = (low + high) / 2
        
        # Configuração temporária para o teste
        case = {
            "name": f"test_f_{mid:.2e}",
            "model": model,
            "params": {"M": 1.0, "a": a, "E": 0.8932, "L": 3.1416, "capture_r": 1.5},
            "state0": [r0, 0.0],
            "pr0": 0.0,
            "span": [0.0, 3000.0],
            "solver": {"dt": 0.05, "record_every": 20},
            "thrust": {
                "F_phi": mid,
                "F_r": 0.0,
                "isp_s": 3000.0,
                "mass0_kg": 1000.0,
                "dry_mass_kg": 300.0,
                "mode": "TANGENTIAL_ONLY"
            }
        }

        traj = simulate_case(case, model)
        final_r = traj.r[-1]
        status = str(traj.status)

        # Critério de sucesso: atingiu o raio alvo e não foi capturado
        escaped = (final_r >= target_r) and ("CAPTURE" not in status)
        
        history.append((mid, escaped, final_r))
        print(f"    Iter {i+1:02d}: F_phi={mid:.2e} | R_final={final_r:.2f} | Escape: {escaped}")

        if escaped:
            best_f = mid
            high = mid # Tenta um valor ainda menor
        else:
            low = mid # Precisa de mais empuxo

        if (high - low) < tol:
            break

    return best_f, history

def plot_targeting_results(history, outdir="out/missions"):
    os.makedirs(outdir, exist_ok=True)
    fs = [h[0] for h in history]
    rs = [h[2] for h in history]
    
    plt.figure(figsize=(10, 5))
    plt.semilogx(fs, rs, 'o-', label="Raio Final Alcançado")
    plt.axhline(50.0, color='red', linestyle='--', label="Alvo de Escape")
    plt.title("Shooting Method: Busca do Limiar de Escape")
    plt.xlabel("Empuxo Tangencial F_phi [geom]")
    plt.ylabel("Raio Final [M]")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.savefig(os.path.join(outdir, "targeting_convergence.png"))
    print(f"\n==> Gráfico de convergência salvo em {outdir}")

if __name__ == "__main__":
    best_thrust, hist = find_minimum_escape_thrust()
    print(f"\n[SUCESSO] Empuxo mínimo encontrado: F_phi = {best_thrust:.6e}")
    plot_targeting_results(hist)