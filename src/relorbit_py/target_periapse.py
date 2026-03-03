# src/relorbit_py/target_periapse.py
import numpy as np
import relorbit_py as rp
from relorbit_py.simulate import simulate_case
import os

def find_periapse_maneuver(r0=20.0, target_rp=3.0, tol=1e-4):
    """
    Shooting Method (Bisseção) para encontrar o dv_phi (kick tangencial)
    necessário para atingir um periapse alvo em Schwarzschild.
    """
    print(f"\n{'='*60}")
    print(f"==> PLANEJADOR DE MISSÃO: TARGETING NUMÉRICO")
    print(f"{'='*60}")
    print(f"    Origem: r = {r0}M")
    print(f"    Alvo (Periapse): r = {target_rp}M")
    print(f"    Tolerância: {tol*100}%")

    # Limites da busca binária para o kick tangencial (dv_phi)
    # dv_phi < 0 reduz o momento angular L e faz a sonda descer.
    low = -2.0  
    high = 0.0
    best_dv = 0.0

    for i in range(20): # Máximo de 20 iterações para alta precisão
        mid = (low + high) / 2
        
        # Caso de teste baseado no seu YAML (E e L circulares para r=20M)
        case = {
            "name": f"target_iter_{i}",
            "model": "schwarzschild_equatorial",
            "params": {"M": 1.0, "E": 0.974, "L": 4.3}, 
            "state0": [r0, 0.0],
            "pr0": 0.0,
            "span": [0.0, 1000.0],
            "solver": {
                "dt": 0.05,
                "maneuvers": [{"tau": 0.1, "dv_phi": mid}]
            }
        }

        # Rodamos a simulação via simulate_case
        traj = simulate_case(case, "schwarzschild_equatorial")
        
        # O periapse é o raio mínimo alcançado na trajetória
        current_rp = min(traj.r)
        error = abs(current_rp - target_rp) / target_rp

        print(f"    Iter {i+1:02d}: dv_phi={mid:.6f} | Periapse={current_rp:.4f} | Erro={error*100:.4f}%")

        if current_rp > target_rp:
            # Se o periapse está acima do alvo, precisamos tirar mais L (diminuir dv_phi)
            high = mid 
        else:
            # Se o periapse está abaixo do alvo, tiramos menos L
            low = mid
            best_dv = mid

        if error < tol:
            print(f"\n[SUCESSO] Alvo atingido com precisão de {error*100:.5f}%")
            break

    return best_dv

if __name__ == "__main__":
    # Executa a busca para o cenário do Item 6: r=20 -> r_p=3
    best_dv = find_periapse_maneuver(r0=20.0, target_rp=3.0)
    print(f"\n>>> Sugestão de Manobra para o YAML: dv_phi = {best_dv:.6f}")