# Kerr 6-DOF com Torque de Maré — Documentação de Implementação

## Visão Geral

Este módulo implementa o acoplamento completo **Kerr 6-DOF + Atitude + Torque de Maré (Spaghettification)**
para o projecto `relorbit`. É uma extensão directa do modelo `schwarzschild_6dof` existente,
usando as equações orbitais Kerr já presentes em `kerr_lowthrust`.

---

## Ficheiros Novos / Alterados

| Ficheiro | Descrição |
|---|---|
| `src_cpp/include/relorbit/gr/kerr_metric.hpp` | Métrica Kerr, tetrada ZAMO, tensor de maré (3 níveis) |
| `src_cpp/lib/gr/kerr_metric.cpp` | Implementação: Christoffel analítico, Riemann FD, E_ij |
| `src_cpp/include/relorbit/models/kerr_6dof.hpp` | Header do integrador 6-DOF Kerr |
| `src_cpp/lib/kerr_6dof.cpp` | Integrador RK4 14D acoplado |
| `src_cpp/bindings/pybind_module.cpp` | + bindings TidalCfg, AttitudeCfgKerr, TrajectoryCoupledKerr |
| `src/relorbit_py/simulate_kerr_6dof.py` | Mission runner Python + plots |
| `src/relorbit_py/run_mission.py` | + modelo `kerr_6dof` no runner principal |
| `kerr_6dof_cases.yaml` | 6 casos de teste cobrindo todos os modos |

---

## Estado (14D)

```
y = [r, φ, pr, E, L, m_kg,  q0, q1, q2, q3,  ωx, ωy, ωz,  (pad)]
     0  1   2  3  4   5      6   7   8   9     10  11  12    13
```

---

## Modelos de Torque de Maré

### WEAK_N (gravity-gradient clássico)
```
τ_body = 3(M/r³) · n_body × (I · n_body)
n_body = Rᵀ(q) · [1,0,0]ᵀ   (direcção radial no body frame)
```
Rápido, útil como sanity check. Limite correcto para r → ∞.

### DIAG_EIJ (quadrupolo + E_ij analítico)
```
E_local = diag(−2M/r³, +M/r³, +M/r³)   [r̂, θ̂, φ̂]
E_body  = Rᵀ · E_local · R
Q_body  = I − (trI/3) · I₃
τ_body  = −axial(Q·E − E·Q)
```
Com `spin_correction=true`, adiciona termo off-diagonal `E_{rφ} ~ −3Ma/r⁴`.

### RIEMANN_FD (modo monstro — tensor de maré exacto)
Cadeia completa:
1. `g_{μν}(r)` analítico em Boyer-Lindquist equatorial
2. `∂_r g_{μν}` analítico → `Γ^μ_{αβ}` analítico
3. `∂_r Γ^μ_{αβ}` por diferença finita centrada (passo `fd_eps_r`)
4. `R^μ_{νρσ}` = quadrático em Γ + ∂Γ
5. Projecção na tetrada ZAMO → `E_{îĵ} = R_{î 0̂ ĵ 0̂}`
6. Torque via quadrupolo: `τ = −axial(Q·E_body − E_body·Q)`

**Log de convergência** impresso automaticamente ao fim de cada missão RIEMANN_FD:
```
[KERR_6DOF] RIEMANN_FD convergência @ r=50M: err(eps)=X err(eps/2)=Y ratio=Z (esperado ~4)
```
Ratio ≈ 4 confirma convergência de 2ª ordem do esquema centrado.

---

## Convenção de Índices E_ij

```
E_local[i,j]:   0=r̂, 1=θ̂, 2=φ̂
E_local[0,0] = E_{r̂r̂}  ≈ −2M/r³  (compressão)
E_local[1,1] = E_{θ̂θ̂}  ≈ +M/r³   (estiramento)
E_local[2,2] = E_{φ̂φ̂}  ≈ +M/r³   (estiramento)
```

---

## Telemetria Extra (TrajectoryCoupledKerr)

| Campo | Descrição |
|---|---|
| `tidal_tau_x/y/z` | Componentes do torque de maré no body frame [N·m] |
| `tidal_norm` | `|τ_tidal|` |
| `align_angle_rad` | Ângulo entre eixo x_body e direcção radial local |
| `tidal_E_norm` | Norma de Frobenius de `E_ij` (diagnóstico) |
| `qnorm` | `‖q‖` (deve ser 1 ± renorm_tol) |
| `T_rot` | Energia cinética rotacional [J] |
| `epsilon` | Desvio da constraint Carter (≈0 geodésica) |

---

## Uso via YAML

```yaml
missions:
  - name: Kerr_ModoMonstro
    model: kerr_6dof
    params: {M: 1.0, a: 0.5, E: 0.95, L: 3.8}
    state0: [10.0, 0.0]
    pr0: 0.0
    attitude0: {q0: 0.707, q1: 0.707, q2: 0.0, q3: 0.0, wx: 0.0, wy: 0.01, wz: 0.0}
    inertia:   {Ixx: 50.0, Iyy: 200.0, Izz: 150.0}
    engine:    {F_newton: 0.0, isp_s: 3000.0, mass0_kg: 1000.0, dry_mass_kg: 300.0}
    ext_torque: {tx: 0.0, ty: 0.0, tz: 0.0}
    tidal:
      enabled: true
      model: RIEMANN_FD    # ou WEAK_N | DIAG_EIJ
      fd_eps_r: 1.0e-5
      Q_from_inertia: true
      spin_correction: false
    span: [0.0, 300.0]
    solver: {dt: 0.005, record_every: 20}
```

```bash
python -m relorbit_py.run_mission --yaml kerr_6dof_cases.yaml --out out/missions
```

---

## Validações

| Teste | Critério |
|---|---|
| `kerr_6dof_tidal_baseline` | T_rot = const (drift < 1e-6 relativo) |
| `kerr_6dof_riemann_fd_weakfield` | `E_ij(FD)` ≈ `diag(−2M/r³, M/r³, M/r³)` com erro < 1% @ r=50M |
| Convergência FD | `ratio ≈ 4` (FD centrada 2ª ordem) |
| `‖q‖ = 1` | `max|‖q‖ − 1| < 1e-6` após renormalização |

---

## Ficheiros em Falta no Contexto (TODOs)

Os seguintes ficheiros não foram recebidos nos uploads e podem precisar de ajuste:

- `src/relorbit_py/attitude_mission.py` — usado em `run_mission.py` para model=`attitude`
- `relorbit/types.hpp` — deve definir `OrbitStatus`, `SolverCfg`, `Maneuver`
- `relorbit/api.hpp` — API de Newton etc.
- `src/relorbit_py/simulate.py`, `mission.py`, `plots_*.py` — wrappers de missão

Estes são referenciados no código existente e não requerem modificação para o Kerr 6-DOF funcionar.
O novo modelo é **aditivo** e não altera os módulos existentes.