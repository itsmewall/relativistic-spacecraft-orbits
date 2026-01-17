# relorbit — Schwarzschild (Plano Equatorial) + Validação

Este repositório implementa e valida um motor numérico (C++ com bindings em Python via `pybind11`) para simulação de órbitas no problema de dois corpos (Newtoniano plano) e, nesta etapa, geodésicas no espaço-tempo de Schwarzschild restritas ao plano equatorial.

A filosofia do projeto é **validar primeiro** (invariantes e constraints), antes de avançar para modelos mais complexos.

---

## 1. O que foi implementado nesta etapa (Schwarzschild)

### 1.1 Modelo físico
Trabalhamos com geodésicas de uma partícula teste no espaço-tempo de Schwarzschild, no **plano equatorial** (θ = π/2) e em **unidades geométricas** (G = c = 1).

No formalismo padrão, existem duas constantes de movimento:

- Energia específica: **E**
- Momento angular específico: **L**

O movimento radial pode ser escrito por uma equação de “potencial efetivo” na forma:

\[
\left(\frac{dr}{d\tau}\right)^2 = E^2 - \left(1-\frac{2M}{r}\right)\left(1+\frac{L^2}{r^2}\right)
\]

onde:
- M é a massa (parâmetro) do buraco negro,
- r é a coordenada radial,
- τ é o **tempo próprio**.

Define-se então a constraint (diagnóstico numérico):

\[
\epsilon(\tau) = \left(\frac{dr}{d\tau}\right)^2 - \Big(E^2 - (1-\frac{2M}{r})(1+\frac{L^2}{r^2})\Big)
\]

Idealmente, **ε(τ) = 0** para uma solução exata. Na prática, usamos **max |ε|** como métrica de erro/deriva.

### 1.2 Integração numérica
O motor C++ integra as equações no tempo próprio usando um integrador de passo fixo RK4, retornando séries temporais:
- τ, r(τ), φ(τ)
- ε(τ) como diagnóstico
- `OrbitStatus` (ex.: `BOUND`, `CAPTURE`, `UNBOUND`, `ERROR`)

### 1.3 Critérios de validação (banca-friendly)
Para cada caso de teste, avaliamos:

1) **Consistência da constraint**
- `constraint_abs_max = max(|ε(τ)|)`
- deve ser menor que um limite (ex.: 1e-10 ou 1e-8)

2) **Comportamento esperado**
- órbita circular: `r_max_dev = max(|r(τ) - r0|)` deve ser pequeno
- plunge/capture: o status deve indicar `CAPTURE` (evento de captura definido por critério radial)

---

## 2. Casos de teste Schwarzschild incluídos

### 2.1 `schwarzschild_circular_M1_r10`
- Objetivo: verificar órbita circular estável em r = 10M.
- Esperado:
  - `OrbitStatus.BOUND`
  - `r_max_dev` pequeno (ex.: <= 1e-3)
  - `max|ε|` ~ 0 (dentro da tolerância)

### 2.2 `schwarzschild_plunge_M1_r5p8`
- Objetivo: verificar evento de captura (plunge) em regime relativístico forte.
- Esperado:
  - `OrbitStatus.CAPTURE`
  - `max|ε|` baixo (dentro da tolerância)
  - `r(τ)` decrescendo até o raio de captura

Observação importante: para um “plunge” robusto, o modelo deve permitir selecionar o ramo radial/inicialização apropriada (ex.: via condição inicial radial, ou escolha explícita do sinal de dr/dτ). O teste foi estruturado para exercer o caminho de captura via critério de evento.

---

## 3. Como rodar

### 3.1 Instalar em modo editável (compila o C++ via CMake)
No Windows, rode dentro do **Developer Command Prompt x64**:

```powershell
cd C:\Users\walla\Workspace\TCC\relativistic-spacecraft-orbits
python -m pip install -e .
````

Verificar engine:

```powershell
python -c "import relorbit_py as r; print(r.engine_hello())"
```

### 3.2 Executar validação (gera relatório)

```powershell
python -m relorbit_py.validate
```

### 3.3 Executar validação com plots

```powershell
python -m relorbit_py.validate --plots
```

Saídas:

* `out/report.json` com resultados por suíte/caso
* `out/plots/` com gráficos

---

## 4. O que você deve observar nos plots

Para Schwarzschild:

* `*_orbit.png`:
  projeção planar (x=r cosφ, y=r sinφ) apenas para visualização.

* `*_r_tau.png`:
  r vs τ.

  * circular: linha praticamente constante
  * plunge: r decai até captura

* `*_constraint.png` e `*_constraint_log.png`:
  ε(τ) e |ε(τ)| em escala log.
  O objetivo é verificar que o erro numérico permanece controlado ao longo da integração.

---

## 5. Estrutura relevante (Schwarzschild)

* `src_cpp/include/relorbit/models/schwarzschild_equatorial.hpp`

  * definição do modelo equatorial de Schwarzschild e utilitários

* `src_cpp/lib/api.cpp` e `src_cpp/include/relorbit/api.hpp`

  * API C++ exposta ao Python (funções e structs)

* `src_cpp/bindings/pybind_module.cpp`

  * bindings pybind11 (expondo `simulate_schwarzschild_equatorial_rk4`, structs e enums)

* `src/relorbit_py/simulate.py`

  * carregamento de YAML e chamada do engine com assinatura correta

* `src/relorbit_py/validate.py`

  * execução das suítes, critérios, geração de `report.json` e plots

* `src/relorbit_py/cases.yaml`

  * definição declarativa dos casos e critérios

---

## 6. Limitações atuais (deixar claro na escrita do TCC)

* O modelo atual é voltado a validação inicial e comportamento qualitativo (circular/capture).
* Para análise científica mais rica (periélio, frequência radial, comparação com soluções analíticas/parametrizações), o próximo passo é evoluir o modelo para um estado dinâmico completo (ex.: incluir variável radial conjugada, melhor controle do ramo radial e diagnósticos independentes da constraint “por construção”).

---

## 7. Referências (para citar no TCC)

* Misner, Thorne & Wheeler — *Gravitation* (geodésicas, Schwarzschild, integrais de movimento).
* Schutz — *A First Course in General Relativity* (introdução prática e geodésicas).
* Chandrasekhar — *The Mathematical Theory of Black Holes* (tratamento clássico e profundo).