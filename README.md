# Relativistic Spacecraft Orbits — Simulação Numérica de Missão Próxima a Buraco Negro

Este repositório implementa um **motor de simulação numérica** (C++ + Python) para estudar, com rigor físico e critérios de validação, a dinâmica de uma **sonda espacial em missão próxima a um buraco negro**, cobrindo desde o regime clássico (Newton/2-corpos) até o regime relativístico (geodésicas em Schwarzschild no plano equatorial).  
O objetivo final do TCC é evoluir este núcleo para uma **missão completa**: fases orbitais, janelas de manobra, critérios de captura/escape, análise de robustez numérica e extração de observáveis (por exemplo: precessão relativística, regimes de estabilidade, transições bound/unbound/capture).

---

## 1. Visão do Projeto

### 1.1 Arquitetura
- **C++ (engine)**: integra EDOs com foco em desempenho e reprodutibilidade numérica.
- **pybind11 (bindings)**: expõe o motor C++ como `relorbit_py._engine`.
- **Python (orquestração)**:
  - leitura de casos (`cases.yaml`)
  - execução (`simulate.py`)
  - validação e relatórios (`validate.py`)
  - geração de plots e `report.json`

### 1.2 Por que C++ + Python?
- C++ entrega **velocidade** e controle fino de performance (passo fixo, loops longos).
- Python entrega **produtividade**, inspeção, gráficos, relatórios e automação de experimentos.

---

## 2. Modelos Físicos e Equações Regentes

> Importante: este projeto separa explicitamente **regimes físicos** e **sistemas de unidades**.
- Newtoniano: unidades adimensionais (ou SI, se você definir `μ` em SI e estados coerentes).
- Schwarzschild: **unidades geométricas** típicas em GR (frequentemente `G=c=1`), onde `M` tem dimensão de comprimento/tempo.

### 2.1 Modelo Newtoniano (2-corpos planar)
A dinâmica do movimento relativo no problema de dois corpos (massa reduzida) no plano é dada por:

\[
\ddot{\mathbf r}(t) = -\mu \frac{\mathbf r(t)}{\|\mathbf r(t)\|^3},
\quad \mathbf r=(x,y),
\quad \mu = G(M_1+M_2)
\]

Estado no plano:
\[
\mathbf y = [x,\,y,\,v_x,\,v_y]
\]

Invariantes clássicos (específicos, por unidade de massa):
- **Energia específica**
\[
E = \frac{v^2}{2} - \frac{\mu}{r}, 
\quad v^2=v_x^2+v_y^2,\quad r=\sqrt{x^2+y^2}
\]
- **Momento angular específico (z)**
\[
h = x v_y - y v_x
\]

Classificação física (Newton):
- \(E < 0\) → órbita ligada (elipse/círculo)
- \(E = 0\) → parabólica
- \(E > 0\) → hiperbólica (não ligada)

O `validate` mede a deriva relativa máxima de \(E\) e \(h\) ao longo da integração.

---

### 2.2 Relatividade Geral: Schwarzschild Equatorial (geodésicas)
No entorno de uma massa esfericamente simétrica não rotante, usa-se a métrica de Schwarzschild:

\[
ds^2 = -\left(1-\frac{2M}{r}\right) dt^2 +
\left(1-\frac{2M}{r}\right)^{-1} dr^2 +
r^2(d\theta^2+\sin^2\theta\,d\phi^2)
\]

Restrição ao plano equatorial:
\[
\theta=\frac{\pi}{2},\quad d\theta=0
\]

A trajetória de uma partícula livre (sonda sem propulsão) é uma **geodésica temporal**, parametrizada pelo **tempo próprio** \(\tau\). O 4-velocidade é \(u^\mu = dx^\mu/d\tau\) e satisfaz a normalização:

\[
g_{\mu\nu}u^\mu u^\nu = -1
\]

Devido às simetrias (vetores de Killing), existem constantes de movimento:
- **Energia específica relativística** \(\mathcal E\)
- **Momento angular específico** \(\mathcal L\)

Uma forma padrão de escrever o movimento radial usa o **potencial efetivo**:

\[
\left(\frac{dr}{d\tau}\right)^2 + V_\text{eff}(r) = \mathcal E^2
\]
\[
V_\text{eff}(r) = \left(1-\frac{2M}{r}\right)\left(1+\frac{\mathcal L^2}{r^2}\right)
\]

Interpretação:
- Se \(\mathcal E^2\) estiver abaixo/ acima de barreiras de \(V_\text{eff}\), surgem turning points (órbita ligada) ou queda (captura).
- Em Schwarzschild, a estrutura de órbitas circulares e estabilidade muda drasticamente no regime forte (ex.: região próxima ao ISCO).

O projeto usa dois diagnósticos essenciais:
1) **Constraint/epsilon** (vínculo): mede quão bem a integração respeita a normalização ou a consistência interna das constantes (\(\mathcal E,\mathcal L\)) com o estado.
2) **Classificação do status** (BOUND/UNBOUND/CAPTURE): baseada em critérios físicos/eventos (ex.: cruzar um limiar \(r\) de captura).

---

## 3. Método Numérico (Integrador) e Critérios de Qualidade

### 3.1 Integração RK4 (passo fixo)
O motor implementa RK4 clássico para um sistema:
\[
\dot{\mathbf y} = f(t,\mathbf y)
\]

Com passo \(h\), RK4 faz:
\[
k_1 = f(t_n, y_n)
\]
\[
k_2 = f(t_n+\frac{h}{2}, y_n+\frac{h}{2}k_1)
\]
\[
k_3 = f(t_n+\frac{h}{2}, y_n+\frac{h}{2}k_2)
\]
\[
k_4 = f(t_n+h, y_n+h k_3)
\]
\[
y_{n+1}=y_n+\frac{h}{6}(k_1+2k_2+2k_3+k_4)
\]

Propriedade relevante:
- erro global típico ~ \(O(h^4)\) (para soluções suaves).  
Por isso, reduzir \(h\) pela metade tende a reduzir erro ~16×, e isso é um teste forte de sanidade numérica.

### 3.2 O que “passar” significa?
- Newton: deriva relativa de energia e momento angular abaixo dos limites do caso.
- Schwarzschild: máximo de \(|\epsilon(\tau)|\) abaixo do limite, e status compatível com o esperado (BOUND ou CAPTURE), além de critérios adicionais quando aplicável (ex.: desvio de raio em circular).

---

## 4. Como Rodar

### 4.1 Instalação em modo desenvolvimento
No diretório do projeto:
```bash
python -m pip install -e .
```

### 4.2 Rodar validação (sem plots)

```bash
python -m relorbit_py.validate
```

### 4.3 Rodar validação (com plots)

```bash
python -m relorbit_py.validate --plots
```

Saídas típicas:

* Console: PASS/FAIL por caso
* `out/report.json`: relatório estruturado
* `out/plots/*.png`: órbitas e diagnósticos

---

## 5. Casos de Teste (cases.yaml)

O arquivo `cases.yaml` define suites e casos.

Exemplos de casos:

* Newton circular / elíptico / hiperbólico: testa conservação de (E) e (h)
* Schwarzschild circular: testa constância de (r(\tau)) e constraint
* Schwarzschild plunge/capture: testa evento de captura + constraint

Cada caso contém:

* `model`, `params` (μ ou M, E, L), `state0`, `span`, `solver` (dt, n_steps) e `criteria`

---

## 6. O que Você Deve “Ver” e Concluir (interpretação dos plots)

Newton:

* **Orbit plot (x,y)**: forma geométrica (círculo/elipse/hipérbole).
* **Invariants vs time**: (E(t)) e (h(t)) aproximadamente constantes.
* **Drift (log)**: (|E-E_0|) e (|h-h_0|) em escala log para visualizar erro acumulado.

Schwarzschild:

* **Orbit equatorial (x,y)**: projeção plana de (r(\tau),\phi(\tau)).
* **r(tau)**: revela estabilidade/captura (queda monotônica ou turning points).
* **Constraint signed / log**: qualidade numérica do vínculo (quanto mais próximo de 0, melhor).

---

## 7. Roadmap: de “Órbita” para “Missão Completa”

O TCC não termina em “geodésica bonita”. O objetivo é **missão completa**: decisões, fases e engenharia.

Evoluções planejadas:

1. **Observáveis relativísticos**: precessão periélio, regimes bound/unbound/capture, comparação com predições analíticas.
2. **Modelos mais ricos**:

   * Kerr (buraco negro rotante), frame-dragging
   * parametrizações alternativas e robustas do estado
3. **Missão**:

   * planejamento por fases (aproximação → inserção → science orbit → saída)
   * manobras impulsivas (Δv) e/ou baixa propulsão (thrust contínuo)
   * restrições (segurança térmica, limite de maré, limite de radiação — se modelado)
4. **Numérica**:

   * controle de erro (adaptativo) e detecção de eventos (capture/escape/periapsis)
   * testes de convergência e verificação cruzada (Python vs C++)

---

## 8. Estrutura do Repositório (alto nível)

* `src_cpp/`

  * `include/relorbit/...` (headers do motor)
  * `lib/` (implementações)
  * `bindings/pybind_module.cpp` (pybind)
* `src/relorbit_py/`

  * `simulate.py` (execução de casos)
  * `validate.py` (banco de provas + plots + report)
  * `cases.yaml` (suites/casos)
  * `__init__.py` (carregamento do engine)

---

## 9. Nota de Rigor

Este repositório é construído para ser defendível: toda etapa deve ter

* equações claras,
* critérios de verificação,
* logs e evidências (plots + relatórios),
* e testes que não “se autoenganam” (ex.: convergência com dt).

---

## 10. Licença e Citação

Uso acadêmico e experimental. Se este repositório for usado como base para trabalho acadêmico, cite o autor e descreva claramente as modificações.

```