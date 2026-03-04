# Relativistic Spacecraft Orbits — Simulação Numérica de Missão 6-DOF em Campos Extremos

Este repositório implementa um **motor de simulação híbrido** (C++ + Python) para o estudo de missões espaciais de alta fidelidade física. O sistema evoluiu de um integrador de geodésicas para um simulador completo de **6 Graus de Liberdade (6-DOF)**, integrando dinâmica orbital e de atitude em espaços-tempos curvos (Schwarzschild e Kerr).

O projeto atua como o núcleo computacional para engenharia de missões próximas a objetos compactos, unindo a mecânica orbital clássica à relatividade geral, com análise de incertezas via Monte Carlo e suporte a empuxo vetorizado.

---

## 1. Arquitetura e Tecnologia

### 1.1 Stack Técnica
- **C++ (Core Engine)**: Utiliza a biblioteca **Eigen** para álgebra linear otimizada. Implementa integradores de passo fixo e adaptativo (RK4) para os estados orbitais e rotacionais.
- **pybind11**: Bindings de baixa latência que expõem o motor C++ para o ecossistema Python.
- **Python (Orquestração)**: Gerencia o `Mission Runner`, processa configurações via YAML, propaga estatísticas de Monte Carlo e gera a telemetria visual.



---

## 2. Fundamentação Teórica e Modelos Físicos

A separação de regimes físicos e sistemas de unidades é estrita. O modelo Newtoniano utiliza o SI ou unidades adimensionais clássicas, enquanto os modelos relativísticos adotam **unidades geométricas** ($G = c = 1$), nas quais massa ($M$), tempo ($t$) e distância ($r$) possuem a mesma dimensão fundamental.

### 2.1 Mecânica Clássica e Dinâmica de Foguetes (2-Corpos)
A base do movimento de uma espaçonave sob um campo gravitacional central é descrita pela equação fundamental da astrodinâmica:

$$\frac{d^2\mathbf{r}}{dt^2} = -\frac{\mu}{r^3}\mathbf{r}$$

Onde $\mu = G(M_1 + M_2)$ é o parâmetro gravitacional padrão. As trajetórias são regidas pela conservação de energia e momento angular específicos:
$$\epsilon = \frac{v^2}{2} - \frac{\mu}{r} \quad \text{e} \quad \mathbf{h} = \mathbf{r} \times \mathbf{v}$$

Para o deslocamento translacional ativo, aplica-se a equação do foguete de Tsiolkovsky:
$$\Delta v = I_{sp} g_0 \ln\left(\frac{m_0}{m_f}\right)$$
O empuxo é tratado como uma força perturbadora incorporada ao integrador numérico, permitindo a transição entre órbitas (e.g., Transferência de Hohmann).

### 2.2 Relatividade Geral: Schwarzschild (Buracos Negros Estáticos)
No entorno de uma massa esfericamente simétrica e não rotante, o espaço-tempo é descrito pela métrica de Schwarzschild. No plano equatorial ($\theta = \pi/2$), o elemento de linha é:

$$ds^2 = -\left(1-\frac{2M}{r}\right) dt^2 + \left(1-\frac{2M}{r}\right)^{-1} dr^2 + r^2 d\phi^2$$

A trajetória de uma sonda em queda livre é uma **geodésica temporal**, parametrizada pelo tempo próprio $\tau$, obedecendo à condição de normalização da quadri-velocidade $g_{\mu\nu}u^\mu u^\nu = -1$. Devido aos vetores de Killing, a energia e o momento angular relativísticos ($\mathcal{E}$ e $\mathcal{L}$) são conservados. A dinâmica radial reduz-se a um problema unidimensional num **potencial efetivo**:

$$\left(\frac{dr}{d\tau}\right)^2 + V_{\text{eff}}(r) = \mathcal{E}^2$$
$$V_{\text{eff}}(r) = \left(1-\frac{2M}{r}\right)\left(1+\frac{\mathcal{L}^2}{r^2}\right)$$



Este modelo prevê fenômenos inexistentes em Newton, como o avanço do periastro e a existência da **ISCO** (Innermost Stable Circular Orbit) em $r = 6M$.

### 2.3 Relatividade Geral: Kerr (O Efeito Lense-Thirring)
Quando o objeto central possui momento angular (spin $a = J/M$), a métrica de Kerr introduz o arraste do espaço-tempo (*Frame-Dragging*). Na região da **Ergossfera**, um observador é fisicamente incapaz de permanecer estático em relação ao infinito. A métrica possui termos cruzados $g_{t\phi} \neq 0$:

$$ds^2 = g_{tt}dt^2 + 2g_{t\phi}dtd\phi + g_{rr}dr^2 + g_{\theta\theta}d\theta^2 + g_{\phi\phi}d\phi^2$$

Isso gera torques geodésicos na sonda e exige um tratamento diferenciado do consumo de propelente: manobras progradas (a favor do spin) tornam-se drasticamente mais eficientes que retrógradas. A métrica de Kerr é vital para avaliar a sobrevivência de sondas inseridas em regimes de gravidade forte de buracos negros rotativos.



### 2.4 Dinâmica de Atitude 6-DOF com Quaternions
Para evitar singularidades matemáticas (*Gimbal Lock*) presentes nos ângulos de Euler, a orientação da espaçonave é integrada utilizando **quaternions unitários** $q = q_0 + q_1i + q_2j + q_3k$.

A cinemática atitudinal é governada por:
$$\dot{q} = \frac{1}{2} \Omega(\boldsymbol{\omega}) q$$
Onde $\boldsymbol{\omega} = [\omega_x, \omega_y, \omega_z]^T$ é o vetor de velocidade angular no sistema do corpo. 

A dinâmica de rotação responde às Equações de Euler para um corpo rígido com tensor de inércia $\mathbf{I}$:
$$\dot{\boldsymbol{\omega}} = \mathbf{I}^{-1} [\boldsymbol{\tau}_{\text{ext}} - \boldsymbol{\omega} \times (\mathbf{I}\boldsymbol{\omega})]$$
Onde $\boldsymbol{\tau}_{\text{ext}}$ representa torques de controle interno ou gradientes de gravidade (maré relativística). O empuxo vetorizado mapeia o eixo estrutural do motor de volta para o referencial de simulação através da matriz de rotação $\mathbf{R}(q)$.

---

## 3. Método Numérico e Propagação de Incertezas

### 3.1 Integrador de Passo Fixo (RK4 Otimizado)
O motor resolve EDOs não lineares $\dot{\mathbf{y}} = f(t, \mathbf{y})$ com o método de Runge-Kutta de 4ª ordem, processado via matrizes densas na Eigen. A qualidade da integração é avaliada pela manutenção rigorosa de:
1.  **Vínculo do Hamiltoniano**: $|\epsilon| = |p_r^2 + V_{\text{eff}} - \mathcal{E}^2| \to 0$.
2.  **Norma do Quaternion**: $\|q\| = 1$, com tolerâncias mantidas abaixo do *machine epsilon* ($< 10^{-15}$).

### 3.2 Análise Computacional (Monte Carlo)
A simulação abandona o determinismo de corpo único para incorporar uma análise de envelopes de missão. Através de simulações de Monte Carlo, aplica-se ruído estocástico (gaussiano) às variáveis de estado iniciais $[r_0, \phi_0, \mathbf{v}_0]$ e à eficiência do motor ($I_{sp}$). Isso mapeia os limites de instabilidade (e.g., captura inevitável vs escape) em regimes onde os gradientes de potencial são críticos.

---

## 4. Como Rodar

### 4.1 Instalação
O pacote Python chama o sistema de build para compilar as bibliotecas nativas via `scikit-build-core`:
```bash
python -m pip install -e .

```

### 4.2 Executar Missões (Mission Runner)

Processa planos de voo definidos em configuração YAML, acoplando a dinâmica orbital ao budget de massa.

```bash
python -m relorbit_py.run_mission --config src/relorbit_py/mission.yaml

```

### 4.3 Validação Estrutural

Roda o suíte de testes contra predições analíticas (drift de energia, precessão, constância atitudinal).

```bash
python -m relorbit_py.validate --plots

```

---

## 5. Referências Bibliográficas

1. BATE, R. R.; MUELLER, D. D.; WHITE, J. E. *Fundamentals of Astrodynamics*. Nova York: Dover Publications, 1971.
2. CARROLL, S. M. *Spacetime and Geometry: An Introduction to General Relativity*. São Francisco: Addison-Wesley, 2004.
3. CHOBOTOV, V. A. *Orbital Mechanics*. 3. ed. Reston: AIAA Education Series, 2002.
4. CURTIS, H. D. *Orbital Mechanics for Engineering Students*. 4. ed. Oxford: Butterworth-Heinemann, 2020.
5. MISNER, C. W.; THORNE, K. S.; WHEELER, J. A. *Gravitation*. Princeton: Princeton University Press, 2017.
6. SCHUTZ, B. *A First Course in General Relativity*. 2. ed. Cambridge: Cambridge University Press, 2009.
7. SILVA, W. R. *Equações Diferenciais Parciais*. Notas de Aula. Faculdade UnB Gama, Universidade de Brasília.
8. SILVA, W. R. *Mecânica do Voo Espacial - Introdução*. Notas de Aula. Faculdade UnB Gama, Universidade de Brasília.
9. SILVA, W. R. *Série de Fourier*. Notas de Aula. Faculdade UnB Gama, Universidade de Brasília.
10. SUTTON, G. P.; BIBLARZ, O. *Rocket Propulsion Elements*. 7. ed. Nova York: John Wiley & Sons, 2001.