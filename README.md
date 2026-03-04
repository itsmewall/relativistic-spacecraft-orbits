# Relativistic Spacecraft Orbits — Simulação Numérica de Missão 6-DOF em Campos Extremos

Este repositório implementa um motor de simulação híbrido (C++ + Python) para o estudo de missões espaciais de alta fidelidade física. O sistema evoluiu de um integrador de geodésicas para um simulador completo de 6 Graus de Liberdade (6-DOF), integrando dinâmica orbital translacional e atitudinal em espaços-tempos curvos (Schwarzschild e Kerr).

O projeto atua como o núcleo computacional para a engenharia de missões próximas a objetos compactos, unindo a mecânica orbital clássica à relatividade geral. Inclui suporte a análise estocástica de incertezas via Monte Carlo e simulação de empuxo vetorizado sob efeitos severos de dilatação temporal e arraste de referencial.

## 1. Arquitetura e Tecnologia

### 1.1 Stack Técnica

C++ (Core Engine): Utiliza a biblioteca Eigen para álgebra linear otimizada, permitindo a vetorização de operações (SIMD). Implementa integradores de passo fixo e adaptativo (RK4) para os estados orbitais e rotacionais.

pybind11: Bindings de baixa latência que expõem o motor C++ para o ecossistema Python sob a interface relorbit_py._engine.

Python (Orquestração): Gerencia o pipeline de Mission Runner, processa configurações declarativas via YAML, executa simulações em lote (Batch) para Monte Carlo e gera a telemetria visual e relatórios (report.json).

## 2. Fundamentação Teórica e Modelos Físicos

A simulação separa rigorosamente os regimes físicos. No escopo Newtoniano, adotam-se as grandezas usuais da astrodinâmica. Para as soluções exatas das equações de campo de Einstein, adotam-se unidades geométricas ($G = c = 1$), reduzindo massa ($M$), tempo ($t$) e distância ($r$) à mesma base dimensional.

### 2.1 Astrodinâmica Clássica e Propulsão (Problema de 2-Corpos)

A base do movimento de uma espaçonave sob um campo gravitacional central esférico é regida pela equação fundamental:

$$\frac{d^2\mathbf{r}}{dt^2} = -\frac{\mu}{r^3}\mathbf{r}$$

Onde $\mu = G(M_1 + m) \approx GM_1$ é o parâmetro gravitacional padrão. A trajetória preserva a energia orbital específica e o momento angular específico:


$$\epsilon = \frac{v^2}{2} - \frac{\mu}{r} \quad \text{e} \quad \mathbf{h} = \mathbf{r} \times \mathbf{v}$$

Para o deslocamento translacional ativo (manobras), a simulação acopla a equação fundamental de Tsiolkovsky, modelando a variação de massa discreta ou contínua do veículo:


$$\Delta v = I_{sp} g_0 \ln\left(\frac{m_0}{m_f}\right)$$

### 2.2 Relatividade Geral: Schwarzschild e Geodésicas

No entorno de uma massa $M$ esfericamente simétrica e estática, o espaço-tempo é descrito pela métrica de Schwarzschild. No plano equatorial ($\theta = \pi/2$), o elemento de linha invariante é dado por:

$$ds^2 = -\left(1-\frac{2M}{r}\right) dt^2 + \left(1-\frac{2M}{r}\right)^{-1} dr^2 + r^2 d\phi^2$$

A trajetória livre de forças não gravitacionais é uma geodésica, que extremiza o tempo próprio $\tau$. Ela satisfaz a equação da geodésica com os símbolos de Christoffel $\Gamma^\mu_{\alpha\beta}$:

$$\frac{d^2 x^\mu}{d\tau^2} + \Gamma^\mu_{\alpha\beta} \frac{dx^\alpha}{d\tau} \frac{dx^\beta}{d\tau} = 0$$

Em vez de integrar a equação de 2ª ordem diretamente, o motor explora as simetrias do espaço-tempo (vetores de Killing espaciais e temporais) que garantem a conservação da energia relativística $\mathcal{E}$ e do momento angular relativístico $\mathcal{L}$. Reduz-se a dinâmica a uma equação de 1ª ordem com um potencial efetivo $V_{\text{eff}}$:

$$\left(\frac{dr}{d\tau}\right)^2 + V_{\text{eff}}(r) = \mathcal{E}^2, \quad \text{onde} \quad V_{\text{eff}}(r) = \left(1-\frac{2M}{r}\right)\left(1+\frac{\mathcal{L}^2}{r^2}\right)$$

#### 2.2.1 Dilatação Temporal e Telemetria

O tempo coordenado $t$ (observador no infinito) diverge do tempo próprio $\tau$ da sonda. O motor resolve ativamente a razão:


$$\frac{dt}{d\tau} = \mathcal{E} \left(1 - \frac{2M}{r}\right)^{-1}$$


Isso permite prever o exato instante em que pacotes de telemetria sofrerão atraso assintótico e o momento adequado de ignição autônoma dos motores antes do cruzamento do Horizonte de Eventos ($r = 2M$).

### 2.3 Relatividade Geral: Kerr e o Efeito Lense-Thirring

Quando a singularidade possui rotação (parâmetro de spin $a = J/M$), a simetria esférica é quebrada para uma simetria axial (métrica de Kerr). O elemento de linha em coordenadas de Boyer-Lindquist revela o termo cruzado $g_{t\phi}$:

$$ds^2 = -\left(1 - \frac{2Mr}{\Sigma}\right)dt^2 - \frac{4aMr\sin^2\theta}{\Sigma}dtd\phi + \frac{\Sigma}{\Delta}dr^2 + \Sigma d\theta^2 + \left(r^2 + a^2 + \frac{2a^2Mr\sin^2\theta}{\Sigma}\right)\sin^2\theta d\phi^2$$

O termo $g_{t\phi}$ induz o Frame-Dragging (arraste do referencial). Na Ergossfera, $g_{tt}$ torna-se positivo, forçando qualquer partícula a co-rotacionar com o buraco negro, impossibilitando órbitas estáticas. As integrais de movimento em Kerr incluem a Constante de Carter ($\mathcal{Q}$), gerando torques intrínsecos no cálculo do empuxo, favorecendo dramaticamente órbitas progradas em detrimento de órbitas retrógradas.

### 2.4 Dinâmica de Atitude 6-DOF (Quaternions e Euler)

O controle de orientação da sonda exige a transição dos ângulos de Euler tradicionais para os quaternions unitários $q = q_0 + q_1i + q_2j + q_3k$, erradicando o problema do Gimbal Lock.

A evolução cinemática no tempo é governada por:


$$\dot{q} = \frac{1}{2} \Omega(\boldsymbol{\omega}) q$$

A dinâmica obedece às Equações de Euler para rotação de corpos rígidos via tensor de inércia $\mathbf{I}$:


$$\dot{\boldsymbol{\omega}} = \mathbf{I}^{-1} [\boldsymbol{\tau}_{\text{ext}} - \boldsymbol{\omega} \times (\mathbf{I}\boldsymbol{\omega})]$$

No momento do acionamento do motor principal, o empuxo $\vec{F}$ (fixo no referencial da nave ao longo do eixo $\hat{k}$) é mapeado para o referencial coordenado pela matriz de transformação direcional de cossenos derivada de $q$, acoplando atitude e alteração orbital rigorosamente ($\vec{a}_{\text{thrust}} = \mathbf{R}(q) \cdot \frac{\vec{F}}{m}$).

## 3. Método Numérico e Propagação Estocástica

### 3.1 Runge-Kutta 4 Otimizado via Eigen

A integração das EDOs acopladas (variando de 7 a 13 estados por integração) é feita via RK4 de passo fixo alocado diretamente em structs mapeados para a biblioteca Eigen. A validação da estabilidade numérica do RK4 $\mathcal{O}(h^4)$ verifica a flutuação do invariante hamiltoniano ($|\epsilon| \to 0$) e a divergência da norma atitudinal ($|\|q\| - 1| \to 0$).

### 3.2 Análise Computacional de Missão (Monte Carlo)

A injeção em órbitas relativísticas puras (como a ISCO em $r=6M$ ou em torno de Kerr) possui margem de falha microscópica. O sistema automatiza a execução em lote (simulação de Monte Carlo), aplicando ruído gaussiano às condições iniciais do vetor de estado $\mathcal{N}(\mu_{state}, \sigma^2_{sensor})$ e analisando a distribuição das resultantes no hiperplano de fase. A análise estatística determina o envelope seguro de parâmetros $\Delta v$, tempos de queima e eficiências térmicas do sistema de propulsão.

## 4. Como Rodar

### 4.1 Dependências C++ (Eigen)

O motor nativo depende integralmente da biblioteca Eigen para otimização algébrica SIMD. Por ser uma biblioteca header-only, basta cloná-la para a pasta third_party do seu workspace.

#### Crie a pasta third_party caso ela não exista

```bash
mkdir -p third_party
```

#### Clone o repositório da Eigen

```bash
git clone [https://gitlab.com/libeigen/eigen.git](https://gitlab.com/libeigen/eigen.git) third_party/eigen
```

(Certifique-se de que a estrutura resultante possua o caminho third_party/eigen/Eigen visível ao CMakeLists.txt do projeto).

### 4.2 Compilação e Instalação do Projeto

Com a Eigen preparada, invoque o pip para engatilhar o build do backend via scikit-build-core e compilar as dependências de Python via pybind11:

```bash
python -m pip install -e .
```

### 4.3 Validação Estrutural e Física

Para rodar a suíte de provas e testar as derivas de energia, precessões e conservação atitudinal:

```bash
python -m relorbit_py.validate --plots
```

### 4.4 Orquestração de Missões

Para invocar o plano de voo e processar acoplamento 6-DOF, budget de massa e telemetria:

```bash
python -m relorbit_py.run_mission --config src/relorbit_py/mission.yaml
```

### 5. Referências Bibliográficas

Os modelos matemáticos, escolhas numéricas e arquitetura de engenharia deste software foram fundamentados rigorosamente nas seguintes referências:

* BATE, R. R.; MUELLER, D. D.; WHITE, J. E. Fundamentals of Astrodynamics. Nova York: Dover Publications, 1971.

* CARROLL, S. M. Spacetime and Geometry: An Introduction to General Relativity. São Francisco: Addison-Wesley, 2004.

* CHOBOTOV, V. A. Orbital Mechanics. 3. ed. Reston: AIAA Education Series, 2002.

* CURTIS, H. D. Orbital Mechanics for Engineering Students. 4. ed. Oxford: Butterworth-Heinemann, 2020.

* MISNER, C. W.; THORNE, K. S.; WHEELER, J. A. Gravitation. Princeton: Princeton University Press, 2017.

* SCHUTZ, B. A First Course in General Relativity. 2. ed. Cambridge: Cambridge University Press, 2009.

* SILVA, W. R. Equações Diferenciais Parciais. Notas de Aula. Faculdade UnB Gama, Universidade de Brasília.

* SILVA, W. R. Mecânica do Voo Espacial - Introdução. Notas de Aula. Faculdade UnB Gama, Universidade de Brasília.

* SILVA, W. R. Série de Fourier. Notas de Aula. Faculdade UnB Gama, Universidade de Brasília.

* SUTTON, G. P.; BIBLARZ, O. Rocket Propulsion Elements. 7. ed. Nova York: John Wiley & Sons, 2001.