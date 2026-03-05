# TASKLIST 

## CORREÇÕES

#### v1 🆗
    1. Declarar convenções e unidades (obrigatório de banca) 🆗

        * Criar `docs/conventions.md` e referenciar no README.
        * Definir explicitamente: sistema de unidades (geométricas com G=c=1 ou SI), assinatura do métrico, o que é M e o que é μ, e o significado de E e L (por unidade de massa).
        * Critério de aceitação: qualquer pessoa consegue ler e entender o que significam E/L/M/μ e em que unidade estão. Nada “implícito”.

    2. Consertar a validação Schwarzschild para não ser “constraint por construção” 🆗

        * No C++: garantir que `epsilon` NÃO seja calculado a partir da mesma identidade usada para “forçar” o estado (se estiver).
        * Implementar uma checagem independente: `norm_u = g_{μν} u^μ u^ν + 1` (para partícula massiva). Isso exige expor `u^t, u^r, u^phi` ou `dt/dτ, dr/dτ, dφ/dτ`.
        * Onde: `src_cpp/include/relorbit/models/schwarzschild_equatorial.hpp` e a struct de output; depois expor via pybind.
        * Critério: `max|norm_u|` deve diminuir quando você reduz `dt` (teste de convergência).

    3. Event detection de verdade (horizonte e turning points) 🆗

        * Implementar detecção de eventos:

        * Horizon crossing: raiz de `r(τ) - 2M = 0`.
        * Turning point: `dr/dτ = 0` (periapse/apapse).
        * Onde: no integrador Schwarzschild no C++ (a função `simulate_schwarzschild_equatorial_rk4`), com “localização” simples (bissecção/secante no intervalo do passo).
        * Critério: o instante do evento muda pouco quando você reduz `dt` (convergência), e aparece no report.

    4. Corrigir classificação BOUND/UNBOUND/CAPTURE (Newton e Schwarzschild) 🆗

        * Newton: caso hiperbólico não pode sair como BOUND se sua definição for física. Definir status por energia específica: E<0 bound, E≥0 unbound.
        * Schwarzschild: CAPTURE tem que significar “cruzou horizonte” (ou r<r+ em Kerr), não “cheguei perto e chutei”.
        * Onde: C++ (`newton.hpp` e `schwarzschild_equatorial.hpp`) e refletir no report.
        * Critério: status bate com teoria (energia/potencial efetivo).

    5. Teste de convergência automático (varrer dt) 🆗

        * No Python: criar um modo `--convergence` que roda cada case com dt, dt/2, dt/4 e mede slope da ordem (Newton deve tender a 4 em regime suave com RK4).
        * Onde: `src/relorbit_py/validate.py`.
        * Critério: relatório com “ordem observada” e plots comparativos.

    6. Ajustar plots para não mentirem visualmente 🆗

        * Plots log: hoje você faz clip em 1e-300 e pode gerar gráficos “quadrados” gigantes. Melhor:

        * usar `np.maximum(abs_eps, eps_floor)` com `eps_floor` documentado,
        * e colocar no plot também uma linha horizontal do critério (ex.: 1e-10).
        * Onde: `_plot_schw` e `_plot_newton` em `validate.py`.
        * Critério: plots interpretáveis, com limites e referência do threshold.

    7. Reprodutibilidade (um comando que faz tudo) 🆗

        * Criar um `Makefile` simples (ou `scripts/run_all.ps1`) que:

        * instala `pip -e .`,
        * roda `python -m relorbit_py.validate --plots`,
        * gera `out/report.json`.
        * Critério: qualquer máquina roda igual, sem “passos mágicos”.

#### v2 🆗

    1. Risco de Memória (Out of Memory) em Missões Longas 🆗
        O Problema: Hoje, o SolverCfg no C++ aceita apenas dt e n_steps. A cada passo do RK4, você dá um .push_back() nos vetores de tempo, posição, derivadas, etc. Se uma missão de transferência orbital levar meses e exigir 10^7 passos para manter a precisão do RK4, o seu C++ vai alocar Gigabytes de RAM e devolver listas gigantescas para o Python. O Matplotlib vai travar instantaneamente ao tentar plotar isso.

        A Solução: Adicionar um parâmetro record_every (ou stride) no SolverCfg. O RK4 continua rodando os cálculos finos (ex: a cada 5×10 −4τ), mas só salva o estado na struct a cada N passos (ex: a cada 100 passos). Isso salva a memória e acelera absurdamente a ponte C++ ↔ Python.

    2. Dívida Técnica Brutal no validate.py (Risco de Manutenção) 🆗
        O Problema: O seu arquivo validate.py virou um monólito gigante. A lógica que imprime a tabela e gera o JSON da suíte "schwarzschild" foi praticamente copiada e colada para a suíte "kerr_equatorial". Quando adicionarmos as suítes de propulsão, esse arquivo vai passar de 1000 linhas e ficar incontrolável.

        A Solução: Refatorar o validate.py. Criar uma função genérica run_and_report_suite(suite_name, cases, validator_func) que centraliza a formatação do terminal e a montagem do dicionário JSON. Isso vai enxugar o arquivo pela metade e deixá-lo pronto para aceitar N novas suítes de missões.

    3. Interpolação Linear de Eventos Perto do Horizonte (Risco de Precisão) 🆗
        O Problema: No C++, quando a sonda cruza o horizonte ou chega no periastro, você detecta a troca de sinal e usa uma função lerp (interpolação linear) para achar o ponto exato da travessia. Em órbitas Newtonianas isso é ótimo. Mas perto do horizonte de Kerr, o espaço-tempo é incrivelmente deformado e a coordenada de tempo t diverge para o infinito. Uma reta ligando o ponto anterior e o próximo gera um "borrão" na métrica exata.

        A Solução: Para os cálculos do TCC, a interpolação linear até serve se o dt for minúsculo, mas o ideal é implementar uma Interpolação Cúbica de Hermite. Como o RK4 já te dá a posição e a velocidade (dr/dτ) nos dois pontos, nós podemos traçar uma curva suave e fisicamente exata para cravar o evento no milissegundo correto, sem precisar diminuir o dt da simulação inteira.


#### v3 

    1. Extração de Invariantes Físicos (Verificação de Erro) 🆗

        * O problema: Integradores numéricos acumulam erros de truncamento que podem criar trajetórias fisicamente impossíveis. Sem um gráfico de erro de conservação, não há prova de que a sonda não está ganhando ou perdendo energia artificialmente devido às limitações do algoritmo RK4.
        * A solução: Implementar o monitoramento do invariante $\epsilon = p_\mu p^\mu$, gerando um gráfico de $\Delta \epsilon$ vs Tempo para provar a estabilidade numérica e o rigor científico da simulação.

    2. Análise de Redshift Assintótico 🆗

        * O problema: Gráficos de redshift em escala linear escondem o comportamento matemático real perto do horizonte de eventos, dificultando a validação da métrica. Sem uma análise de lei de potência, o trabalho não confirma se a latência diverge conforme previsto pela Relatividade Geral.
        * A solução: Realizar a análise em escala log-log do Redshift em função da distância ao horizonte $(r - r_s)$, demonstrando visualmente que o atraso de sinal segue a divergência teórica exata.

    3. Mapa de Visibilidade Relativística (Light Bending)

        * O problema: A visibilidade baseada em oclusão geométrica (linha reta) é fisicamente incorreta em campos fortes, pois ignora que a gravidade curva a luz. Isso invalida qualquer análise de link de comunicação real entre a sonda e a Terra.
        * A solução: Implementar o cálculo do Cone de Escape de Fótons, definindo a visibilidade com base no ângulo crítico de emissão para determinar se o sinal de rádio consegue vencer a curvatura do espaço-tempo e atingir o infinito.

#### v4

    1. Sensibilidade do Custo Energético vs. Spin

        * O problema: Simulações isoladas não demonstram como a rotação do buraco negro afeta a viabilidade técnica de uma missão espacial. Sem uma variação paramétrica do spin, o trabalho não quantifica a economia de combustível (ou o custo extra) gerada pelo arraste do espaço-tempo (frame-dragging) em órbitas prógradas e retrógradas.
        * A solução: Implementar um Estudo de Sensibilidade Paramétrica que execute simulações em lote variando o spin de $a=0$ a $a=0.99$. O resultado será um gráfico de Delta-V total vs. Spin, provando como o parâmetro de Kerr dita as especificações de hardware da sonda.

    2. Link Budget Relativístico e Capacidade de Canal

        *O problema: O cálculo do Redshift atual é apenas geométrico e não traduz o impacto real na comunicação. Para a engenharia, é necessário saber se a dilatação temporal reduz a taxa de transmissão (bitrate) a ponto de inviabilizar o envio de dados científicos conforme a sonda se aproxima do horizonte.
        * A solução: Modelar o Link Budget Relativístico, onde a frequência recebida ($f_{obs}$) dita a largura de banda disponível. Gerar um gráfico de Capacidade de Canal (bits/s) vs. Raio ($r$), transformando a telemetria em um dado de projeto de telecomunicações espaciais.

    3. Geodésicas Nulas (Light Bending) na Visibilidade

        * O problema: A visibilidade baseada em oclusão linear ignora que a luz faz curvas em campos gravitacionais fortes. Isso pode levar a conclusões erradas no TCC, como afirmar que a sonda está oculta quando, na verdade, a luz contorna o buraco negro e atinge o observador (Lente Gravitacional).
        * A solução: Integrar as Geodésicas Nulas para o sinal de rádio, calculando a trajetória do fóton da sonda ao observador. Isso permitirá definir o Cone de Aceitação de Sinal, garantindo que a visibilidade reportada no Item 7 seja fisicamente rigorosa.

    4. Otimização de Trajetória de Consumo Mínimo

        * O problema: O Targeting atual por bisseção encontra uma manobra funcional, mas não necessariamente a mais eficiente. Em missões reais, é vital minimizar a integral do empuxo para maximizar a vida útil da sonda.
        * A solução: Implementar um algoritmo de Otimização de Trajetória (como o método de gradiente ou algoritmos genéticos simples) para encontrar o perfil de empuxo que atinge o alvo com o mínimo consumo de massa.


#### v5 🆗   

    1. Simulação de Monte Carlo (Análise de Dispersão) 🆗
        * Em vez de simular uma trajetória, simule 100.000 ao mesmo tempo.
        * O detalhe: Cada "partícula" na sua nuvem teria um erro de sensor, uma variação de massa ou um ruído no empuxo.
        * O custo: O tempo de execução sobe linearmente. Se 1 rodada leva 1s, 100.000 rodadas levam 27 horas.
        * Resultado: Você gera um mapa de probabilidade de onde a nave estará. Isso é o que a NASA faz para pousar em Marte.

    2. Acoplamento de Atitude de Alta Frequência 🆗
        * Atualmente, você integra a atitude e a órbita. Mas e se a nave não for um ponto?
        * O detalhe: Implemente o Torque de Maré. Em campos gravitacionais extremos, a frente da nave é puxada com mais força que a traseira. Isso gera um torque que tenta "esticar" a nave (espaguetificação).
        * O custo: Você terá que calcular tensores de gradiente de gravidade a cada microssegundo de tempo próprio.

    3. Integração de "Ray Tracing" de Telemetria 🆗
        * Em vez de apenas dizer "há visibilidade", simule os fótons saindo da antena da nave e viajando pelo espaço curvo até a Terra.
        * O detalhe: Para cada ponto da trajetória, dispare 1.000 "partículas de luz" (geodésicas nulas) em várias direções para ver quais atingem o receptor.
        * O custo: Isso transforma sua simulação em um algoritmo de busca, pesadíssimo para a CPU.


## FEATURES

### A - Física GR de alto nível (P1) 🆗

    1. Schwarzschild completo (expor também t(τ)) 🆗

        * Hoje você plota só r, φ. Para missão você precisa de coordenada temporal:

            * integrar `dt/dτ = E / (1 - 2M/r)`.
        * Onde: Schwarzschild C++ e pybind (TrajectorySchwarzschildEq).
        * Critério: report inclui `t(τ)` e você consegue discutir dilatação temporal.

    2. Validação clássica de Schwarzschild: periélio e ISCO 🆗

        * Implementar casos e métricas:

        * Precessão do periélio (Δφ por órbita).
        * Verificar ISCO em r=6M: estabilidade muda.
        * Onde: novos casos em YAML e validação em `validate.py`.
        * Critério: seus resultados reproduzem tendências esperadas e você consegue citar o valor e mostrar plot/medida.

    3. Kerr equatorial (P1/P2) 🆗

        * Implementar geodésicas equatoriais em Kerr: prograde vs retrograde.
        * Onde: novo header `src_cpp/include/relorbit/models/kerr_equatorial.hpp`, nova função no engine e pybind.
        * Critério: plots mostrando diferença prograde/retrograde (frame dragging).

### B - Missão e propulsão (P1)

    #### 4. Manobras impulsivas ($\Delta v$) e Gerenciamento de Massa 🆗

    * Implementação: Aplicar saltos instantâneos no vetor de estado $[p_r, L]$.
    * Novidade: Introduzir a **massa da sonda ($m$)** como variável. Cada $\Delta v$ deve calcular o consumo de combustível pela Equação de Tsiolkovsky: $\Delta m = m_{atual} \cdot (1 - e^{-\Delta v / (I_{sp} \cdot g_0)})$.
    * Onde: Lógica de consumo no Python (`mission.py`) integrando com as unidades do `units.py`.
    * Critério: Tabela de "Delta-v Budget" e "Mass Budget". A simulação deve falhar se o combustível acabar antes da manobra final.

    #### 5. Low-thrust (thrust contínuo) + Dinâmica Não-Geodésica 🆗

    * Modelo: Adicionar um termo de força própria $f^\mu$ (aceleração do motor) nas equações diferenciais. A trajetória deixa de ser uma geodésica pura (queda livre) e passa a ser uma **geodésica forçada**.
    * Novidade: Implementar o empuxo em componentes: *Radial* (para mudar a excentricidade rapidamente) e Tangencial (para ganhar energia orbital/subir a órbita).
    * Onde: Novo integrador em C++ que aceita um parâmetro `thrust_vector` e `Isp`.
    * Critério: Demonstração de **Orbit Raising** (subida em espiral). Comparar o tempo de subida de $6M$ para $10M$ usando diferentes níveis de aceleração.

    #### 6. Planejamento e Targeting (Targeting Numérico) 🆗

    * Lambert Relativístico: Como não há solução fechada, implementar um **Solver de Shooting** (Bisseção ou Newton-Raphson) em Python.
    * Cenário de Missão: "Dado que estou em $r=20M$, qual $\Delta v$ devo aplicar para que meu periapse seja exatamente $3M$ (limite da ISCO de Schwarzschild)?"
    * Critério: O planejador deve sugerir a manobra e a simulação deve confirmar a chegada no alvo com erro inferior a 0.1%.

    #### 7. Telemetria e Observáveis 🆗

    * Redshift e Doppler: Calcular a razão $dt/d\tau$ para cada ponto da trajetória. Isso simula o atraso de comunicação e a mudança de frequência dos sinais enviados pela sonda para a base na Terra.
    * Visibilidade: Implementar o ângulo de visibilidade (horizonte local) para saber se a sonda está "escondida" atrás do buraco negro em relação a um observador distante.
    * Critério: Gráfico de "Communication Latency" (atraso de sinal) conforme a sonda mergulha no potencial gravitacional.

### C - Atitude 6-DOF e GNC (P1)

    7. Dinâmica de atitude com quaternions 🆗
        * Estado: q (4) + ω (3). Equações padrão.
        * Onde: novo módulo C++ (ou Python primeiro): `attitude.hpp` + bindings.
        * Critério: norma de q = 1 (com renormalização controlada), energia rotacional conservada sem torque.

    8. Acoplamento órbita–atitude via thrust no body frame

        * Thrust definido no corpo; converter para frame inercial e aplicar na dinâmica orbital.
        * Onde: camada “mission sim” (Python orquestra) e depois engine.
        * Critério: mudar atitude muda trajetória (acoplamento real).

    9. Controle de atitude (PD e depois LQR)

        * PD primeiro: erro de apontamento converge.
        * Depois LQR (se quiser brilhar): linearização local e ganho.
        * Critério: plot do erro angular caindo e torque dentro de limites.

    10. “Perfil de missão” real

    * Definir uma missão demo:

    * aproximação, observação (manter pointing), correção de periapse, e saída/capture.
    * Onde: `missions/` com scripts e YAML.
    * Critério: “roteiro” executável que gera plots, relatório e narrativa de missão.