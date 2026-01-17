CORREÇÕES (P0 — faça antes de qualquer “feature bonita”)

1. Declarar convenções e unidades (obrigatório de banca) 🆗

    * Criar `docs/conventions.md` e referenciar no README.
    * Definir explicitamente: sistema de unidades (geométricas com G=c=1 ou SI), assinatura do métrico, o que é M e o que é μ, e o significado de E e L (por unidade de massa).
    * Critério de aceitação: qualquer pessoa consegue ler e entender o que significam E/L/M/μ e em que unidade estão. Nada “implícito”.

2. Consertar a validação Schwarzschild para não ser “constraint por construção” 🆗

    * No C++: garantir que `epsilon` NÃO seja calculado a partir da mesma identidade usada para “forçar” o estado (se estiver).
    * Implementar uma checagem independente: `norm_u = g_{μν} u^μ u^ν + 1` (para partícula massiva). Isso exige expor `u^t, u^r, u^phi` ou `dt/dτ, dr/dτ, dφ/dτ`.
    * Onde: `src_cpp/include/relorbit/models/schwarzschild_equatorial.hpp` e a struct de output; depois expor via pybind.
    * Critério: `max|norm_u|` deve diminuir quando você reduz `dt` (teste de convergência).

3. Event detection de verdade (horizonte e turning points)

    * Implementar detecção de eventos:

      * Horizon crossing: raiz de `r(τ) - 2M = 0`.
      * Turning point: `dr/dτ = 0` (periapse/apapse).
    * Onde: no integrador Schwarzschild no C++ (a função `simulate_schwarzschild_equatorial_rk4`), com “localização” simples (bissecção/secante no intervalo do passo).
    * Critério: o instante do evento muda pouco quando você reduz `dt` (convergência), e aparece no report.

4. Corrigir classificação BOUND/UNBOUND/CAPTURE (Newton e Schwarzschild)

* Newton: caso hiperbólico não pode sair como BOUND se sua definição for física. Definir status por energia específica: E<0 bound, E≥0 unbound.
* Schwarzschild: CAPTURE tem que significar “cruzou horizonte” (ou r<r+ em Kerr), não “cheguei perto e chutei”.
* Onde: C++ (`newton.hpp` e `schwarzschild_equatorial.hpp`) e refletir no report.
* Critério: status bate com teoria (energia/potencial efetivo).

5. Teste de convergência automático (varrer dt)

* No Python: criar um modo `--convergence` que roda cada case com dt, dt/2, dt/4 e mede slope da ordem (Newton deve tender a 4 em regime suave com RK4).
* Onde: `src/relorbit_py/validate.py`.
* Critério: relatório com “ordem observada” e plots comparativos.

6. Ajustar plots para não mentirem visualmente

* Plots log: hoje você faz clip em 1e-300 e pode gerar gráficos “quadrados” gigantes. Melhor:

  * usar `np.maximum(abs_eps, eps_floor)` com `eps_floor` documentado,
  * e colocar no plot também uma linha horizontal do critério (ex.: 1e-10).
* Onde: `_plot_schw` e `_plot_newton` em `validate.py`.
* Critério: plots interpretáveis, com limites e referência do threshold.

7. Reprodutibilidade (um comando que faz tudo)

* Criar um `Makefile` simples (ou `scripts/run_all.ps1`) que:

  * instala `pip -e .`,
  * roda `python -m relorbit_py.validate --plots`,
  * gera `out/report.json`.
* Critério: qualquer máquina roda igual, sem “passos mágicos”.

FEATURES (P1/P2 — para “botar pra quebrar” no TCC)

A) Física GR de alto nível (P1)

1. Schwarzschild completo (expor também t(τ))

* Hoje você plota só r, φ. Para missão você precisa de coordenada temporal:

  * integrar `dt/dτ = E / (1 - 2M/r)`.
* Onde: Schwarzschild C++ e pybind (TrajectorySchwarzschildEq).
* Critério: report inclui `t(τ)` e você consegue discutir dilatação temporal.

2. Validação clássica de Schwarzschild: periélio e ISCO

* Implementar casos e métricas:

  * Precessão do periélio (Δφ por órbita).
  * Verificar ISCO em r=6M: estabilidade muda.
* Onde: novos casos em YAML e validação em `validate.py`.
* Critério: seus resultados reproduzem tendências esperadas e você consegue citar o valor e mostrar plot/medida.

3. Kerr equatorial (P1/P2)

* Implementar geodésicas equatoriais em Kerr: prograde vs retrograde.
* Onde: novo header `src_cpp/include/relorbit/models/kerr_equatorial.hpp`, nova função no engine e pybind.
* Critério: plots mostrando diferença prograde/retrograde (frame dragging).

B) Missão e propulsão (P1)

4. Manobras impulsivas (Δv)

* Implementar em nível “missão”: aplicar Δv em instantes/eventos (ex.: no periapse).
* Onde: Python (mission runner) primeiro; depois otimizar em C++ se precisar.
* Critério: tabela de Δv budget e mudança clara nos elementos/orbita.

5. Low-thrust (thrust contínuo) + consumo de massa

* Modelo simples: força constante ou throttle controlada + equação de massa (Tsiolkovsky contínuo simplificado).
* Onde: um novo integrador/força no Newton primeiro; depois versão GR (mais avançada).
* Critério: trajetória muda de forma previsível, consumo de propelente coerente.

6. Planejamento: targeting básico (Lambert / “match periapse”)

* Newton: Lambert clássico ou solver simples para atingir periapse/rendezvous.
* GR: ao menos targeting numérico por shooting (varrer E/L ou Δv até atingir um periapse-alvo).
* Critério: dado um alvo (r_p desejado), o solver encontra parâmetros e reporta erro final.

C) Atitude 6-DOF e GNC (P1)

7. Dinâmica de atitude com quaternions

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

10. “Perfil de missão” real (o teu diferencial)

* Definir uma missão demo:

  * aproximação, observação (manter pointing), correção de periapse, e saída/capture.
* Onde: `missions/` com scripts e YAML.
* Critério: “roteiro” executável que gera plots, relatório e narrativa de missão.
