# KERR–MC (Monte Carlo · Kerr 6-DOF) — Guia rápido da interface

Este painel executa muitas simulações (“amostras”) do mesmo cenário orbital/atitude em Kerr 6-DOF, aplicando incertezas (dispersões) nos parâmetros, e depois mostra estatísticas e gráficos da distribuição dos resultados.

---

## 1. Parâmetros nominais (lado esquerdo)

São os valores “centrais” antes das dispersões. Cada simulação sorteia perturbações em cima disso (se as fontes de incerteza estiverem ligadas).

- **M [geom]**  
  Massa do buraco negro em unidades geométricas. É a escala do problema (tudo fica em “M”).  
  Observação: os eixos/valores de raio no app aparecem em **[M]**.

- **a [M]**  
  Parâmetro de rotação (spin) do buraco negro (Kerr).  
  Restrições típicas: `0 ≤ a < M` (no código costuma haver clamp pra evitar `a ≥ M`).

- **E [mc²]**  
  Energia específica (constante de movimento do geodésico, na tua convenção).  
  Em termos práticos, ajuda a definir “o quão ligada/solta” é a órbita.

- **L [mM]**  
  Momento angular específico (ou constante equivalente), também define forma da órbita.

- **r₀ [M]**  
  Raio inicial da sonda (condição inicial).

- **τ_final [M]**  
  Tempo próprio final da simulação (duração). É o “até quando” integrar.

- **dt [M]**  
  Passo de integração em tempo próprio. Menor dt → mais caro e geralmente mais preciso.

- **massa [kg]**  
  Massa inicial da nave (usada quando empuxo/consumo de massa está ativo).  
  Importante: se **THRUST** estiver desligado, esse campo não altera a órbita (porque F=0).

---

## 2. TIDAL (maré gravitacional)

Ativa/desativa torque de maré no corpo rígido (6-DOF “de verdade” no sentido de atitude).

- **TIDAL enabled**  
  Quando ligado, a nave deixa de ser “ponto”: o gradiente de gravidade gera um torque que muda a rotação (ω) e a atitude (quaternion).

- **Modelo (dropdown)**  
  Escolhe como o tensor tidal/curvatura é obtido para gerar o torque.

  - **DIAG_EIJ**  
    Usa uma forma diagnóstica do tensor tidal \(E_{ij}\) (parte elétrica do Weyl / tidal tensor) para construir o torque. É rápido e bom para varrer muitos casos.

  - **WEAK_N**  
    Aproximação “campo fraco / newtoniana” do tidal (escala típica ~ 3M/r³).  
    Serve muito bem pra validação e sanity-check longe do buraco negro.

  - **RIEMANN_FD**  
    Calcula curvatura/tensor de Riemann via diferenças finitas (FD). É mais caro, mas é o modo mais “fundamental” e útil pra validação (inclui testes de convergência).

---

## 3. THRUST (propulsão)

Liga/desliga empuxo (e portanto consumo de massa). Isso testa acoplamento órbita-massa (e eventualmente atitude, se o nozzle estiver no frame do corpo).

- **THRUST enabled**  
  Quando ligado, o motor atua e a massa diminui ao longo da simulação (depende do modelo de Isp).

- **F_newton [N]**  
  Magnitude do empuxo (em Newton).  
  Mais empuxo → maior aceleração → maior impacto na órbita e maior consumo.

- **Isp [s]**  
  Impulso específico do motor.  
  Maior Isp → mais eficiente (menos massa consumida para o mesmo empuxo ao longo do tempo).

---

## 4. Fontes de incerteza (σ relativo)

Cada checkbox liga uma dispersão gaussiana em torno do nominal.  
O campo amarelo é o **σ relativo** (fração). Ex.: `0.001` = 0,1%.

- **Energia E**  
  Aplica ruído em E. Mexe bastante na forma/energia orbital.

- **Mom. angular L**  
  Aplica ruído em L. Afeta periastro/apoastro e precessões.

- **Raio inicial r₀**  
  Aplica ruído no raio inicial.

- **Massa da nave**  
  Aplica ruído na massa inicial (relevante quando THRUST está ligado; caso contrário é quase “decorativo”).

- **Mom. radial pr₀**  
  Perturba o momento radial inicial. Pode mudar excentricidade/“fase” da órbita.

- **Spin ωz**  
  Perturba a rotação inicial do corpo (componente z no frame do corpo).  
  Relevante se atitude/torques estiverem ativos.

- **Spin Kerr a**  
  Perturba o spin do buraco negro (a). Muda a geometria e afeta órbitas/efeitos de frame dragging.

Observação: “σ relativo” significa que o parâmetro vira `p * (1 + N(0,σ))`.  
Alguns parâmetros podem ser tratados como σ absoluto (depende da tua configuração interna).

---

## 5. Configuração da corrida

- **N simulações**  
  Quantidade de amostras Monte Carlo (ex.: 1.000, 10.000, 100.000).

- **Workers (CPU)**  
  Quantos processos paralelos (ProcessPool). Em geral, use algo perto do nº de núcleos.

- **Seed RNG**  
  Semente do gerador aleatório. Mesma seed → mesma sequência de dispersões → reprodutível.

- **Atalhos (100 / 1.000 / 10.000 / 100.000)**  
  Só ajustam rapidamente o N.

- **INICIAR**  
  Começa a corrida.

- **PARAR**  
  Interrompe a corrida (o app deve fechar o CSV e registrar o estado).

- **EXPORTAR CÓPIA CSV**  
  Salva uma cópia manual (o auto-save já deve estar gerando um CSV durante a corrida).

---

## 6. Painel superior (lado direito) — Progresso e métricas

### Barra de PROGRESSO
- Percentual do total concluído (`done / total`).

### Métricas (caixas)

- **Concluídas**  
  `n_done / n_total` — quantas simulações terminaram.

- **PASS**  
  Quantas simulações terminaram “OK” (não capturou e não deu erro).

- **CAPTURA**  
  Quantas cruzaram o raio de captura (horizonte/limiar definido), e a %.

- **Velocidade**  
  Simulações por segundo (sim/s).

- **Decorrido**  
  Tempo já gasto desde o start.

- **ETA**  
  Estimativa de tempo restante (depende da velocidade atual).

- **r_final μ**  
  Média online (Welford) do raio final das simulações PASS.

- **r_final σ**  
  Desvio padrão online do raio final (PASS).

- **ε_rms μ**  
  Média do erro RMS (métrica interna do simulador; boa para acompanhar estabilidade numérica).

- **‖q‖ err μ**  
  Média do erro de norma do quaternion. Idealmente ~1e-16 a 1e-12 (dependendo do integrador).

---

## 7. Log em tempo real

Mostra mensagens do coordenador e alguns resultados periódicos:
- `OK` (PASS), `CAPTURE`, `ERRO`.
- Também imprime amostras com `r=...`, `ε=...`, `‖q‖=...` conforme a taxa de log.

Importante: o log é limitado para não explodir memória.

---

## 8. Abas de gráficos (direita)

- **HIST r_final**  
  Histograma do raio final. Mostra dispersão e centralidade (μ e σ).

- **HIST ε_rms**  
  Histograma do erro RMS (normalmente em log10). Útil para ver “caudas” de instabilidade.

- **DISPERSÃO**  
  Scatter plot (ex.: `r_final` vs `ε_rms`). Mostra correlações e outliers.

- **CONVERGÊNCIA**  
  Curvas de convergência das médias com N (μ(r_final) e μ(ε_rms)).  
  O normal é oscilar bastante no início e estabilizar conforme N cresce.

---

## 9. Interpretação rápida do “0/0” no começo

Quando aparece `0/0` e 0%:
- A corrida ainda não iniciou (ou ainda está a montar/abrir o pool),
- ou o primeiro resultado ainda não retornou do worker.

Assim que os primeiros futures completam, o contador começa a subir e os gráficos saem de “aguardando dados”.

---

## 10. Dica prática de uso (padrão de engenharia)

- Para validar estabilidade: rode `N=1.000` com dispersões pequenas e verifique `‖q‖ err` e `ε_rms`.
- Para medir sensibilidade: aumente `σ` de E e L, e compare o histograma de `r_final`.
- Para 6-DOF “de verdade”: ligue **TIDAL** (DIAG_EIJ ou RIEMANN_FD) e **THRUST** quando quiser também consumo de massa.