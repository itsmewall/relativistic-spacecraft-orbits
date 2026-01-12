# Requisitos do Projeto

**Tema:** Órbitas Relativísticas de Sondas Próximas a Buracos Negros: Simulação Numérica, Comparação Newtoniana e Implicações de Projeto de Missão

## 1. Objetivo e escopo

### 1.1 Objetivo central

Construir um simulador numérico reprodutível que compare a dinâmica orbital Newtoniana com a dinâmica relativística (Schwarzschild, plano equatorial) para sondas próximas a buracos negros, e traduzir as diferenças físicas em implicações de missão (envelopes de viabilidade, estabilidade, custo relativo e sincronização temporal).

### 1.2 Perguntas que o projeto deve responder

* Como o modelo relativístico altera a forma orbital em comparação ao Newtoniano?
* Quais regimes e condições levam a órbitas estáveis, instáveis, escape ou captura?
* Qual é o impacto do campo forte em:

  * precessão do periastro,
  * raio mínimo de órbita estável (ISCO),
  * dilatação temporal (τ vs t),
  * período orbital medido localmente vs por observador distante?
* Quais implicações práticas isso impõe ao planejamento de missão (envelope operacional, margens, custo relativo)?

### 1.3 Itens fora de escopo

* Propulsão realista (motores, massa variável, performance térmica).
* Radiação/acréscimo/disco de acreção, efeitos MHD, pressão.
* Perturbações N-corpos, oblaticidade, J2, etc.
* Autopilot completo (GNC completo). Apenas métricas e envelopes.
* Modelos completos de comunicação (apenas análises de sincronização e atraso temporal de forma conceitual/paramétrica).

---

## 2. Modelos físicos obrigatórios

### 2.1 Newtoniano

**Propósito:** referência clássica de mecânica orbital.

* Potencial: (V(r)=-GM/r)
* Equações de movimento planar:

  * estado mínimo: ([x,y,v_x,v_y]) ou ([r,\phi,\dot r, \dot\phi])
* Conservações (devem ser monitoradas e testadas):

  * energia específica ( \epsilon = v^2/2 - GM/r )
  * momento angular específico ( h = r^2 \dot\phi ) (ou ( \mathbf{r}\times\mathbf{v} ))

**Órbitas a suportar:**

* Elíptica (bound)
* Parabólica (limite)
* Hiperbólica (unbound/flyby)

### 2.2 Relativístico 

**Métrica principal:** Schwarzschild
**Plano:** equatorial (θ = π/2)

**Devem existir dois tempos no simulador:**

* tempo próprio: ( \tau )
* tempo coordenado: ( t )

**Constantes de movimento:**

* (E) (energia específica relativística)
* (L) (momento angular específico)

**Forma mínima aceita do modelo:**

* Integração das geodésicas usando:

  * Lagrangiano/variacional **ou**
  * equações com potencial efetivo

**Efeitos que precisam ser demonstrados com resultados:**

* Precessão do periastro em órbitas bound
* ISCO em Schwarzschild ((r=6GM/c^2) em notação usual; ou (r=6M) em unidades geométricas)
* Captura (aproximação do horizonte) vs escape/deflexão
* Dilatação temporal (função (d\tau/dt) e drift acumulado)

### 2.3 Extensões opcionais

* Kerr no plano equatorial (prograde/retrograde), com comparação qualitativa do ISCO e comportamento orbital

---

## 3. Unidades e convenções

### 3.1 Unidades internas do código

* O núcleo relativístico deve operar em unidades geométricas: **G = c = 1**
* O parâmetro de massa do buraco negro entra como (M) (comprimento/tempo equivalente)

### 3.2 Conversão para SI

* Uma camada de utilitários deve converter:

  * (M \leftrightarrow) massa SI (kg) via (GM/c^2)
  * distâncias em (M) para metros
  * tempos (em (M)) para segundos

### 3.3 Convenções de referência

* Horizonte em Schwarzschild: (r = 2M)
* Regimes:

  * Campo fraco: (r \gg 10M)
  * Campo forte: (r \sim 2M–20M)

---

## 4. Requisitos do simulador

### 4.1 Linguagem e dependências

* Linguagem: Python e C++
* Integração:

  * obrigatório: RK45 (adaptativo) ou equivalente
  * desejável: RK4 manual para testes de ordem/convergência
* Visualização: Matplotlib

### 4.2 Arquitetura mínima de módulos

* `relorbit/units.py`
  conversões geométrico ⇄ SI, constantes físicas, helpers
* `relorbit/newton.py`
  dinâmica Newtoniana, integrações e métricas básicas
* `relorbit/schwarzschild.py`
  geodésicas + integrações + eventos
* `relorbit/analysis.py`
  extração de métricas (periastro, período, precessão, τ/t etc.)
* `relorbit/mission.py`
  envelope operacional, custo relativo/Δv-style, restrição de marés
* `relorbit/plots.py`
  geração de figuras padronizadas
* `relorbit/validate.py`
  suíte automática de validação que roda tudo e gera relatório

### 4.3 Tipos de dados

* Estrutura `Trajectory`:

  * arrays de estado
  * tempo (t ou τ)
  * metadados do solver (tolerâncias, método, passos)
  * invariantes calculados (E, L, energia Newtoniana, etc.)
  * status: `BOUND`, `UNBOUND`, `CAPTURE`, `ERROR`

### 4.4 Reprodutibilidade

* Um comando único deve:

  * rodar a suíte de validação
  * gerar todas as figuras e tabelas
  * salvar outputs versionados (CSV/JSON/PNG/PDF)

---

## 5. Requisitos numéricos

### 5.1 Solvers e tolerâncias

* Deve existir um arquivo central (`solver_cfg`) com:

  * método
  * atol/rtol
  * dt inicial (se aplicável)
  * limite de passos
* Deve existir fallback/controlador:

  * se solver divergir, registrar e marcar caso como falho com diagnóstico

### 5.2 Eventos e detecção automática

* Horizonte (captura): disparar evento ao atingir (r \le 2M + \epsilon)
* Escape: (r \ge r_{\max}) configurável
* Turning points: detecção de (dr/d\tau = 0) para periastro/apoastro (bound)

### 5.3 Convergência e ordem

* Teste sistemático:

  * RK4 com dt, dt/2, dt/4
  * relatório de erro vs dt (log-log) mostrando ordem ~4
* Para RK45:

  * varrer rtol/atol e mostrar queda de erro e estabilidade de invariantes

---

## 6. Validações físicas

### 6.1 Newtoniano

* Conservação de energia e momento angular
* Comparação com solução analítica de Kepler (elementos orbitais)
* Casos padrão:

  * elipse (a,e)
  * hipérbole (flyby)

### 6.2 Schwarzschild

* Conservação de (E) e (L)
* Consistência do “constraint” radial via potencial efetivo
* ISCO:

  * demonstrar estabilidade acima de 6M
  * demonstrar instabilidade/captura abaixo (ou no limiar)
* Limite Newtoniano:

  * para r grande, diferenças devem ir a zero (quantificar)
* Precessão (campo fraco):

  * medir Δφ por órbita e comparar com fórmula aproximada (erro decresce no regime fraco)
* Dilatação temporal:

  * computar (d\tau/dt) ao longo da órbita e drift acumulado por órbita

### 6.3 “Validação real”

* GPS (efeitos relativísticos em sincronização) **ou**
* Precessão de Mercúrio (como referência de campo fraco)
* Opcional: estrela S2 em Sgr A* (se sobrar tempo)

---

## 7. Outputs científicos

### 7.1 Figuras mínimas obrigatórias

1. Trajetórias: r(φ) Newton vs Schwarzschild para casos comparáveis
2. Precessão: Δφ por órbita vs parâmetro (r/M ou a,e no campo fraco)
3. ISCO: gráfico de potencial efetivo + simulação (estável vs instável)
4. Dilatação temporal: (d\tau/dt) vs r/M e drift acumulado por órbita
5. Conservação numérica: erro de E e L vs tolerância/passo
6. Convergência: erro vs dt (ordem)

### 7.2 Tabelas obrigatórias

* Tabela de casos de validação (parâmetros, status, erros de invariantes)
* Tabela de métricas orbitais (periastro, período em τ e em t, Δφ, etc.)

---

## 8. Camada Engenharia/Missão

### 8.1 Envelope operacional

* Regiões:

  * proibida por captura (perto do horizonte)
  * instável (até ISCO)
  * estável (acima do ISCO)
  * recomendada com margem (ex.: ≥ 8M ou ≥ 10M, justificar)
* Critérios exibidos em gráfico:

  * estabilidade
  * dilatação temporal
  * custo relativo/Δv-style
  * restrição de marés (limite adicional)

### 8.2 “Δv-style”

* Newtoniano: Δv de circularização/escape usando vis-viva
* Relativístico: custo relativo baseado em energia específica relativística de órbita circular (ou equivalente), mostrando divergência ao aproximar do ISCO

### 8.3 Marés

* Critério simplificado:

  * gradiente gravitacional ∝ 1/r³
  * limite imposto por tamanho típico de sonda (parâmetro configurável)
* Produzir curva “r mínimo por maré” no envelope

### 8.4 Comunicação e sincronização

* Output mínimo:

  * diferença τ vs t por órbita
  * estimativa de drift acumulado ao longo de N órbitas
* (Opcional) atraso de sinal conceitual (Shapiro delay) apenas como observação/extra

---

## 9. Critérios de aceite

### Gate A — Newtoniano “fechado”

* Erro de energia e L dentro de tolerância definida
* Convergência demonstrada
* Comparação com Kepler analítico passa

### Gate B — Schwarzschild “fechado”

* Conservação de E e L controlada
* ISCO reproduzido e demonstrado
* Precessão medida e compatível no campo fraco
* Dilatação temporal calculada e consistente

### Gate C — Projeto final “pronto”

* `validate.py` roda e gera relatório/figuras automaticamente
* Repositório reproduzível (do zero) em máquina limpa
* Resultados mínimos (figuras + tabelas) prontos para o texto do TCC

---

## 10. Documentação e escrita

### 10.1 README do repositório deve conter

* Motivação e objetivo
* Como instalar e rodar
* Como gerar figuras e relatório
* Lista de validações implementadas
* Exemplos de uso (2–3 comandos)

### 10.2 Estrutura sugerida do TCC

1. Introdução e motivação (campo forte + engenharia de missão)
2. Fundamentação teórica (Newton vs RG; Schwarzschild; geodésicas)
3. Metodologia numérica (EDOs, solvers, convergência, tolerâncias)
4. Validações (limite Newtoniano, ISCO, precessão, GPS/Mercúrio)
5. Resultados e comparação (figuras/tabelas)
6. Implicações de missão (envelope operacional + discussão)
7. Conclusão e trabalhos futuros (Kerr, S2, etc.)

---

## 11. Registro de riscos

### Riscos principais

* Complexidade matemática virar fim em si
* Solver instável perto do horizonte/ISCO
* Falta de validação virar “simulação bonita” sem credibilidade
* Escopo crescer para Kerr/S2 antes do básico estar perfeito

### Mitigações

* Gates rígidos por fase
* Começar por Schwarzschild equatorial
* Validação automatizada desde o início
* Kerr apenas após Gate C, ou vira “trabalho futuro”

---

## 12. Lista mínima de casos de validação

* Newton elipse (a,e fixos) — compara com Kepler
* Newton hipérbole (flyby) — conserva energia/L
* Schwarzschild circular r=8M — estável
* Schwarzschild near-ISCO r≈6M — sensibilidade/limiar
* Schwarzschild r<6M (ex.: 5.5M) — instável/captura
* Schwarzschild weak-field (r grande) — precessão pequena e limite Newtoniano
* Caso extra: varredura (E,L) para mapa bound/unbound/capture
