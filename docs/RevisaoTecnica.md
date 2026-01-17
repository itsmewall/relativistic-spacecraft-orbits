# Revisão Técnica Crítica — Dinâmica Orbital Relativística e Missão 6-DOF

Este documento registra a revisão técnica detalhada do repositório, com foco em elevar o projeto ao nível máximo de rigor acadêmico para dinâmica orbital relativística e simulação de missão completa (6-DOF + GNC). Ele complementa a avaliação oral e o relatório final da banca.

## 1) Mapa do Estado Atual

### Modelos e equações presentes
- **Newtoniano (2 corpos, plano)**: integração explícita em coordenadas cartesianas 2D com estado `[x, y, vx, vy]` e forças centrais 
  \( \ddot{\mathbf{r}} = -\mu \mathbf{r}/r^3 \). 
  Saídas incluem energia específica e momento angular específico. 
- **Schwarzschild equatorial (geodésicas em tempo próprio)**: sistema reduzido com estado `[t, r, \phi, p_r]`, usando 
  \( \dot{t} = E/(1-2M/r) \), \( \dot{\phi} = L/r^2 \), \( \dot{r} = p_r \), \( \dot{p}_r = -\tfrac{1}{2} dV_{eff}/dr \).
  Diagnóstico por constraint \( \epsilon = p_r^2 + V_{eff}(r) - E^2 \).

### Variáveis de estado e outputs
- Newton: `t`, `y=[x,y,vx,vy]`, `energy`, `h`, `status`.
- Schwarzschild: `tau`, `r`, `phi`, `tcoord`, `pr`, `epsilon`, `E_series`, `L_series`, `status`.

### Implementação e integração
- Integração **RK4 de passo fixo** é usada para ambos os modelos. Não há integradores adaptativos reais (apesar da existência de headers vazios para RK4/RK45).
- Há detecção de eventos rudimentar para **captura** (\( r \le 2M + \epsilon \)) e uma regra grosseira de escape (\( r > 10^6 M \)).

### Ferramentas Python e validação
- Validação automática com `cases.yaml`, geração de plots e `report.json`.
- Scripts auxiliares em `scripts/` referenciam API antiga e **não são compatíveis** com a API atual.

### Conformidade com boas práticas
- **Unidades**: há utilitário de unidades e conversões geométricas em `units.py`, mas não é integrado ao pipeline de simulação.
- **Reprodutibilidade**: existe validação automatizada, mas sem testes de regressão/CI e sem versão controlada de outputs.
- **Estrutura**: módulos vazios (`analysis.py`, `mission.py`, `report.py`, `plots.py`) indicam lacunas estruturais.

---

## 2) Falhas Críticas e Riscos (P0)

### Física e consistência
1. **Estado Schwarzschild incompleto e parametrização não física para cenários gerais**: 
   o estado é reduzido e assume `p_r = 0` como condição inicial (turning point), ignorando a direção real do movimento radial. Isso invalida cenários inbound/outbound definidos por `E, L` e `r0`. 
   **Correção**: permitir `sign(pr)` ou `pr0` no YAML; computar \( p_r = \pm \sqrt{E^2 - V_{eff}(r_0)} \) com escolha explícita. 
   **Validação**: reproduzir órbitas inbound/outbound simétricas com mesmo \(E,L\) e comparar turning points.

2. **Uso de \(E, L\) como parâmetros fixos sem checagem dinâmica**: 
   o integrador usa \(E\) e \(L\) constantes impostas, e apenas calcula \(\epsilon\) como diagnóstico. Isso não garante consistência da 4-velocidade nem do vínculo \(u_\mu u^\mu=-1\). 
   **Correção**: integrar equações completas com 4-velocidade e recalcular \(E, L\) ao longo da trajetória, medindo deriva. 
   **Validação**: monitorar deriva relativa de \(E\) e \(L\) e a norma \(u_\mu u^\mu\).

3. **Ausência de dinâmica de atitude (6-DOF) e acoplamento**: 
   não existe qualquer implementação de atitude, torque, controle ou referência ao body frame, inviabilizando o objetivo de missão completa. 
   **Correção**: implementar equações de Euler + quaternion, torque de atuadores e acoplamento com thrust no body frame. 
   **Validação**: testes de resposta a degrau e tracking de apontamento.

### Numérico e integradores
1. **Somente RK4 de passo fixo**: 
   não há controle de erro, nem detecção robusta de eventos (periastro, turning points, crossing). 
   **Correção**: implementar RK45/DOPRI5 com controle adaptativo, e event detection. 
   **Validação**: varredura de tolerâncias e teste de ordem (log-log erro vs dt/rtol).

2. **Headers de solvers vazios**: 
   `rk4.hpp` e `rk45.hpp` estão vazios, o que sugere arquitetura pretendida mas não implementada. 
   **Correção**: consolidar integradores em um módulo real, exposto ao Python.

### Validação
1. **Testes automatizados vazios**: 
   arquivos em `tests/` estão vazios, portanto não há cobertura CI nem regressão. 
   **Correção**: implementar testes mínimos que exercitem `validate.py` e chequem métricas.

2. **Scripts de plots quebrados**: 
   `scripts/plot_newton_suite.py` e `plot_schwarzschild_suite.py` usam API antiga (retorno `res.t`, `res.y`). 
   **Correção**: atualizar para a API atual (ou remover, se obsoletos).

---

## 3) Upgrades de Alto Nível Acadêmico (P1)

### Física GR avançada
- **Schwarzschild completo (t,r,\theta,\phi)**: 
  remover restrição ao plano equatorial; permitir \(\theta\) variável e avaliar precessão nodal. 
- **Formulação Hamiltoniana** para geodésicas com integradores geométricos (symplectic). 
- **Kerr equatorial e genérico**: 
  incluir frame dragging, constantes de Carter, ZAMO. 
- **Tetrads locais**: 
  necessário para conectar acelerações físicas e forças de thrust no referencial da sonda.

### Missão e propulsão
- **Manobras impulsivas (\(\Delta v\))** com planejamento de trajetória (Lambert/transfer). 
- **Low-thrust guidance**: integração contínua de thrust, otimização de perfil com constraints. 
- **Eventos de missão**: janelas, periastro, passagem por ISCO, transição bound/unbound.

### Atitude e GNC
- **Equações de Euler + quaternions** com controladores PD/LQR. 
- **Apontamento**: lei de pointing (body frame) para alvo científico, acoplado ao thrust. 
- **Estimativa (EKF)**: opcional, mas desejável para estado orbital/atitude e ruído.

---

## 4) Validação e Referências

### Testes mandatórios (por módulo)
- **Newtoniano**: 
  conservação de energia e momento angular; comparação com solução analítica Kepleriana; teste de convergência (dt, dt/2, dt/4). 
- **Schwarzschild**: 
  constraint \(\epsilon\), conservação de \(E,L\), raio ISCO, precessão do periélio, limite Newtoniano (r \(\gg\) 10M). 
- **Kerr**: 
  comparação do ISCO prograde/retrograde; verificação de frame dragging; comparação com referências clássicas.
- **Atitude/GNC**: 
  tracking de quaternion, estabilidade com controladores PD/LQR, erro RMS em apontamento, torque saturado.

### Referências clássicas
- **MTW** (Misner, Thorne, Wheeler): geodésicas e Schwarzschild/Kerr.
- **Wald**: formalismo GR e constantes de movimento. 
- **Chandrasekhar**: órbitas em buracos negros e equações efetivas.
- **Vallado/Curtis**: mecânica orbital clássica e Lambert.
- **Schaub & Junkins**: dinâmica e controle de atitude.

---

## 5) Plano de Ataque (Sprints)

### Sprint 0 — Arquitetura e padronização
**Entregáveis**: 
- especificação unificada de unidades e convenções; 
- YAML com schema formal; 
- CI rodando `validate.py` e testes básicos. 
**Critério de aceite**: 
- `python -m relorbit_py.validate` reproduz resultados em máquina limpa; 
- documentação atualizada.

### Sprint 1 — Robustez numérica
**Entregáveis**: 
- integrador RK45/DOPRI5 com controle adaptativo; 
- event detection (periastro, horizon, turning points). 
**Aceite**: 
- convergência de ordem confirmada; 
- eventos reproduzidos com erro tolerável.

### Sprint 2 — Física GR avançada
**Entregáveis**: 
- Schwarzschild completo; 
- Kerr equatorial. 
**Aceite**: 
- ISCO e precessão reproduzidos; 
- validação com referências.

### Sprint 3 — Missão e propulsão
**Entregáveis**: 
- impulsive \(\Delta v\); low-thrust guidance; planejamento de trajetória. 
**Aceite**: 
- cenários completos (inserção, ciência, saída) reproduzidos.

### Sprint 4 — Atitude e GNC
**Entregáveis**: 
- 6-DOF; quaternions; controladores PD/LQR; apontamento. 
**Aceite**: 
- tracking com erro RMS abaixo do especificado; 
- acoplamento thrust-body validado.

---

## 6) Checklist Final de Banca

1. **Capítulos e figuras**: 
   - comparações Newton vs Schwarzschild (órbita, precessão, ISCO); 
   - mapa bound/unbound/capture; 
   - resultados de convergência numérica. 
2. **Tabelas**: 
   - métricas orbitais; 
   - erro de invariantes; 
   - comparação com soluções analíticas. 
3. **Discussão de erros**: 
   - sensibilidade numérica; 
   - limitações físicas; 
   - regime de validade. 
4. **Reprodutibilidade**: 
   - comandos exatos para rodar simulações e gerar figuras. 
5. **Demonstração ao vivo**: 
   - rodar `validate.py`; 
   - exibir plots principais e relatório gerado.
