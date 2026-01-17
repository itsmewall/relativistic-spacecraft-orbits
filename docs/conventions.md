# Convenções, Unidades e Símbolos (RelOrbit)

Este projeto implementa e valida simuladores numéricos para:
1) dinâmica orbital Newtoniana (2-corpos no plano), e
2) geodésicas equatoriais em Schwarzschild (Relatividade Geral),
com foco em extensões para missão completa próxima a buracos negros.

O objetivo deste documento é eliminar qualquer ambiguidade de unidades, sinais e definições. Nada fica implícito.

---

## 1. Sistema de unidades

### 1.1. Newton (2-corpos plano)
- **Unidades:** arbitrárias/normalizadas por caso, porém **consistentes** dentro de cada simulação.
- O parâmetro gravitacional do problema é **μ = G (m₁ + m₂)** (no limite m₂ ≪ m₁, μ ≈ G m₁).
- O estado é integrado em coordenadas cartesianas no plano:
  - **state0 = [x, y, v_x, v_y]**
- O tempo de integração usa coordenada **t** (tempo Newtoniano).

Observação: Nos casos de validação atuais, frequentemente usamos **μ = 1** como unidade natural do problema (não-SI), para facilitar checagem de invariantes.

### 1.2. Schwarzschild (equatorial)
- **Unidades geométricas (GR):** por padrão adotamos **G = c = 1**.
- Nestas unidades:
  - massa, comprimento e tempo têm a mesma dimensão,
  - o **raio do horizonte** é **r_h = 2M**,
  - o **ISCO** (órbita circular estável interna) ocorre em **r = 6M**.

O tempo de integração é o **tempo próprio τ** do corpo de prova.

---

## 2. Assinatura do métrico e coordenadas

### 2.1. Assinatura (Relatividade Geral)
Adotamos a assinatura:
- **(-, +, +, +)**

### 2.2. Coordenadas Schwarzschild
Usamos coordenadas (t, r, θ, φ), e restringimos ao plano equatorial:
- **θ = π/2** e **dθ/dτ = 0**

O elemento de linha (unidades geométricas) é:
- **ds² = -(1 - 2M/r) dt² + (1 - 2M/r)^{-1} dr² + r² dφ²**  (equatorial)

---

## 3. Grandezas físicas e definições formais

### 3.1. Newton: energia específica e momento angular específico
Para o problema Newtoniano planar com μ:

- **Energia específica (por unidade de massa):**
  - **E = v²/2 - μ/r**
  - onde r = √(x² + y²) e v² = v_x² + v_y²

- **Momento angular específico (componente z):**
  - **h = x v_y - y v_x**

**Classificação (Newtoniano):**
- **BOUND (ligada/elíptica)** se E < 0  
- **UNBOUND (não ligada: parabólica/hiperbólica)** se E ≥ 0

Obs.: o projeto também pode aplicar heurísticas adicionais (ex.: escape radial), mas esta é a definição física mínima.

### 3.2. Schwarzschild: constantes de movimento E e L
Para geodésicas timelike (partícula massiva), no plano equatorial, existem dois integrais de movimento:

- **Energia específica relativística (por unidade de massa de repouso):**
  - **E = (1 - 2M/r) dt/dτ**

- **Momento angular específico:**
  - **L = r² dφ/dτ**

Além disso, a normalização da 4-velocidade deve satisfazer:
- **g_{μν} u^μ u^ν = -1**
onde u^μ = dx^μ/dτ.

Uma forma equivalente (muito usada em validação) é o potencial efetivo:
- **(dr/dτ)² = E² - (1 - 2M/r) (1 + L²/r²)**

**Classificação (GR / Schwarzschild):**
- **BOUND:** r(τ) oscila entre periapse e apoapse sem cruzar horizonte.
- **CAPTURE:** trajetória cruza **r ≤ 2M** (horizonte) em tempo próprio finito.
- **UNBOUND:** r → ∞ mantendo-se fora do horizonte.

Observação importante: “CAPTURE” deve significar cruzamento físico do horizonte (ou critério equivalente), não apenas “aproximar e parar”.

---

## 4. Convenções de parâmetros do projeto

### 4.1. μ vs M (não confundir)
- **μ** (mu) é usado no **modelo Newtoniano** e representa o parâmetro gravitacional (G(m₁+m₂)).
- **M** é usado em **Schwarzschild** e representa a massa do buraco negro (em unidades geométricas, G=c=1).

Em simulações de teste, é comum usar μ=1 e M=1, mas eles pertencem a modelos físicos diferentes.

### 4.2. Estados e spans
- Newton:
  - estado inicial: **[x, y, v_x, v_y]**
  - integração em **t ∈ [t0, tf]**
- Schwarzschild (equatorial):
  - estado inicial mínimo: **[r0, φ0]** (e os parâmetros E e L definem dt/dτ, dφ/dτ e dr/dτ)
  - integração em **τ ∈ [τ0, τf]**

---

## 5. Saídas (outputs) e o que significam

### 5.1. Newton
O engine retorna séries temporais (t, y(t)) e também séries de invariantes calculadas:
- **energy(t)**: energia específica
- **h(t)**: momento angular específico
Validação típica: verificar drift relativo máximo de energy e h.

### 5.2. Schwarzschild equatorial
O engine retorna séries em τ:
- **r(τ), φ(τ)** e uma medida de consistência/constraint (ex.: epsilon)
Validação típica:
- medir o erro de normalização/constraint,
- verificar estabilidade (ex.: circular em r fixo),
- checar captura (cruzamento r ≤ 2M).

---

## 6. Conversão para SI (quando necessário)
Caso seja desejável interpretar resultados em unidades SI:

- Defina:
  - M_SI (kg), então o raio de Schwarzschild é:
    - r_s = 2 G M_SI / c²
- Se você usar unidades geométricas com M=1, então:
  - 1 unidade de comprimento geométrico = (G M_SI / c²) metros
  - 1 unidade de tempo geométrico = (G M_SI / c³) segundos

Este projeto pode operar em unidades geométricas internamente e converter para SI apenas na camada de pós-processamento/relatório.

---

## 7. Critério de aceitação (banca)
Este documento é considerado “aprovado” quando:
1) um leitor externo consegue responder, sem perguntar nada, o que são **μ, M, E, L, h**, e em quais unidades estão;
2) fica claro o que é integrado (t vs τ) e quais variáveis compõem o estado;
3) a assinatura do métrico e o plano equatorial estão declarados;
4) “CAPTURE” tem significado físico inequívoco (horizonte).
