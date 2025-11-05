# Projeto MO824 — Metaheurísticas (GA) MKP

Este repositório contém uma base de **Algoritmo Genético (GA)** genérico em Java e um caso de uso: 

* **MKP** (Multidimensional Knapsack Problem) — leitura no formato da **OR-Library** e avaliação com penalização e *repair* opcional.
* **ISGA para MKP** — variação do GA com **seleção sexual** que prioriza diversidade por distância de Hamming, preservando o GA baseline para comparações.
* **KMeansGA para MKP** — variação do GA com **escolha de pais por k-means** (acasalamento entre *clusters* distintos), focada em manter diversidade estrutural com baixo custo.

> Curso-alvo / referência: MO824/MC859 (otimização/heurísticas).

---

## Visão geral

* **Framework GA** (`metaheuristics.ga.AbstractGA`): define o ciclo GA (seleção por torneio, crossover 2-pontos, mutação, elitismo) de forma abstrata. (Disponibilizado pelo professor)
* **MKP** (`problems.mkp`): inclui `MKP_ORLib` (parser e avaliador no formato OR-Library) e três solvers:

  * `problems.mkp.solvers.GA_MKP` — **baseline** com *repair* guloso opcional.
  * `problems.mkp.solvers.ISGA_MKP` — **ISGA** (seleção sexual) que **sobrescreve apenas a seleção de pais** do baseline.
  * `problems.mkp.solvers.KMeansGA_MKP` — **KMeansGA** (pais por k-means) que **clusteriza a população** e força acasalamento **entre clusters distintos**.

Características do GA para MKP (comuns aos solvers):

* Genótipo binário (0/1); fenótipo é o subconjunto de itens.
* Avaliação **penalizada**: lucro − λ·(soma dos excessos). `λ` pode ser automático (média dos lucros) ou definido via linha de comando.
* *Greedy repair* opcional (fenótipo): remove itens de pior densidade até tornar a solução viável (sem alterar o cromossomo).
* Custos incrementais O(m) na função-objetivo.

**O que o ISGA adiciona**:

* Seleção de fêmeas por **torneio** em fitness.
* Para cada fêmea, escolha de macho entre **k** candidatos por escore composto:
  [
  \text{score} = \alpha \cdot \text{fitness_normalizado} + (1-\alpha) \cdot \text{Hamming_normalizado}
  ]
* Objetivo: **aumentar diversidade** e reduzir convergência prematura mantendo o resto do pipeline (crossover, mutação, elitismo, *repair*) do baseline.

**O que o KMeansGA adiciona**:

* **Clustering periódico (k-means)** sobre cromossomos 0/1 (distância euclidiana ≡ Hamming) a cada (G_c) gerações.
* **Amostragem de bits opcional** para baratear o k-means em instâncias de alta dimensão.
* **Política de pais inter-cluster**: um pai de cada *cluster* distinto, com **torneio por fitness dentro do cluster** para equilibrar intensificação/diversificação.
* Integração “não intrusiva”: preserva crossover, mutação, elitismo e o mesmo módulo de restrições/*repair* do baseline.

---

## Arquitetura do repositório

```
projeto-mo824/
├── src/
│   ├── metaheuristics/
│   │   └── ga/
│   │       └── AbstractGA.java
│   ├── problems/
│   │   ├── mkp/
│   │   │   ├── MKP_ORLib.java
│   │   │   └── solvers/
│   │   │       ├── GA_MKP.java           # GA baseline
│   │   │       ├── ISGA_MKP.java         # GA com seleção sexual (ISGA)
│   │   │       └── KMeansGA_MKP.java     # GA com pais por k-means (clusters distintos)
│   └── solutions/
│       └── Solution.java
├── instances/
│   └── mkp/
│       ├── README.md
│       ├── mknap1.txt
│       ├── mknapcb1.txt
│       ├── mknapcb2.txt
│       └── mknapcb3.txt
└── README.md
```

### Pacotes principais

* `metaheuristics.ga`

  * `AbstractGA.java` — esqueleto do GA (genérico).

* `problems.mkp`

  * `MKP_ORLib.java` — avaliador/leitor no formato OR-Library (mknap1, mknapcb*).
  * `solvers/GA_MKP.java` — GA **baseline** para MKP (com *repair* opcional).
  * `solvers/ISGA_MKP.java` — **ISGA** (seleção sexual); altera apenas `selectParents`.
  * `solvers/KMeansGA_MKP.java` — **KMeansGA** (pais por k-means); altera apenas `selectParents`.

* `solutions.Solution` — estrutura genérica de solução (lista de elementos + custo).

---

## Instâncias (dados)

* Pasta `instances/mkp/` contém arquivos no **formato OR-Library** (ex.: `mknap1.txt`, `mknapcb1.txt`, …).
* O *README* dessa pasta resume o formato das instâncias e a origem (OR-Library).

---

## Como compilar

Pré-requisito: JDK 11+ (recomendado 17).

### Linux/macOS

```bash
mkdir -p out
find src -name "*.java" > sources.txt
javac --release 17 -d out @sources.txt
```

### Windows (PowerShell)

```powershell
mkdir out | Out-Null
Get-ChildItem -Recurse src -Filter *.java | ForEach-Object { $_.FullName } > sources.txt
javac --release 17 -d out @sources.txt
```

---

## Como executar

### GA baseline para MKP (OR-Library)

Classe principal: `problems.mkp.solvers.GA_MKP`

Parâmetros:

1. **caminho** do arquivo OR-Library (ex.: `instances/mkp/mknapcb1.txt`)
2. **índice** da instância no arquivo (1-based)
3. **geraçōes** (padrão 500)
4. **população** (padrão 100)
5. **taxa de mutação** (padrão 0.02)
6. **useRepair** (`true`/`false`, padrão `true`)
7. **λ** (opcional; se omitido, usa média dos lucros)

Exemplos:

```bash
# Linux/macOS
java -cp out problems.mkp.solvers.GA_MKP instances/mkp/mknapcb1.txt 1 1000 100 0.02 true
java -cp out problems.mkp.solvers.GA_MKP instances/mkp/mknap1.txt  3  800 150 0.03 false 500.0
```

### ISGA (seleção sexual) para MKP

Classe principal: `problems.mkp.solvers.ISGA_MKP`

Parâmetros (os 7 primeiros são idênticos ao baseline):
1–7. **iguais ao GA_MKP** (ver acima)
8. **alpha** *(opcional, padrão 0.5)* — peso do fitness vs. diversidade (Hamming)
9. **kMale** *(opcional, padrão 6)* — nº de candidatos a macho por fêmea
10. **tournF** *(opcional, padrão 3)* — tamanho do torneio para escolher fêmeas
11. **tournM** *(opcional, padrão 2)* — tamanho do torneio para pré-filtrar machos por fitness

Exemplos:

```bash
# Execução padrão do ISGA com lambda automático
java -cp out problems.mkp.solvers.ISGA_MKP instances/mkp/mknapcb1.txt 1 1000 100 0.02 true

# Mais peso para diversidade (alpha menor) e mais candidatos
java -cp out problems.mkp.solvers.ISGA_MKP instances/mkp/mknapcb1.txt 1 1000 100 0.02 true 0.4 8 3 2
```

### KMeansGA (pais por k-means) para MKP

Classe principal: `problems.mkp.solvers.KMeansGA_MKP`

Parâmetros (os 7 primeiros são idênticos ao baseline):
1–7. **iguais ao GA_MKP** (ver acima)
8. **k** *(opcional, padrão 2)* — número de *clusters* do k-means
9. **tourn** *(opcional, padrão 3)* — tamanho do torneio **dentro do cluster**
10. **maxIter** *(opcional, padrão 10)* — iterações do k-means por reagrupamento
11. **bitSample** *(opcional, padrão 0)* — nº de loci amostrados (0 = usa todos)
12. **clusterEveryG** *(opcional, padrão 1)* — reexecuta o k-means a cada G gerações

Exemplos:

```bash
# Execução padrão (k=2, torneio 3, k-means todo ciclo)
java -cp out problems.mkp.solvers.KMeansGA_MKP instances/mkp/mknapcb1.txt 1 1000 100 0.02 true

# k-means mais barato em alta dimensão (amostra 256 bits, reagrupa a cada 5 gerações)
java -cp out problems.mkp.solvers.KMeansGA_MKP instances/mkp/mknapcb1.txt 1 1000 100 0.02 true 2 3 10 256 5
```

**Importante:** o **GA baseline permanece inalterado**. ISGA e KMeansGA são **novas classes** que apenas trocam a política de seleção de pais, mantendo crossover, mutação, elitismo e *repair* do baseline — o que facilita comparações justas entre abordagens.

---

## Dicas de experimentação

* **ISGA**: varie `alpha ∈ [0.3, 0.7]` e `kMale ∈ {4,6,8,10}`; comece com `tournF=3`, `tournM=2`.
* **KMeansGA**:

  * `k=2` costuma ser um bom ponto de partida; teste `k ∈ {2,3}`.
  * Para n grande, use `bitSample` (p.ex., 128–512) para reduzir custo sem perder a ideia.
  * Ajuste `clusterEveryG` (p.ex., 2–5) para balancear custo de re-clusterização e benefício.
* Mantenha `useRepair=true` para estabilizar a viabilidade ao longo das gerações.
* Registre métricas por geração (melhor custo, média de Hamming, taxa de viabilidade, separação entre *clusters*) e avalie TTT quando aplicável.

---

## Roadmap (ideias)

* Mutação adaptativa em função da diversidade (Hamming) e/ou separação entre *clusters*.
* Estratégias de escolha de pais baseadas em **medoides/centroides** (KMeansGA).
* Análise de métricas mais complexas (TTT, *performance profiles*).

---

## Créditos & Referências

* **OR-Library**: conjunto de instâncias clássicas para o MKP.
* Base do GA/QBF inspirada em exemplos didáticos (ccavellucci, fusberti).
* Este projeto é para fins educacionais (MO824/MC859).

---
