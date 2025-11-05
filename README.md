# Projeto MO824 — Metaheurísticas (GA) MKP

Este repositório contém uma base de **Algoritmo Genético (GA)** genérico em Java e um caso de uso: 

* **MKP** (Multidimensional Knapsack Problem) — leitura no formato da **OR-Library** e avaliação com penalização e *repair* opcional.
* **ISGA para MKP** — variação do GA com **seleção sexual** que prioriza diversidade por distância de Hamming, preservando o GA baseline para comparações.

> Curso-alvo / referência: MO824/MC859 (otimização/heurísticas).

---

## Visão geral

* **Framework GA** (`metaheuristics.ga.AbstractGA`): define o ciclo GA (seleção por torneio, crossover 2-pontos, mutação, elitismo) de forma abstrata. (Disponibilizado pelo professor)
* **MKP** (`problems.mkp`): inclui `MKP_ORLib` (parser e avaliador no formato OR-Library) e dois solvers:

  * `problems.mkp.solvers.GA_MKP` — **baseline** com *repair* guloso opcional.
  * `problems.mkp.solvers.ISGA_MKP` — **ISGA** (seleção sexual) que **sobrescreve apenas a seleção de pais** do baseline.

Características do GA para MKP (ambos os solvers):

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
* Objetivo: **aumentar diversidade** e reduzir convergência prematura mantendo o resto do pipeline (crossover, mutação, elitismo, repair) do baseline.

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
│   │   │       ├── GA_MKP.java          # GA baseline (inalterado)
│   │   │       └── ISGA_MKP.java        # NOVO: GA com seleção sexual (ISGA)
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
  * `solvers/ISGA_MKP.java` — **ISGA** (seleção sexual), estende `GA_MKP` e altera apenas `selectParents`.

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
java -cp out problems.mkp.solvers.ISGA_MKP instances/mkp/mknapcb1.txt 1 1000 100 0.02 true \
  0.0   0.4  8  3  2
#       ^λ(auto) ^alpha ^kMale tournF tournM
```

**Importante:** o **GA baseline permanece inalterado**. O ISGA foi adicionado como **nova classe**, permitindo comparar facilmente **GA_MKP** vs **ISGA_MKP** apenas trocando a classe principal.

---

## Dicas de experimentação

* **Diversidade**: experimente `alpha ∈ [0.3, 0.7]` e `kMale ∈ {4,6,8,10}`.
* **Cenários mais “apertados”** (restrições mais rígidas) tendem a se beneficiar de **maior diversidade** (alpha menor).
* Mantenha `useRepair=true` para estabilizar a viabilidade ao longo das gerações.
* Registre métricas por geração (melhor custo, média de Hamming, taxa de viabilidade) para análises comparativas.

---

## Roadmap (ideias)

* *TS + Strategic oscillation* entre regiões viáveis/inviáveis (ajuste dinâmico de λ).
* Métricas de diversidade (ISGA) + **seleção por *k*-means** entre clusters distintos (alternativa ao torneio).
* Análise de métricas mais complexas (TTT, etc.).

---

## Créditos & Referências

* **OR-Library**: conjunto de instâncias clássicas para o MKP.
* Base do GA/QBF inspirada em exemplos didáticos (ccavellucci, fusberti).
* Este projeto é para fins educacionais (MO824/MC859).

---
