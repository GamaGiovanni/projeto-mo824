# Projeto MO824 — Metaheurísticas (GA) MKP

Este repositório contém uma base de **Algoritmo Genético (GA)** genérico em Java e um caso de uso:

* **MKP** (Multidimensional Knapsack Problem) — leitura no formato da **OR-Library** e avaliação com penalização e *repair* opcional.
* **ISGA para MKP** — variação do GA com **seleção sexual** que prioriza diversidade por distância de Hamming, preservando o GA baseline para comparações.
* **KMeansGA para MKP** — variação do GA com **escolha de pais por k-means** (acasalamento entre *clusters* distintos), focada em manter diversidade estrutural com baixo custo.
* **(Novo)** **TS+SO para MKP** — **melhoria local** via Tabu Search com *Strategic Oscillation*, **plugável** em qualquer solver (baseline, ISGA, KMeans) por linha de comando.

> Curso-alvo / referência: MO824/MC859 (otimização/heurísticas).

---

## Visão geral

* **Framework GA** (`metaheuristics.ga.AbstractGA`): define o ciclo GA (seleção por torneio, crossover 2-pontos, mutação, elitismo) de forma abstrata.
* **Hook memético**: `AbstractGA` expõe `postGenerationHook(...)` — os solvers podem acoplar melhoria local (ex.: TS) **sem** alterar o pipeline do GA.
* **Módulo de melhoria local**:

  * `metaheuristics.ls.LocalImprover` — interface genérica para melhorias locais.
  * `problems.mkp.ls.TabuSO_MKP` — implementação de **Tabu Search + Strategic Oscillation** para MKP, usando deltas incrementais.
* **MKP** (`problems.mkp`): inclui `MKP_ORLib` (parser/avaliador OR-Library) e três solvers:

  * `problems.mkp.solvers.GA_MKP` — **baseline** com *repair* guloso opcional.
  * `problems.mkp.solvers.ISGA_MKP` — **ISGA** (seleção sexual) que **sobrescreve apenas a seleção de pais**.
  * `problems.mkp.solvers.KMeansGA_MKP` — **KMeansGA** (pais por k-means) que **clusteriza a população** e força acasalamento **entre clusters distintos**.

Características comuns (GA para MKP):

* Genótipo binário (0/1); fenótipo é o subconjunto de itens.
* Avaliação **penalizada**: lucro − λ·(soma dos excessos). `λ` pode ser **automático** (estimativa) ou definido via CLI.
* *Greedy repair* opcional: remove itens de pior densidade até viabilizar (sem alterar o cromossomo).
* Custos incrementais O(m) na avaliação (deltas de inserção/remoção).

**ISGA — o que adiciona**
Seleção de fêmeas por **torneio**; machos candidatos (**kMale**) ranqueados por escore composto:
[
\text{score} = \alpha \cdot \text{fitness_norm} + (1-\alpha)\cdot \text{Hamming_norm}
]
Aumenta diversidade e reduz convergência prematura, mantendo crossover/mutação/elitismo/*repair*.

**KMeansGA — o que adiciona**
**K-means periódico** em {0,1}^n (Hamming ≡ euclidiana), **amostragem de bits** opcional e **pais inter-cluster** (torneio dentro do cluster). Diversifica com baixo custo.

**TS+SO — como funciona (opcional)**
Tabu Search 1-flip com **lista tabu** (tenure) e **Strategic Oscillation** sobre a penalidade (λ): alterna controladamente entre regiões inviáveis/viáveis para escapar de platôs; integra-se como *LocalImprover* aos três solvers.

---

## Arquitetura do repositório

```
projeto-mo824/
├── src/
│   ├── metaheuristics/
│   │   ├── ga/
│   │   │   └── AbstractGA.java
│   │   └── ls/
│   │       └── LocalImprover.java                 # interface p/ melhorias locais
│   ├── problems/
│   │   ├── mkp/
│   │   │   ├── MKP_ORLib.java
│   │   │   ├── ls/
│   │   │   │   └── TabuSO_MKP.java               # TS+SO para MKP
│   │   │   └── solvers/
│   │   │       ├── GA_MKP.java                   # baseline
│   │   │       ├── ISGA_MKP.java                 # seleção sexual
│   │   │       └── KMeansGA_MKP.java             # pais por k-means
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

---

## Instâncias

* `instances/mkp/` contém arquivos no **formato OR-Library** (`mknap1.txt`, `mknapcb*.txt`).
* O `README.md` da pasta resume o formato e a origem.

---

## Compilar

Pré-requisito: **JDK 11+** (recomendado 17).

```bash
mkdir -p out
find src -name "*.java" > sources.txt
javac --release 17 -d out @sources.txt
```

Windows (PowerShell):

```powershell
mkdir out | Out-Null
Get-ChildItem -Recurse src -Filter *.java | ForEach-Object { $_.FullName } > sources.txt
javac --release 17 -d out @sources.txt
```

---

## Executar (agora com **parâmetros nomeados**)

Os três solvers aceitam `--chave valor` **ou** `--chave=valor` em **qualquer ordem**.
Se você não informar `--lambda`, é usado λ **automático**.

### GA baseline

Classe principal: `problems.mkp.solvers.GA_MKP`

Parâmetros principais:

* `--path` (arquivo OR-Library)
* `--instance|--inst` (índice 1-based)
* `--generations|--gens`
* `--pop|--popSize`
* `--mutation|--mut`
* `--repair` (`true/false`)
* `--lambda|--lam|--lmb` (opcional; usa automático se ausente)

**Exemplos**

```bash
# Baseline (λ automático)
java -cp out problems.mkp.solvers.GA_MKP \
  --path instances/mkp/mknapcb1.txt --instance 1 --generations 800 --pop 150 --mutation 0.02 --repair true

# Baseline (λ fixo)
java -cp out problems.mkp.solvers.GA_MKP \
  --path instances/mkp/mknap1.txt --inst 3 --gens 800 --pop 150 --mut 0.03 --repair false --lambda=500.0
```

### ISGA (seleção sexual)

Classe principal: `problems.mkp.solvers.ISGA_MKP`
Além dos parâmetros do baseline:

* `--alpha` (peso fitness vs. diversidade; **default 0.5**)
* `--kMale` (**default 6**)
* `--tournF|--tf` (**default 3**)
* `--tournM|--tm` (**default 2**)

**Exemplo**

```bash
java -cp out problems.mkp.solvers.ISGA_MKP \
  --path instances/mkp/mknapcb1.txt --inst 1 --gens 800 --pop 150 --mut 0.02 --repair true \
  --alpha 0.4 --kMale 8 --tournF 3 --tournM 2
```

### KMeansGA (pais por k-means)

Classe principal: `problems.mkp.solvers.KMeansGA_MKP`
Além dos parâmetros do baseline:

* `--k` (**default 2**)
* `--tourn` (**default 3**)
* `--maxIter` (**default 10**)
* `--bitSample` (**default 0** = usa todos os loci)
* `--clusterEveryG` (**default 1** = clusteriza toda geração)

**Exemplos**

```bash
# Padrão (k=2; k-means toda geração)
java -cp out problems.mkp.solvers.KMeansGA_MKP \
  --path instances/mkp/mknapcb1.txt --inst 1 --gens 800 --pop 150 --mut 0.02 --repair true

# Mais barato p/ n grande: amostra 256 bits; re-cluster a cada 5 gerações
java -cp out problems.mkp.solvers.KMeansGA_MKP \
  --path instances/mkp/mknapcb1.txt --inst 1 --gens 800 --pop 150 --mut 0.02 --repair true \
  --k 2 --tourn 3 --maxIter 10 --bitSample 256 --clusterEveryG 5
```

---

## (Opcional) Ativando **TS+SO** como melhoria local

Em **qualquer** solver (baseline, ISGA, KMeans), acrescente:

* `--ts true` (ou apenas passe algum parâmetro de TS);
* `--tenure <n>` (ex.: 7)
* `--ts-steps|--steps <n>` (passos por chamada, ex.: 500)
* `--vmin <v>` `--vmax <v>` (banda de violação para SO; ex.: `0.0` e `100.0`)
* `--lmbMin <λmin>` `--lmbMax <λmax>` (faixa da penalidade; ex.: `0.1` e `10000`)
* `--up <fator>` `--down <fator>` (ajuste multiplicativo de λ; ex.: `1.2` e `0.9`)

**Exemplo (ISGA + TS)**

```bash
java -cp out problems.mkp.solvers.ISGA_MKP \
  --path instances/mkp/mknapcb1.txt --inst 1 --gens 800 --pop 150 --mut 0.02 --repair true \
  --alpha 0.5 --kMale 6 --tournF 3 --tournM 2 \
  --ts true --tenure 7 --ts-steps 500 --vmin 0.0 --vmax 100.0 --lmbMin 0.1 --lmbMax 10000 --up 1.2 --down 0.9
```

**Guidelines rápidos**

* `tenure ≈ ⌈0.6√n⌉`
* `ts-steps`: 300–1000
* `vmin=0`, `vmax` pequeno (p.ex., 1–3% da soma de capacidades escaladas)
* `lmbMin/lmbMax`: faixa ampla; `up` em ~1.1–1.3; `down` em ~0.85–0.95

---

## Dicas de experimentação

* **ISGA**: varie `alpha ∈ [0.3, 0.7]` e `kMale ∈ {4,6,8,10}`; `tournF=3`, `tournM=2`.
* **KMeansGA**: `k ∈ {2,3}`; em n grande use `bitSample` (128–512) e `clusterEveryG ∈ {2,…,5}`.
* **TS+SO**: ative em 1 elite + 1 indivíduo diverso por geração (já configurado); ajuste `tenure`, `vmax` e `steps`.

---

## Roadmap

* Mutação adaptativa guiada por diversidade/viabilidade.
* Pais por medoides/centroides (KMeansGA).
* Avaliações TTT e *performance profiles*.

---

## Créditos & Referências

* **OR-Library** (instâncias MKP).
* Base de GA/QBF inspirada em exemplos didáticos (ccavellucci, fusberti).
* Projeto para fins educacionais (MO824/MC859).

---

### (Anexo) Migração rápida: de posicional → nomeado

Se você usava a execução **posicional**, mapeie:

* `<path_orlib>` → `--path`
* `<instanceIndex>` → `--instance`
* `<generations>` → `--generations`
* `<popSize>` → `--pop`
* `<mutationRate>` → `--mutation`
* `<useRepair>` → `--repair`
* `[lambda]` → `--lambda`

Parâmetros extras seguem os nomes das seções acima (`--alpha`, `--kMale`, `--k`, `--ts`, etc.).
