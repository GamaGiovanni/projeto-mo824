# Projeto MO824 — Metaheurísticas (GA) MKP

Este repositório contém uma base de **Algoritmo Genético (GA)** genérico em Java e um caso de uso: 
- **MKP** (Multidimensional Knapsack Problem) — leitura no formato da **OR-Library** e avaliação com penalização e *repair* opcional.

> Curso-alvo / referência: MO824/MC859 (otimização/heurísticas).

---

## Visão geral

- **Framework GA** (`metaheuristics.ga.AbstractGA`): define o ciclo GA (seleção por torneio, crossover 2‑pontos, mutação, elitismo) de forma abstrata. (Disponibilizado pelo professor)
- **MKP** (`problems.mkp`): inclui `MKP_ORLib` (parser e avaliador no formato OR-Library) e `problems.mkp.solvers.GA_MKP` (GA concreto com *repair* guloso opcional).

Características do GA para MKP:
- Genótipo binário (0/1); fenótipo é o subconjunto de itens.
- Avaliação **penalizada**: lucro − λ·(soma dos excessos). `λ` pode ser automático (média dos lucros) ou definido via linha de comando.
- *Greedy repair* opcional (fenótipo): remove itens de pior densidade até tornar a solução viável (sem alterar o cromossomo).
- Custos incrementais O(m) na função-objetivo (útil para movimentos locais se necessário).

---

## Arquitetura do repositório

```
projeto-mo824/
├── src/
    ├── metaheuristics/
        ├── ga/
            ├── AbstractGA.java
├── problems/
    ├── mkp/
        ├── solvers/
            ├── GA_MKP.java
        ├── MKP_ORLib.java
    ├── Evaluator.java
├── solutions/
    ├── Solution.java
├── instances/
    ├── mkp/
        ├── README.md
        ├── mknap1.txt
        ├── mknapcb1.txt
        ├── mknapcb2.txt
        ├── mknapcb3.txt
└── README.md              # este arquivo
```

### Pacotes principais

- `metaheuristics.ga`
  - `AbstractGA.java` — esqueleto do GA (genérico).
- `problems.mkp`
  - `MKP_ORLib.java` — avaliador/leitor no formato OR-Library (mknap1, mknapcb*).
  - `solvers/GA_MKP.java` — GA concreto para MKP (com *repair* opcional).

- `solutions.Solution` — estrutura genérica de solução (lista de elementos + custo).

---

## Instâncias (dados)

- Pasta `instances/mkp/` contém arquivos no **formato OR-Library** (ex.: `mknap1.txt`, `mknapcb1.txt`, …).
- O *README* dessa pasta resume o formato das instâncias e a origem (OR-Library).

---

## Como compilar

Pré‑requisito: JDK 11+ (recomendado 17).

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

### GA para MKP (OR-Library)

Classe principal: `problems.mkp.solvers.GA_MKP`

Parâmetros:
1. **caminho** do arquivo OR-Library (ex.: `instances/mkp/mknapcb1.txt`)
2. **índice** da instância no arquivo (1‑based)
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

## Roadmap (ideias)

- *TS + Strategic oscillation* entre regiões viáveis/inviáveis (ajuste dinâmico de λ).
- Métricas de diversidade (*ISGA*) e *k*-means para seleção de pais (clusters distintos).
- Análise de métricas mais complexas (TTT, etc.)

---

## Créditos & Referências

- **OR-Library**: conjunto de instâncias clássicas para o MKP.
- Base do GA/QBF inspirada em exemplos didáticos (ccavellucci, fusberti).
- Este projeto é para fins educacionais (MO824/MC859).

---