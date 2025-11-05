package problems.mkp.solvers;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import metaheuristics.ga.AbstractGA;
import problems.Evaluator;
import problems.mkp.MKP_ORLib;
import solutions.Solution;

/**
 * GA para MKP com escolha de pais por k-means.
 *
 * Ideia (resumo): em cada geração (ou a cada Gc gerações), clusterizamos a população
 * (k clusters, por padrão k=2) pelos genótipos 0/1 (distância euclidiana em {0,1}^n,
 * equivalente à Hamming). Depois, montamos pares escolhendo 1 pai de um cluster e 1
 * do outro (clusters distintos), com torneio por fitness dentro de cada cluster.
 *
 * Referências/metodologia:
 * - Proposta (seção “Pais por k-means: clusters distintos”): clusterização periódica,
 *   amostragem de bits (opcional) p/ custo linear em npop, e cruzamentos forçados
 *   entre clusters distintos, priorizando elites locais quando desejado. 
 *   (vide MO824_proposta_247122). 
 * - Tese (Laabadi, 2020): fixa k=2, divide a população, e, para cada indivíduo do
 *   grupo menor, o parceiro é escolhido no outro grupo via torneio — aumenta diversidade
 *   sem perder exploração (intensificação). 
 *   (vide Thèse2020.pdf). 
 *
 * CLI (mesmos args do GA_MKP; params opcionais no fim):
 *   java -cp out problems.mkp.solvers.KMeansGA_MKP <path_orlib> <instanceIndex> <generations> <popSize> <mutationRate> <useRepair:true/false> [lambda]
 *                                           [k] [tourn] [maxIter] [bitSample] [clusterEveryG]
 *
 * Defaults:
 *   k = 2                (número de clusters)
 *   tourn = 3            (tamanho do torneio em cada cluster)
 *   maxIter = 10         (iterações do k-means)
 *   bitSample = 0        (0 => usa todos os loci; >0 => amostra 'bitSample' loci p/ distância)
 *   clusterEveryG = 1    (re-executa k-means a cada G gerações; 1 = toda geração)
 */
public class KMeansGA_MKP extends GA_MKP {

    private final int k;
    private final int tourn;
    private final int maxIter;
    private final int bitSample;
    private final int clusterEveryG;
    private int selectParentsCalls = 0;

    public KMeansGA_MKP(Evaluator<Integer> objFunction,
                         Integer generations,
                         Integer popSize,
                         Double mutationRate,
                         boolean useGreedyRepair,
                         int k, int tourn, int maxIter,
                         int bitSample, int clusterEveryG) {
        super(objFunction, generations, popSize, mutationRate, useGreedyRepair);
        this.k = Math.max(2, k);
        this.tourn = Math.max(2, tourn);
        this.maxIter = Math.max(1, maxIter);
        this.bitSample = Math.max(0, bitSample);
        this.clusterEveryG = Math.max(1, clusterEveryG);
    }

    /** Seleção de pais sobreposta: usa k-means em vez do torneio global. */
    @Override
    protected Population selectParents(Population population) {
        selectParentsCalls++;

        // Se não é “hora de clusterizar”, cai no baseline (torneio padrão).
        if ((selectParentsCalls % clusterEveryG) != 0) {
            return super.selectParents(population);
        }

        final int n = chromosomeSize;
        // Amostra de loci p/ reduzir custo (0 => usa todos)
        int[] loci = buildLociSample(n, bitSample);

        // --- K-means em {0,1}^n ---
        int[] assign = new int[popSize];
        double[][] centers = initCentersFromPopulation(population, k, n, loci);

        // Loop principal do k-means
        for (int it = 0; it < maxIter; it++) {
            boolean moved = assignToNearest(population, centers, assign, loci);

            // Recalcula centros; se algum cluster ficou vazio, repara.
            boolean emptyFixed = recomputeCenters(population, centers, assign, loci);

            if (!moved && !emptyFixed) break;
        }

        // Agrupa índices por cluster
        List<List<Integer>> byCluster = new ArrayList<>();
        for (int c = 0; c < k; c++) byCluster.add(new ArrayList<>());
        for (int i = 0; i < popSize; i++) byCluster.get(assign[i]).add(i);

        // Monta pais: pares entre clusters distintos, priorizando o menor cluster.
        Population parents = new Population();
        int targetParents = popSize; // mesmo contrato do framework

        // Estratégia: sempre escolher 1 pai do cluster menor e o 2º de outro cluster,
        // com torneio por fitness dentro de cada um.
        while (parents.size() < targetParents) {
            int cA = indexOfSmallestNonEmpty(byCluster);
            if (cA < 0) { // todos vazios? improvável, mas previne loop
                return super.selectParents(population);
            }
            int cB = pickDifferentNonEmptyCluster(byCluster, cA);
            if (cB < 0) { // sobrou cluster único não-vazio — fallback
                return super.selectParents(population);
            }

            AbstractGA<Integer, Integer>.Chromosome pA =
                tournamentWithin(population, byCluster.get(cA), tourn);
            AbstractGA<Integer, Integer>.Chromosome pB =
                tournamentWithin(population, byCluster.get(cB), tourn);

            parents.add(pA);
            if (parents.size() < targetParents) {
                parents.add(pB);
            }

            // Remoção opcional para “consumir” indivíduos do cluster menor primeiro:
            removeIndexOnce(byCluster.get(cA), population.indexOf(pA));
            removeIndexOnce(byCluster.get(cB), population.indexOf(pB));
        }

        return parents;
    }

    // ----------------- Helpers de K-means / torneio -----------------

    private int[] buildLociSample(int n, int sample) {
        if (sample <= 0 || sample >= n) {
            int[] all = new int[n];
            for (int i = 0; i < n; i++) all[i] = i;
            return all;
        }
        // sample aleatória sem reposição
        int[] idx = new int[n];
        for (int i = 0; i < n; i++) idx[i] = i;
        // Fisher-Yates parcial
        for (int i = 0; i < sample; i++) {
            int j = i + rng.nextInt(n - i);
            int tmp = idx[i]; idx[i] = idx[j]; idx[j] = tmp;
        }
        return Arrays.copyOf(idx, sample);
    }

    private double[][] initCentersFromPopulation(Population pop, int k, int n, int[] loci) {
        double[][] centers = new double[k][n];
        // sementes = k indivíduos aleatórios distintos
        int[] chosen = new int[k];
        Arrays.fill(chosen, -1);
        int filled = 0;
        while (filled < k) {
            int idx = rng.nextInt(pop.size());
            boolean used = false;
            for (int t = 0; t < filled; t++) if (chosen[t] == idx) { used = true; break; }
            if (!used) chosen[filled++] = idx;
        }
        for (int c = 0; c < k; c++) {
            AbstractGA<Integer,Integer>.Chromosome ch = pop.get(chosen[c]);
            for (int j : loci) centers[c][j] = (ch.get(j) != null && ch.get(j) != 0) ? 1.0 : 0.0;
        }
        return centers;
    }

    private boolean assignToNearest(Population pop, double[][] centers, int[] assign, int[] loci) {
        boolean moved = false;
        for (int i = 0; i < pop.size(); i++) {
            int bestC = -1;
            double bestD = Double.POSITIVE_INFINITY;
            for (int c = 0; c < centers.length; c++) {
                double d = sqDist(pop.get(i), centers[c], loci);
                if (d < bestD) { bestD = d; bestC = c; }
            }
            if (assign[i] != bestC) {
                if (assign[i] !=  -1) moved = true;
                assign[i] = bestC;
            }
        }
        return moved;
    }

    private boolean recomputeCenters(Population pop, double[][] centers, int[] assign, int[] loci) {
        int k = centers.length;
        int n = centers[0].length;
        double[][] sum = new double[k][n];
        int[] cnt = new int[k];

        for (int i = 0; i < pop.size(); i++) {
            int c = assign[i];
            cnt[c]++;
            AbstractGA<Integer,Integer>.Chromosome ch = pop.get(i);
            for (int j : loci) sum[c][j] += (ch.get(j) != null && ch.get(j) != 0) ? 1.0 : 0.0;
        }

        boolean emptyFixed = false;
        for (int c = 0; c < k; c++) {
            if (cnt[c] == 0) {
                // cluster vazio: re-semeia com um indivíduo aleatório
                int idx = rng.nextInt(pop.size());
                AbstractGA<Integer,Integer>.Chromosome ch = pop.get(idx);
                for (int j : loci) centers[c][j] = (ch.get(j) != null && ch.get(j) != 0) ? 1.0 : 0.0;
                emptyFixed = true;
            } else {
                for (int j : loci) centers[c][j] = sum[c][j] / cnt[c];
            }
        }
        return emptyFixed;
    }

    private double sqDist(AbstractGA<Integer,Integer>.Chromosome ch, double[] center, int[] loci) {
        double acc = 0.0;
        for (int j : loci) {
            double x = (ch.get(j) != null && ch.get(j) != 0) ? 1.0 : 0.0;
            double d = x - center[j];
            acc += d * d;
        }
        return acc;
    }

    private int indexOfSmallestNonEmpty(List<List<Integer>> byCluster) {
        int best = -1, bestSize = Integer.MAX_VALUE;
        for (int c = 0; c < byCluster.size(); c++) {
            int sz = byCluster.get(c).size();
            if (sz > 0 && sz < bestSize) { best = c; bestSize = sz; }
        }
        return best;
    }

    private int pickDifferentNonEmptyCluster(List<List<Integer>> byCluster, int avoid) {
        // tenta escolher o maior cluster != avoid, caso contrário o primeiro não vazio
        int best = -1, bestSize = -1;
        for (int c = 0; c < byCluster.size(); c++) {
            if (c == avoid) continue;
            int sz = byCluster.get(c).size();
            if (sz > bestSize) { best = c; bestSize = sz; }
        }
        return (bestSize > 0) ? best : -1;
    }

    private void removeIndexOnce(List<Integer> list, int idx) {
        for (int i = 0; i < list.size(); i++) {
            if (list.get(i) == idx) { list.remove(i); return; }
        }
    }

    /** Torneio restrito a um conjunto de índices (cluster). */
    private Chromosome tournamentWithin(Population pop, List<Integer> indices, int kTourn) {
        Chromosome best = null;
        double bestFit = Double.NEGATIVE_INFINITY;
        int n = indices.size();
        // Se o cluster tem poucos, degrade p/ torneio no tamanho possível
        int draws = Math.min(kTourn, Math.max(1, n));
        for (int r = 0; r < draws; r++) {
            int pos = indices.get(rng.nextInt(n));
            Chromosome cand = pop.get(pos);
            double fit = fitness(cand);
            if (fit > bestFit) { bestFit = fit; best = cand; }
        }
        // fallback (não deveria ocorrer)
        return (best != null) ? best : pop.get(indices.get(rng.nextInt(n)));
    }

    // ---------------- Runner de conveniência ----------------
    public static void main(String[] args) throws Exception {
        // java problems.mkp.solvers.KMeansGA_MKP <path_orlib> <instanceIndex> <generations> <popSize> <mutationRate> <useRepair:true/false> [lambda]
        //                                         [k] [tourn] [maxIter] [bitSample] [clusterEveryG]
        String path = (args.length >= 1) ? args[0] : "instances/mkp/mknapcb1.txt";
        int inst = (args.length >= 2) ? Integer.parseInt(args[1]) : 1;
        int generations = (args.length >= 3) ? Integer.parseInt(args[2]) : 500;
        int popSize = (args.length >= 4) ? Integer.parseInt(args[3]) : 100;
        double mut = (args.length >= 5) ? Double.parseDouble(args[4]) : 0.02;
        boolean repair = (args.length >= 6) ? Boolean.parseBoolean(args[5]) : true;

        MKP_ORLib evaluator =
            (args.length >= 7)
                ? new MKP_ORLib(path, inst, Double.parseDouble(args[6]))
                : new MKP_ORLib(path, inst); // λ automático

        int k = (args.length >= 8) ? Integer.parseInt(args[7]) : 2;
        int tourn = (args.length >= 9) ? Integer.parseInt(args[8]) : 3;
        int maxIter = (args.length >= 10) ? Integer.parseInt(args[9]) : 10;
        int bitSample = (args.length >= 11) ? Integer.parseInt(args[10]) : 0;
        int clusterEveryG = (args.length >= 12) ? Integer.parseInt(args[11]) : 1;

        KMeansGA_MKP ga = new KMeansGA_MKP(evaluator, generations, popSize, mut, repair,
                                            k, tourn, maxIter, bitSample, clusterEveryG);

        Solution<Integer> best = ga.solve();
        System.out.println("Best (cost=" + best.cost + "): " + best);
        if (evaluator.optimalFromFile != null && evaluator.optimalFromFile > 0) {
            System.out.println("Opt (file) = " + evaluator.optimalFromFile);
        }
    }
}
