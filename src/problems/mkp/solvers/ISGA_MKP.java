package problems.mkp.solvers;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

import metaheuristics.ga.AbstractGA;
import problems.Evaluator;
import solutions.Solution;

/**
 * GA para MKP com Seleção Sexual (ISGA) que preserva o GA baseline original
 * (GA_MKP) e apenas sobrescreve a política de seleção de pais.
 *
 * Uso (mesmos argumentos do GA_MKP; parâmetros opcionais do ISGA ao final):
 *   java problems.mkp.solvers.ISGA_MKP <path_orlib> <instanceIndex> <generations> <popSize> <mutationRate> <useRepair:true/false> [lambda] [alpha] [kMale] [tournF] [tournM]
 * Padrões:
 *   alpha = 0,5 (peso para aptidão vs. dissimilaridade)
 *   kMale = 6   (machos candidatos amostrados por fêmea)
 *   tournF = 3  (tamanho do torneio para escolha da fêmea)
 *   tournM = 2  (pré-filtro por torneio para candidatos machos por aptidão)
 */
public class ISGA_MKP extends GA_MKP {

    /** Peso para aptidão (alpha) vs. dissimilaridade (1 - alpha) na escolha do parceiro. */
    private final double alpha;
    /** Número de machos candidatos considerados para cada fêmea. */
    private final int kMale;
    /** Tamanho do torneio ao selecionar fêmeas. */
    private final int tournF;
    /** Tamanho do pré-filtro por torneio do conjunto de machos (por aptidão). */
    private final int tournM;

    public ISGA_MKP(Evaluator<Integer> objFunction,
                    Integer generations,
                    Integer popSize,
                    Double mutationRate,
                    boolean useGreedyRepair,
                    double alpha,
                    int kMale,
                    int tournF,
                    int tournM) {
        super(objFunction, generations, popSize, mutationRate, useGreedyRepair);
        this.alpha = alpha;
        this.kMale = kMale;
        this.tournF = tournF;
        this.tournM = tournM;
    }

    // ------------------------ Núcleo do ISGA ------------------------
    @Override
    protected Population selectParents(Population population) {
        Population parents = new Population();

        // Pré-calcula aptidão mínima/máxima para normalização.
        final double[] minFit = {Double.POSITIVE_INFINITY};
        double maxFit = Double.NEGATIVE_INFINITY;
        for (Chromosome c : population) {
            double f = fitness(c);
            if (f < minFit[0]) minFit[0] = f;
            if (f > maxFit) maxFit = f;
        }
        final double span = (maxFit > minFit[0]) ? (maxFit - minFit[0]) : 1.0;

        while (parents.size() < popSize) {
            // 1) Escolhe uma fêmea por torneio baseado em aptidão.
            Chromosome female = tournamentSelect(population, tournF);

            // 2) Constrói um conjunto de machos candidatos: pré-filtro por torneio (aptidão)
            //    e, em seguida, pontua por escore composto (alpha * aptidão + (1-alpha) * dissimilaridade).
            List<Chromosome> malePool = new ArrayList<>();
            for (int i = 0; i < kMale; i++) {
                malePool.add(tournamentSelect(population, tournM));
            }

            // Evita acasalamento consigo mesma se houver duplicatas na população.
            malePool.removeIf(m -> m == female);
            if (malePool.isEmpty()) {
                malePool.add(tournamentSelect(population, tournM));
            }

            // 3) Escolhe o melhor macho segundo o escore composto.
            Chromosome male = Collections.max(
                malePool,
                Comparator.comparingDouble(m -> compositeMateScore(female, m, minFit[0], span, alpha))
            );

            parents.add(female);
            parents.add(male);
        }

        return parents;
    }

    /** Seleção por torneio de tamanho k baseada em aptidão (quanto maior, melhor). */
    private Chromosome tournamentSelect(Population pop, int k) {
        Chromosome best = null;
        double bestFit = Double.NEGATIVE_INFINITY;
        for (int i = 0; i < k; i++) {
            int idx = rng.nextInt(pop.size());
            Chromosome c = pop.get(idx);
            double f = fitness(c);
            if (f > bestFit) {
                bestFit = f;
                best = c;
            }
        }
        return best;
    }

    /** Score composto: alpha * normFitness(male) + (1 - alpha) * normHamming(female, male). */
    private double compositeMateScore(Chromosome female,
                                      Chromosome male,
                                      double minFit,
                                      double span,
                                      double alpha) {
        double fNorm = (fitness(male) - minFit) / span;
        double dNorm = hamming01(female, male) / (double) chromosomeSize;
        return alpha * fNorm + (1.0 - alpha) * dNorm;
    }

    /** Distância de Hamming entre cromossomos binários (genes Integer 0/1). */
    private int hamming01(Chromosome a, Chromosome b) {
        int d = 0;
        for (int i = 0; i < chromosomeSize; i++) {
            int ai = (a.get(i) == null) ? 0 : a.get(i).intValue();
            int bi = (b.get(i) == null) ? 0 : b.get(i).intValue();
            if (ai != bi) d++;
        }
        return d;
    }

    // ------------------------ Execução (Runner) ------------------------
    public static void main(String[] args) throws Exception {
        // Argumentos iguais ao GA_MKP, com parâmetros opcionais do ISGA ao final.
        // <path_orlib> <instanceIndex> <generations> <popSize> <mutationRate> <useRepair:true/false> [lambda] [alpha] [kMale] [tournF] [tournM]
        String path = (args.length >= 1) ? args[0] : "instances/mkp/mknapcb1.txt";
        int inst = (args.length >= 2) ? Integer.parseInt(args[1]) : 1;
        int generations = (args.length >= 3) ? Integer.parseInt(args[2]) : 500;
        int popSize = (args.length >= 4) ? Integer.parseInt(args[3]) : 100;
        double mut = (args.length >= 5) ? Double.parseDouble(args[4]) : 0.02;
        boolean repair = (args.length >= 6) ? Boolean.parseBoolean(args[5]) : true;

        // Lambda opcional (fator de penalidade) — igual ao main do GA_MKP.
        problems.mkp.MKP_ORLib evaluator =
            (args.length >= 7)
                ? new problems.mkp.MKP_ORLib(path, inst, Double.parseDouble(args[6]))
                : new problems.mkp.MKP_ORLib(path, inst); // lambda automático

        double alpha = (args.length >= 8) ? Double.parseDouble(args[7]) : 0.5;
        int kMale = (args.length >= 9) ? Integer.parseInt(args[8]) : 6;
        int tournF = (args.length >= 10) ? Integer.parseInt(args[9]) : 3;
        int tournM = (args.length >= 11) ? Integer.parseInt(args[10]) : 2;

        ISGA_MKP ga = new ISGA_MKP(evaluator, generations, popSize, mut, repair, alpha, kMale, tournF, tournM);
        Solution<Integer> best = ga.solve();
        System.out.println("Best (cost=" + best.cost + "): " + best);
        if (evaluator.optimalFromFile != null && evaluator.optimalFromFile > 0) {
            System.out.println("Opt (file) = " + evaluator.optimalFromFile);
        }
    }
}
