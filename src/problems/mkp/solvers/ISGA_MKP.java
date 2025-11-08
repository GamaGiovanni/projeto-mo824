package problems.mkp.solvers;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

import problems.Evaluator;
import solutions.Solution;
import metaheuristics.ls.LocalImprover;

/**
 * GA para MKP com Seleção Sexual (ISGA) que preserva o GA baseline original
 * (GA_MKP) e apenas sobrescreve a política de seleção de pais.
 *
 * Uso (mesmos argumentos do GA_MKP; parâmetros opcionais do ISGA ao final):
 *   java problems.mkp.solvers.ISGA_MKP <path_orlib> <instanceIndex> <popSize> <mutationRate> <useRepair:true/false> [lambda] [alpha] [kMale] [tournF] [tournM]
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
                    Integer popSize,
                    Double mutationRate,
                    boolean useGreedyRepair,
                    double alpha,
                    int kMale,
                    int tournF,
                    int tournM) {
        super(objFunction, popSize, mutationRate, useGreedyRepair);
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

    // ------------------------ Execução (Runner com parâmetros nomeados) ------------------------
    public static void main(String[] args) throws Exception {
        // Exemplos:
        //  Baseline ISGA:
        //  java problems.mkp.solvers.ISGA_MKP --path instances/mkp/mknapcb1.txt --instance 1 --pop 100 --mutation 0.02 --repair true \
        //       --alpha 0.5 --kMale 6 --tournF 3 --tournM 2
        //
        //  ISGA + TS:
        //  java problems.mkp.solvers.ISGA_MKP --path instances/mkp/mknapcb1.txt --instance 1 --pop 100 --mutation 0.02 --repair true \
        //       --alpha 0.5 --kMale 6 --tournF 3 --tournM 2 \
        //       --ts true --tenure 7 --ts-steps 500 --vmin 0.0 --vmax 100.0 --lmbMin 0.1 --lmbMax 10000 --up 1.2 --down 0.9

        if (hasFlag(args, "--help") || hasFlag(args, "-h")) {
            printHelp();
            return;
        }

        java.util.Map<String,String> cli = parseArgsToMap(args);

        // --- parâmetros base do GA (mesmos do GA_MKP) ---
        String path = cli.getOrDefault("path", "instances/mkp/mknapcb1.txt");
        int inst = getInt(cli, new String[]{"instance","inst"}, 1);
        int popSize = getInt(cli, new String[]{"pop","popSize"}, 100);
        double mut = getDouble(cli, new String[]{"mutation","mut"}, 0.02);
        boolean repair = getBool(cli, new String[]{"repair"}, true);

        // lambda opcional
        Double lambdaFixed = getNullableDouble(cli, new String[]{"lambda","lam","lmb"});
        problems.mkp.MKP_ORLib evaluator = (lambdaFixed != null)
                ? new problems.mkp.MKP_ORLib(path, inst, lambdaFixed)
                : new problems.mkp.MKP_ORLib(path, inst);

        // --- parâmetros do ISGA ---
        double alpha = getDouble(cli, new String[]{"alpha"}, 0.5);
        int kMale = getInt(cli, new String[]{"kMale","kmale"}, 6);
        int tournF = getInt(cli, new String[]{"tournF","tf","tF"}, 3);
        int tournM = getInt(cli, new String[]{"tournM","tm","tM"}, 2);

        ISGA_MKP ga = new ISGA_MKP(evaluator, popSize, mut, repair, alpha, kMale, tournF, tournM);

        // --- Tabu Search + Strategic Oscillation (opcional) ---
        boolean tsOn = getBool(cli, new String[]{"ts"}, false)
                || hasAny(cli, "tenure","ts-steps","steps","vmin","vmax","lmbMin","lmbMax","up","down");
        if (tsOn) {
            int tenure = getInt(cli, new String[]{"tenure"}, 7);
            int tsSteps = getInt(cli, new String[]{"ts-steps","steps"}, 500); // se quiser usar internamente, crie um setter
            double vmin = getDouble(cli, new String[]{"vmin"}, 0.0);
            double vmax = getDouble(cli, new String[]{"vmax"}, 100.0);
            double lmbMin = getDouble(cli, new String[]{"lmbMin"}, 0.1);
            double lmbMax = getDouble(cli, new String[]{"lmbMax"}, 10000.0);
            double up = getDouble(cli, new String[]{"up"}, 1.2);
            double down = getDouble(cli, new String[]{"down"}, 0.9);

            problems.mkp.TabuSO_MKP ts = new problems.mkp.TabuSO_MKP(
                    evaluator, tenure, lmbMin, lmbMax, up, down, vmin, vmax);
            ga.setImprover(ts);
            // opcional: ga.setLocalSearchSteps(tsSteps);
        }

        solutions.Solution<Integer> best = ga.solve();
        System.out.println("Best (cost=" + best.cost + "): " + best);
        if (evaluator.optimalFromFile != null && evaluator.optimalFromFile > 0) {
            System.out.println("Opt (file) = " + evaluator.optimalFromFile);
        }
    }

    protected static void printHelp() {
        System.out.println("Uso (parâmetros nomeados em qualquer ordem):");
        System.out.println("  --path <arquivo OR-Library>        (default: instances/mkp/mknapcb1.txt)");
        System.out.println("  --instance|--inst <id>             (default: 1)");
        System.out.println("  --pop|--popSize <n>                (default: 100)");
        System.out.println("  --mutation|--mut <p>               (default: 0.02)");
        System.out.println("  --repair <true/false>              (default: true)");
        System.out.println("  --lambda|--lam|--lmb <val>         (opcional; se ausente, lambda automático)");
        System.out.println();
        System.out.println("  --alpha <0..1>                     (default: 0.5)");
        System.out.println("  --kMale <n>                         (default: 6)");
        System.out.println("  --tournF|--tf <n>                  (default: 3)");
        System.out.println("  --tournM|--tm <n>                  (default: 2)");
        System.out.println();
        System.out.println("  --ts <true/false>                  (ativa TS+SO; também ativa se qualquer parâmetro de TS for passado)");
        System.out.println("  --tenure <n>                       (default: 7)");
        System.out.println("  --ts-steps|--steps <n>             (default: 500)");
        System.out.println("  --vmin <val>                       (default: 0.0)");
        System.out.println("  --vmax <val>                       (default: 100.0)");
        System.out.println("  --lmbMin <val>                     (default: 0.1)");
        System.out.println("  --lmbMax <val>                     (default: 10000.0)");
        System.out.println("  --up <fator>                       (default: 1.2)");
        System.out.println("  --down <fator>                     (default: 0.9)");
    }

}
