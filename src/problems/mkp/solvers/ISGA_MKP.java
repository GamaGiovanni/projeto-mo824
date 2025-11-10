package problems.mkp.solvers;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

import problems.Evaluator;
import solutions.Solution;
import utils.MetricsLogger;
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
            int tsSteps = getInt(cli, new String[]{"ts-steps","steps"}, 500);
            double vmin = getDouble(cli, new String[]{"vmin"}, 0.0);
            double vmax = getDouble(cli, new String[]{"vmax"}, 100.0);
            double lmbMin = getDouble(cli, new String[]{"lmbMin"}, 0.1);
            double lmbMax = getDouble(cli, new String[]{"lmbMax"}, 10000.0);
            double up = getDouble(cli, new String[]{"up"}, 1.2);
            double down = getDouble(cli, new String[]{"down"}, 0.9);

            problems.mkp.TabuSO_MKP ts = new problems.mkp.TabuSO_MKP(
                    evaluator, tenure, lmbMin, lmbMax, up, down, vmin, vmax);
            ga.setImprover(ts);
        }

        // === parâmetros de execução/registro (opcionais por CLI) ===
        String resultsDir = cli.getOrDefault("results-dir", "results");
        String algo = cli.getOrDefault("algo", ISGA_MKP.class.getSimpleName());
        String variant = cli.getOrDefault("variant", buildAutoVariant(mut, repair, tsOn, alpha, kMale, tournF, tournM, cli));
        long seed = Long.parseLong(cli.getOrDefault("seed", "0"));
        String mkcbresPath = cli.get("mkcbres"); // pode vir por CLI; senão tentamos inferir abaixo

        // === dataset/fileTag e mkcbres ===
        Path orlib = Paths.get(path);
        String fileTag = stripExt(orlib.getFileName().toString()); // "mknapcb3"
        String datasetId = "ORLIB";

        if (mkcbresPath == null) {
            // tenta mkcbres ao lado do mknapcbX.txt; se não existir, usa "mkcbres.txt" na raiz
            Path guess = (orlib.getParent() != null)
                    ? orlib.getParent().resolve("mkcbres.txt")
                    : Paths.get("mkcbres.txt");
            mkcbresPath = Files.exists(guess) ? guess.toString() : "mkcbres.txt";
        }

        // === ler BK e LP da referência e injetar no GA ===
        double bk = Double.NaN, ublp = Double.NaN;
        try {
            Ref ref = readBKLP(mkcbresPath, fileTag, inst);
            if (ref != null) { bk = ref.bk; ublp = ref.ublp; }
        } catch (Exception ex) {
            System.err.println("Aviso: falha ao ler mkcbres ("+mkcbresPath+"): " + ex.getMessage());
        }
        ga.setBenchmark(bk, ublp);
        ga.setRunInfo(datasetId, fileTag, inst, algo, variant, seed);

        // === preparar logger e delegar logging ao AbstractGA.solve() ===
        Files.createDirectories(Paths.get(resultsDir));
        try (MetricsLogger logger = new MetricsLogger(
                Paths.get(resultsDir, "results_runs.csv"),
                Paths.get(resultsDir, "results_gens.csv"),
                true)) {
            ga.setMetricsLogger(logger);
            Solution<Integer> best = ga.solve();
            System.out.println(best.cost);
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

    private static String stripExt(String s) {
        int dot = s.lastIndexOf('.');
        return (dot >= 0) ? s.substring(0, dot) : s;
    }

    /** Monta um 'variant' legível se não vier por CLI — útil nos CSVs. */
    private static String buildAutoVariant(double mut, boolean repair, boolean tsOn,
                                           double alpha, int kMale, int tournF, int tournM,
                                           java.util.Map<String,String> cli) {
        StringBuilder sb = new StringBuilder();
        sb.append("mut=").append(String.format(java.util.Locale.US, "%.4f", mut));
        sb.append(";repair=").append(repair);
        sb.append(";alpha=").append(String.format(java.util.Locale.US, "%.2f", alpha));
        sb.append(";kMale=").append(kMale);
        sb.append(";tournF=").append(tournF);
        sb.append(";tournM=").append(tournM);
        sb.append(";ts=").append(tsOn);
        // se TS estiver on, acrescenta alguns parâmetros relevantes (se existirem)
        if (tsOn) {
            if (cli.containsKey("tenure"))  sb.append(";tenure=").append(cli.get("tenure"));
            if (cli.containsKey("steps") || cli.containsKey("ts-steps"))
                sb.append(";steps=").append(cli.getOrDefault("ts-steps", cli.getOrDefault("steps","")));
            if (cli.containsKey("vmin"))    sb.append(";vmin=").append(cli.get("vmin"));
            if (cli.containsKey("vmax"))    sb.append(";vmax=").append(cli.get("vmax"));
            if (cli.containsKey("lmbMin"))  sb.append(";lmbMin=").append(cli.get("lmbMin"));
            if (cli.containsKey("lmbMax"))  sb.append(";lmbMax=").append(cli.get("lmbMax"));
            if (cli.containsKey("up"))      sb.append(";up=").append(cli.get("up"));
            if (cli.containsKey("down"))    sb.append(";down=").append(cli.get("down"));
        }
        return sb.toString();
    }

    private static final class Ref {
        final double bk, ublp;
        Ref(double bk, double ublp){ this.bk=bk; this.ublp=ublp; }
    }

    private static Ref readBKLP(String mkcbresPath, String fileTag, int idx) throws IOException {
        // mknapcb1..9  ->  blocos: 5.100, 5.250, 5.500, 10.100, 10.250, 10.500, 30.100, 30.250, 30.500
        // idx esperado: 1..30
        int X = Integer.parseInt(fileTag.replaceAll("\\D+", "")); // "mknapcb3" -> 3
        int pos = (X - 1) * 30 + (idx - 1); // 0..269

        java.util.List<Double> bests = new java.util.ArrayList<>(270);
        java.util.List<Double> lps    = new java.util.ArrayList<>(270);

        boolean inBK = false, inLP = false;

        // Linha de dados: "5.500-12   217534"  ou  "5.500-12   2.1761579702e+05"
        java.util.regex.Pattern row = java.util.regex.Pattern.compile(
            "^(\\d+\\.\\d+-\\d{2})\\s+([0-9.+\\-Ee]+)\\s*$"
        );

        try (java.io.BufferedReader br = java.nio.file.Files.newBufferedReader(java.nio.file.Paths.get(mkcbresPath))) {
            String ln;
            while ((ln = br.readLine()) != null) {
                ln = ln.trim();
                if (ln.isEmpty()) continue;

                // Detecta início de cada tabela
                if (ln.contains("Best Feasible Solution Value")) { inBK = true;  inLP = false; continue; }
                if (ln.contains("LP optimal"))                  { inBK = false; inLP = true;  continue; }

                java.util.regex.Matcher m = row.matcher(ln);
                if (!m.matches()) continue; // ignora cabeçalhos e textos

                // m.group(1) = "5.500-12" (não usamos para o 'pos' aqui)
                double val = Double.parseDouble(m.group(2));

                if (inBK) bests.add(val);
                else if (inLP) lps.add(val);
            }
        }

        if (pos < 0 || pos >= bests.size() || pos >= lps.size()) {
            throw new IOException("índice fora do mkcbres (pos=" + pos +
                                "; BK=" + bests.size() + ", LP=" + lps.size() + ")");
        }

        return new Ref(bests.get(pos), lps.get(pos));
    }

}
