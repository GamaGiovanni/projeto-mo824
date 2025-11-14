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
 * GA for MKP with Sexual Selection (ISGA) that preserves the original baseline GA
 * (GA_MKP) and only overrides the parent selection policy.
 *
 * Usage (same arguments as GA_MKP; optional ISGA parameters at the end):
 *   java problems.mkp.solvers.ISGA_MKP <path_orlib> <instanceIndex> <popSize> <mutationRate> <useRepair:true/false> [lambda] [alpha] [kMale] [tournF] [tournM]
 * Defaults:
 *   alpha = 0.5 (weight for fitness vs. dissimilarity)
 *   kMale = 6   (candidate males sampled per female)
 *   tournF = 3  (tournament size for female selection)
 *   tournM = 2  (pre-filter by tournament for male candidates by fitness)
 */
public class ISGA_MKP extends GA_MKP {

    /** Weight for fitness (alpha) vs. dissimilarity (1 - alpha) in partner selection. */
    private final double alpha;
    /** Number of candidate males considered for each female. */
    private final int kMale;
    /** Tournament size when selecting females. */
    private final int tournF;
    /** Pre-filter tournament size for the set of males (by fitness). */
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

    // ------------------------ ISGA Core ------------------------
    @Override
    protected Population selectParents(Population population) {
        Population parents = new Population();

        // Pre-calculate minimum/maximum fitness for normalization.
        final double[] minFit = {Double.POSITIVE_INFINITY};
        double maxFit = Double.NEGATIVE_INFINITY;
        for (Chromosome c : population) {
            double f = fitness(c);
            if (f < minFit[0]) minFit[0] = f;
            if (f > maxFit) maxFit = f;
        }
        final double span = (maxFit > minFit[0]) ? (maxFit - minFit[0]) : 1.0;

        while (parents.size() < popSize) {
            // 1) Select a female by tournament based on fitness.
            Chromosome female = tournamentSelect(population, tournF);

            // 2) Build a set of candidate males: pre-filter by tournament (fitness)
            //    and then score by composite score (alpha * fitness + (1-alpha) * dissimilarity).
            List<Chromosome> malePool = new ArrayList<>();
            for (int i = 0; i < kMale; i++) {
                malePool.add(tournamentSelect(population, tournM));
            }

            // Avoid mating with itself if there are duplicates in the population.
            malePool.removeIf(m -> m == female);
            if (malePool.isEmpty()) {
                malePool.add(tournamentSelect(population, tournM));
            }

            // 3) Select the best male according to the composite score.
            Chromosome male = Collections.max(
                malePool,
                Comparator.comparingDouble(m -> compositeMateScore(female, m, minFit[0], span, alpha))
            );

            parents.add(female);
            parents.add(male);
        }

        return parents;
    }

    /** Tournament selection of size k based on fitness (higher is better). */
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

    /** Composite score: alpha * normFitness(male) + (1 - alpha) * normHamming(female, male). */
    private double compositeMateScore(Chromosome female,
                                      Chromosome male,
                                      double minFit,
                                      double span,
                                      double alpha) {
        double fNorm = (fitness(male) - minFit) / span;
        double dNorm = hamming01(female, male) / (double) chromosomeSize;
        return alpha * fNorm + (1.0 - alpha) * dNorm;
    }

    /** Hamming distance between binary chromosomes (Integer 0/1 genes). */
    private int hamming01(Chromosome a, Chromosome b) {
        int d = 0;
        for (int i = 0; i < chromosomeSize; i++) {
            int ai = (a.get(i) == null) ? 0 : a.get(i).intValue();
            int bi = (b.get(i) == null) ? 0 : b.get(i).intValue();
            if (ai != bi) d++;
        }
        return d;
    }

    // ------------------------ Execution (Runner with named parameters) ------------------------
    public static void main(String[] args) throws Exception {
        // Examples:
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

        // --- Base GA parameters (same as GA_MKP) ---
        String path = cli.getOrDefault("path", "instances/mkp/mknapcb1.txt");
        int inst = getInt(cli, new String[]{"instance","inst"}, 1);
        int popSize = getInt(cli, new String[]{"pop","popSize"}, 100);
        double mut = getDouble(cli, new String[]{"mutation","mut"}, 0.02);
        boolean repair = getBool(cli, new String[]{"repair"}, true);

        // optional lambda
        Double lambdaFixed = getNullableDouble(cli, new String[]{"lambda","lam","lmb"});
        problems.mkp.MKP_ORLib evaluator = (lambdaFixed != null)
                ? new problems.mkp.MKP_ORLib(path, inst, lambdaFixed)
                : new problems.mkp.MKP_ORLib(path, inst);

        // --- ISGA parameters ---
        double alpha = getDouble(cli, new String[]{"alpha"}, 0.5);
        int kMale = getInt(cli, new String[]{"kMale","kmale"}, 6);
        int tournF = getInt(cli, new String[]{"tournF","tf","tF"}, 3);
        int tournM = getInt(cli, new String[]{"tournM","tm","tM"}, 2);

        ISGA_MKP ga = new ISGA_MKP(evaluator, popSize, mut, repair, alpha, kMale, tournF, tournM);

        // --- Tabu Search + Strategic Oscillation (optional) ---
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

        // === Execution/logging parameters (optional via CLI) ===
        String resultsDir = cli.getOrDefault("results-dir", "results");
        String algo = cli.getOrDefault("algo", ISGA_MKP.class.getSimpleName());
        String variant = cli.getOrDefault("variant", buildAutoVariant(mut, repair, tsOn, alpha, kMale, tournF, tournM, cli));
        long seed = Long.parseLong(cli.getOrDefault("seed", "0"));
        String mkcbresPath = cli.get("mkcbres"); // can be passed via CLI; otherwise we try to infer it below

        // === dataset/fileTag and mkcbres ===
        Path orlib = Paths.get(path);
        String fileTag = stripExt(orlib.getFileName().toString()); // "mknapcb3"
        String datasetId = "ORLIB";

        if (mkcbresPath == null) {
            // try for mkcbres next to mknapcbX.txt; if it doesn't exist, use "mkcbres.txt" in the root
            Path guess = (orlib.getParent() != null)
                    ? orlib.getParent().resolve("mkcbres.txt")
                    : Paths.get("mkcbres.txt");
            mkcbresPath = Files.exists(guess) ? guess.toString() : "mkcbres.txt";
        }

        // === read BK and LP from the reference and inject into GA ===
        double bk = Double.NaN, ublp = Double.NaN;
        try {
            Ref ref = readBKLP(mkcbresPath, fileTag, inst);
            if (ref != null) { bk = ref.bk; ublp = ref.ublp; }
        } catch (Exception ex) {
            System.err.println("Warning: failed to read mkcbres ("+mkcbresPath+"): " + ex.getMessage());
        }
        ga.setBenchmark(bk, ublp);
        ga.setRunInfo(datasetId, fileTag, inst, algo, variant, seed);

        // === prepare logger and delegate logging to AbstractGA.solve() ===
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
        System.out.println("Usage (named parameters in any order):");
        System.out.println("  --path <OR-Library file>        (default: instances/mkp/mknapcb1.txt)");
        System.out.println("  --instance|--inst <id>             (default: 1)");
        System.out.println("  --pop|--popSize <n>                (default: 100)");
        System.out.println("  --mutation|--mut <p>               (default: 0.02)");
        System.out.println("  --repair <true/false>              (default: true)");
        System.out.println("  --lambda|--lam|--lmb <val>         (optional; if absent, automatic lambda)");
        System.out.println();
        System.out.println("  --alpha <0..1>                     (default: 0.5)");
        System.out.println("  --kMale <n>                         (default: 6)");
        System.out.println("  --tournF|--tf <n>                  (default: 3)");
        System.out.println("  --tournM|--tm <n>                  (default: 2)");
        System.out.println();
        System.out.println("  --ts <true/false>                  (activates TS+SO; also activates if any TS parameter is passed)");
        System.out.println("  --tenure <n>                       (default: 7)");
        System.out.println("  --ts-steps|--steps <n>             (default: 500)");
        System.out.println("  --vmin <val>                       (default: 0.0)");
        System.out.println("  --vmax <val>                       (default: 100.0)");
        System.out.println("  --lmbMin <val>                     (default: 0.1)");
        System.out.println("  --lmbMax <val>                     (default: 10000.0)");
        System.out.println("  --up <factor>                       (default: 1.2)");
        System.out.println("  --down <factor>                     (default: 0.9)");
    }

    private static String stripExt(String s) {
        int dot = s.lastIndexOf('.');
        return (dot >= 0) ? s.substring(0, dot) : s;
    }

    /** Assembles a readable 'variant' if not passed via CLI — useful for CSVs. */
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
        // if TS is on, add some relevant parameters (if they exist)
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
        // mknapcb1..9  ->  blocks: 5.100, 5.250, 5.500, 10.100, 10.250, 10.500, 30.100, 30.250, 30.500
        // expected idx: 1..30
        int X = Integer.parseInt(fileTag.replaceAll("\\D+", "")); // "mknapcb3" -> 3
        int pos = (X - 1) * 30 + (idx - 1); // 0..269

        java.util.List<Double> bests = new java.util.ArrayList<>(270);
        java.util.List<Double> lps    = new java.util.ArrayList<>(270);

        boolean inBK = false, inLP = false;

        // Data line: "5.500-12   217534"  or  "5.500-12   2.1761579702e+05"
        java.util.regex.Pattern row = java.util.regex.Pattern.compile(
            "^(\\d+\\.\\d+-\\d{2})\\s+([0-9.+\\-Ee]+)\\s*$"
        );

        try (java.io.BufferedReader br = java.nio.file.Files.newBufferedReader(java.nio.file.Paths.get(mkcbresPath))) {
            String ln;
            while ((ln = br.readLine()) != null) {
                ln = ln.trim();
                if (ln.isEmpty()) continue;

                // Detects start of each table
                if (ln.contains("Best Feasible Solution Value")) { inBK = true;  inLP = false; continue; }
                if (ln.contains("LP optimal"))                  { inBK = false; inLP = true;  continue; }

                java.util.regex.Matcher m = row.matcher(ln);
                if (!m.matches()) continue; // ignore headers and texts

                // m.group(1) = "5.500-12" (we don't use it for 'pos' here)
                double val = Double.parseDouble(m.group(2));

                if (inBK) bests.add(val);
                else if (inLP) lps.add(val);
            }
        }

        if (pos < 0 || pos >= bests.size() || pos >= lps.size()) {
            throw new IOException("index out of mkcbres bounds (pos=" + pos +
                                "; BK=" + bests.size() + ", LP=" + lps.size() + ")");
        }

        return new Ref(bests.get(pos), lps.get(pos));
    }

}
