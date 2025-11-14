package problems.mkp.solvers;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import metaheuristics.ga.AbstractGA;
import problems.Evaluator;
import problems.mkp.MKP_ORLib;
import solutions.Solution;
import metaheuristics.ls.LocalImprover;
import utils.MetricsLogger;

/**
 * Genetic Algorithm for MKP with k-means based parent selection.
 *
 * Idea (summary): In each generation (or every Gc generations), the population is clustered
 * (k clusters, default k=2) based on 0/1 genotypes (Euclidean distance in {0,1}^n,
 * equivalent to Hamming distance). Then, pairs are formed by choosing 1 parent from one cluster
 * and 1 from another (distinct clusters), with tournament selection by fitness within each cluster.
 *
 * References/Methodology:
 * - Proposal (section "Parents by k-means: distinct clusters"): periodic clustering,
 *   bit sampling (optional) for linear cost in npop, and forced crossovers
 *   between distinct clusters, prioritizing local elites when desired.
 *   (see MO824_proposta_247122).
 * - Thesis (Laabadi, 2020): fixes k=2, divides the population, and, for each individual of
 *   the smaller group, the partner is chosen from the other group via tournament — increases diversity
 *   without losing exploration (intensification).
 *
 * CLI (same args as GA_MKP; optional params at the end):
 *   java -cp out problems.mkp.solvers.KMeansGA_MKP <path_orlib> <instanceIndex> <popSize> <mutationRate> <useRepair:true/false> [lambda]
 *                                           [k] [tourn] [maxIter] [bitSample] [clusterEveryG]
 *
 * Defaults:
 *   k = 2                (number of clusters)
 *   tourn = 3            (tournament size within each cluster)
 *   maxIter = 10         (k-means iterations)
 *   bitSample = 0        (0 => uses all loci; >0 => samples 'bitSample' loci for distance)
 *   clusterEveryG = 1    (re-executes k-means every G generations; 1 = every generation)
 */
public class KMeansGA_MKP extends GA_MKP {

    private final int k;
    private final int tourn;
    private final int maxIter;
    private final int bitSample;
    private final int clusterEveryG;
    private int selectParentsCalls = 0;

    public KMeansGA_MKP(Evaluator<Integer> objFunction,
                         Integer popSize,
                         Double mutationRate,
                         boolean useGreedyRepair,
                         int k, int tourn, int maxIter,
                         int bitSample, int clusterEveryG) {
        super(objFunction, popSize, mutationRate, useGreedyRepair);
        this.k = Math.max(2, k);
        this.tourn = Math.max(2, tourn);
        this.maxIter = Math.max(1, maxIter);
        this.bitSample = Math.max(0, bitSample);
        this.clusterEveryG = Math.max(1, clusterEveryG);
    }

    /** Overridden parent selection: uses k-means instead of global tournament. */
    @Override
    protected Population selectParents(Population population) {
        selectParentsCalls++;

        // If it's not "clustering time", falls back to baseline (standard tournament).
        if ((selectParentsCalls % clusterEveryG) != 0) {
            return super.selectParents(population);
        }

        final int n = chromosomeSize;
        // Loci sample to reduce cost (0 => uses all)
        int[] loci = buildLociSample(n, bitSample);

        // --- K-means in {0,1}^n ---
        int[] assign = new int[popSize];
        double[][] centers = initCentersFromPopulation(population, k, n, loci);

        // Main k-means loop
        for (int it = 0; it < maxIter; it++) {
            boolean moved = assignToNearest(population, centers, assign, loci);

            // Recalculates centers; if any cluster is empty, repairs it.
            boolean emptyFixed = recomputeCenters(population, centers, assign, loci);

            if (!moved && !emptyFixed) break;
        }

        // Groups indices by cluster
        List<List<Integer>> byCluster = new ArrayList<>();
        for (int c = 0; c < k; c++) byCluster.add(new ArrayList<>());
        for (int i = 0; i < popSize; i++) byCluster.get(assign[i]).add(i);

        // Forms parents: pairs between distinct clusters, prioritizing the smaller cluster.
        Population parents = new Population();
        int targetParents = popSize; // same framework contract

        // Strategy: always choose 1 parent from the smaller cluster and the 2nd from another cluster,
        // with tournament selection by fitness within each.
        while (parents.size() < targetParents) {
            int cA = indexOfSmallestNonEmpty(byCluster);
            if (cA < 0) { // all empty? unlikely, but prevents loop
                return super.selectParents(population);
            }
            int cB = pickDifferentNonEmptyCluster(byCluster, cA);
            if (cB < 0) { // only one non-empty cluster left — fallback
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

            // Optional removal to "consume" individuals from the smaller cluster first:
            removeIndexOnce(byCluster.get(cA), population.indexOf(pA));
            removeIndexOnce(byCluster.get(cB), population.indexOf(pB));
        }

        return parents;
    }

    // ----------------- K-means / Tournament Helpers -----------------

    private int[] buildLociSample(int n, int sample) {
        if (sample <= 0 || sample >= n) {
            int[] all = new int[n];
            for (int i = 0; i < n; i++) all[i] = i;
            return all;
        }
        // Random sample without replacement
        int[] idx = new int[n];
        for (int i = 0; i < n; i++) idx[i] = i;
        // Partial Fisher-Yates
        for (int i = 0; i < sample; i++) {
            int j = i + rng.nextInt(n - i);
            int tmp = idx[i]; idx[i] = idx[j]; idx[j] = tmp;
        }
        return Arrays.copyOf(idx, sample);
    }

    private double[][] initCentersFromPopulation(Population pop, int k, int n, int[] loci) {
        double[][] centers = new double[k][n];
        // Seeds = k distinct random individuals
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
                // Empty cluster: re-seed with a random individual
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

    /**
     * Finds the index of the smallest non-empty cluster.
     * @param byCluster A list of lists, where each inner list represents a cluster of indices.
     * @return The index of the smallest non-empty cluster, or -1 if all clusters are empty.
     */
    /**
     * Finds the index of the smallest non-empty cluster.
     * @param byCluster A list of lists, where each inner list represents a cluster of indices.
     * @return The index of the smallest non-empty cluster, or -1 if all clusters are empty.
     */
    /**
     * Finds the index of the smallest non-empty cluster.
     * @param byCluster A list of lists, where each inner list represents a cluster of indices.
     * @return The index of the smallest non-empty cluster, or -1 if all clusters are empty.
     */
    private int indexOfSmallestNonEmpty(List<List<Integer>> byCluster) {
        int best = -1, bestSize = Integer.MAX_VALUE;
        for (int c = 0; c < byCluster.size(); c++) {
            int sz = byCluster.get(c).size();
            if (sz > 0 && sz < bestSize) { best = c; bestSize = sz; }
        }
        return best;
    }

    /**
     * Picks a non-empty cluster different from the specified one.
     * Attempts to choose the largest cluster not equal to 'avoid', otherwise the first non-empty one.
     * @param byCluster A list of lists, where each inner list represents a cluster of indices.
     * @param avoid The index of the cluster to avoid.
     * @return The index of a different non-empty cluster, or -1 if no such cluster exists.
     */
    private int pickDifferentNonEmptyCluster(List<List<Integer>> byCluster, int avoid) {
        int best = -1, bestSize = -1;
        for (int c = 0; c < byCluster.size(); c++) {
            if (c == avoid) continue;
            int sz = byCluster.get(c).size();
            if (sz > bestSize) { best = c; bestSize = sz; }
        }
        return (bestSize > 0) ? best : -1;
    }

    /**
     * Removes the first occurrence of a given index from a list.
     * @param list The list from which to remove the index.
     * @param idx The index to be removed.
     */
    /**
     * Removes the first occurrence of a given index from a list.
     * @param list The list from which to remove the index.
     * @param idx The index to be removed.
     */
    /**
     * Removes the first occurrence of a given index from a list.
     * @param list The list from which to remove the index.
     * @param idx The index to be removed.
     */
    private void removeIndexOnce(List<Integer> list, int idx) {
        for (int i = 0; i < list.size(); i++) {
            if (list.get(i) == idx) { list.remove(i); return; }
        }
    }

    /**
     * Tournament selection restricted to a set of indices (cluster).
     * @param pop The population.
     * @param indices The indices within the population that belong to the current cluster.
     * @param kTourn The tournament size.
     * @return The best chromosome from the tournament.
     */
    private Chromosome tournamentWithin(Population pop, List<Integer> indices, int kTourn) {
        Chromosome best = null;
        double bestFit = Double.NEGATIVE_INFINITY;
        int n = indices.size();
        // If the cluster has few individuals, degrade to tournament of possible size
        int draws = Math.min(kTourn, Math.max(1, n));
        for (int r = 0; r < draws; r++) {
            int pos = indices.get(rng.nextInt(n));
            Chromosome cand = pop.get(pos);
            double fit = fitness(cand);
            if (fit > bestFit) { bestFit = fit; best = cand; }
        }
        // Fallback (should not occur)
        return (best != null) ? best : pop.get(indices.get(rng.nextInt(n)));
    }

   // ---------------- Runner with Named Parameters ----------------
    public static void main(String[] args) throws Exception {
        // Examples:
        //  Pure KMeans:
        //  java problems.mkp.solvers.KMeansGA_MKP --path instances/mkp/mknapcb1.txt --instance 1 --pop 100 --mutation 0.02 --repair true \
        //       --k 2 --tourn 3 --maxIter 10 --bitSample 0 --clusterEveryG 1
        //
        //  KMeans + TS:
        //  java problems.mkp.solvers.KMeansGA_MKP --path instances/mkp/mknapcb1.txt --inst 1 --gens 500 --pop 100 --mut 0.02 --repair true \
        //       --k 2 --tourn 3 --maxIter 10 --bitSample 0 --clusterEveryG 1 \
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

        // Optional lambda
        Double lambdaFixed = getNullableDouble(cli, new String[]{"lambda","lam","lmb"});
        MKP_ORLib evaluator = (lambdaFixed != null)
                ? new MKP_ORLib(path, inst, lambdaFixed)
                : new MKP_ORLib(path, inst);

        // --- K-means GA parameters ---
        int k = getInt(cli, new String[]{"k"}, 2);
        int tourn = getInt(cli, new String[]{"tourn"}, 3);
        int maxIter = getInt(cli, new String[]{"maxIter"}, 10);
        int bitSample = getInt(cli, new String[]{"bitSample"}, 0);
        int clusterEveryG = getInt(cli, new String[]{"clusterEveryG"}, 1);

        KMeansGA_MKP ga = new KMeansGA_MKP(
                evaluator, popSize, mut, repair,
                k, tourn, maxIter, bitSample, clusterEveryG
        );

        // --- Tabu Search + Strategic Oscillation (optional) ---
        boolean tsOn = getBool(cli, new String[]{"ts"}, false)
                || hasAny(cli, "tenure","ts-steps","steps","vmin","vmax","lmbMin","lmbMax","up","down");
        if (tsOn) {
            int tenure = getInt(cli, new String[]{"tenure"}, 7);
            int tsSteps = getInt(cli, new String[]{"ts-steps","steps"}, 500); // if you want to use, create a setter in GA
            double vmin = getDouble(cli, new String[]{"vmin"}, 0.0);
            double vmax = getDouble(cli, new String[]{"vmax"}, 100.0);
            double lmbMin = getDouble(cli, new String[]{"lmbMin"}, 0.1);
            double lmbMax = getDouble(cli, new String[]{"lmbMax"}, 10000.0);
            double up = getDouble(cli, new String[]{"up"}, 1.2);
            double down = getDouble(cli, new String[]{"down"}, 0.9);

            problems.mkp.TabuSO_MKP ts = new problems.mkp.TabuSO_MKP(
                    evaluator, tenure, lmbMin, lmbMax, up, down, vmin, vmax);
            ga.setImprover(ts);
            // Optional: ga.setLocalSearchSteps(tsSteps);
        }

        // === Execution/Logging Parameters (optional via CLI) ===
        String resultsDir = cli.getOrDefault("results-dir", "results");
        String algo = cli.getOrDefault("algo", KMeansGA_MKP.class.getSimpleName());
        String variant = cli.getOrDefault("variant", buildAutoVariant(mut, repair, tsOn, k, tourn, maxIter, bitSample, clusterEveryG, cli));
        long seed = Long.parseLong(cli.getOrDefault("seed", "0"));
        String mkcbresPath = cli.get("mkcbres"); // Can come via CLI; otherwise, we try to infer below

        // (Optional) Set seed if you adjusted AbstractGA to allow it
        // AbstractGA.setSeed(seed);

        // === Dataset/FileTag and mkcbres ===
        Path orlib = Paths.get(path);
        String fileTag = stripExt(orlib.getFileName().toString()); // "mknapcb3"
        String datasetId = "ORLIB";

        if (mkcbresPath == null) {
            // Tries mkcbres next to mknapcbX.txt; if it doesn't exist, uses "mkcbres.txt" in the root
            Path guess = (orlib.getParent() != null)
                    ? orlib.getParent().resolve("mkcbres.txt")
                    : Paths.get("mkcbres.txt");
            mkcbresPath = Files.exists(guess) ? guess.toString() : "mkcbres.txt";
        }

        // === Read BK and LP from reference and inject into GA ===
        double bk = Double.NaN, ublp = Double.NaN;
        try {
            Ref ref = readBKLP(mkcbresPath, fileTag, inst);
            if (ref != null) { bk = ref.bk; ublp = ref.ublp; }
        } catch (Exception ex) {
            System.err.println("Warning: failed to read mkcbres ("+mkcbresPath+"): " + ex.getMessage());
        }
        ga.setBenchmark(bk, ublp);
        ga.setRunInfo(datasetId, fileTag, inst, algo, variant, seed);

        // === Prepare logger and delegate logging to AbstractGA.solve() ===
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

    private static String stripExt(String s) {
        int dot = s.lastIndexOf('.');
        return (dot >= 0) ? s.substring(0, dot) : s;
    }

    /**
     * Builds a readable 'variant' string if not provided via CLI — useful for CSVs.
     * @param mut Mutation rate.
     * @param repair Repair flag.
     * @param tsOn Tabu Search on flag.
     * @param k Number of clusters.
     * @param tourn Tournament size.
     * @param maxIter Max iterations for k-means.
     * @param bitSample Bit sample size.
     * @param clusterEveryG Cluster every G generations.
     * @param cli Command line arguments map.
     * @return A string representing the variant.
     */
    private static String buildAutoVariant(double mut, boolean repair, boolean tsOn,
                                           int k, int tourn, int maxIter, int bitSample, int clusterEveryG,
                                           java.util.Map<String,String> cli) {
        StringBuilder sb = new StringBuilder();
        sb.append("mut=").append(String.format(java.util.Locale.US, "%.4f", mut));
        sb.append(";repair=").append(repair);
        sb.append(";k=").append(k);
        sb.append(";tourn=").append(tourn);
        sb.append(";maxIter=").append(maxIter);
        sb.append(";bitSample=").append(bitSample);
        sb.append(";clusterEveryG=").append(clusterEveryG);
        sb.append(";ts=").append(tsOn);
        // If TS is on, adds some relevant parameters (if they exist)
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

    /**
     * Reads Best Known (BK) and LP relaxation values from a mkcbres file.
     * @param mkcbresPath Path to the mkcbres file.
     * @param fileTag File tag (e.g., "mknapcb3").
     * @param idx Instance index.
     * @return A Ref object containing BK and UBLP values.
     * @throws IOException If an I/O error occurs or index is out of bounds.
     */
    private static Ref readBKLP(String mkcbresPath, String fileTag, int idx) throws IOException {
        // mknapcb1..9  ->  blocks: 5.100, 5.250, 5.500, 10.100, 10.250, 10.500, 30.100, 30.250, 30.500
        // Expected idx: 1..30
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

                // Detects the start of each table
                if (ln.contains("Best Feasible Solution Value")) { inBK = true;  inLP = false; continue; }
                if (ln.contains("LP optimal"))                  { inBK = false; inLP = true;  continue; }

                java.util.regex.Matcher m = row.matcher(ln);
                if (!m.matches()) continue; // Ignores headers and text

                // m.group(1) = "5.500-12" (not used for 'pos' here)
                double val = Double.parseDouble(m.group(2));

                if (inBK) bests.add(val);
                else if (inLP) lps.add(val);
            }
        }

        if (pos < 0 || pos >= bests.size() || pos >= lps.size()) {
            throw new IOException("Index out of mkcbres bounds (pos=" + pos +
                                 "; BK=" + bests.size() + ", LP=" + lps.size() + ")");
        }

        return new Ref(bests.get(pos), lps.get(pos));
    }

    /** Prints the help message for the CLI arguments. */
    protected static void printHelp() {
        System.out.println("Usage (named parameters in any order):");
        System.out.println("  --path <OR-Library file>             (default: instances/mkp/mknapcb1.txt)");
        System.out.println("  --instance|--inst <id>               (default: 1)");
        System.out.println("  --pop|--popSize <n>                  (default: 100)");
        System.out.println("  --mutation|--mut <p>                 (default: 0.02)");
        System.out.println("  --repair <true/false>                (default: true)");
        System.out.println("  --lambda|--lam|--lmb <val>           (optional; if absent, automatic lambda)");
        System.out.println();
        System.out.println("  --k <n>                              (default: 2)");
        System.out.println("  --tourn <n>                          (default: 3)");
        System.out.println("  --maxIter <n>                        (default: 10)");
        System.out.println("  --bitSample <n>                      (default: 0  => uses all loci)");
        System.out.println("  --clusterEveryG <n>                  (default: 1  => clusters every generation)");
        System.out.println();
        System.out.println("  --ts <true/false>                    (activates TS+SO; also activates if any TS parameter is passed)");
        System.out.println("  --tenure <n>                         (default: 7)");
        System.out.println("  --ts-steps|--steps <n>               (default: 500)");
        System.out.println("  --vmin <val>                         (default: 0.0)");
        System.out.println("  --vmax <val>                         (default: 100.0)");
        System.out.println("  --lmbMin <val>                       (default: 0.1)");
        System.out.println("  --lmbMax <val>                       (default: 10000.0)");
        System.out.println("  --up <factor>                        (default: 1.2)");
        System.out.println("  --down <factor>                      (default: 0.9)");
    }

}
