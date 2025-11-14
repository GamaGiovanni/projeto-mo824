package problems.mkp;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.io.Reader;
import java.io.StreamTokenizer;
import java.util.Arrays;

import problems.Evaluator;
import solutions.Solution;

/**
 * Multidimensional Knapsack Problem (MKP) in OR-Library format.
 *
 * Maximize:    sum_i p[i] * x_i
 * Subject to:  sum_i w[k][i] * x_i <= b[k],  k = 0..m-1
 *              x_i ∈ {0,1}
 *
 * Reads in mknap1/mknapcb* format: multiple instances in the same file
 * (see OR-Library page or ./instances). This class uses penalized evaluation to allow
 * infeasible solutions during the search: f(x) = profit(x) - λ * Σ_k over_k,
 * where over_k = max(0, load_k - b[k]).
 */
public class MKP_ORLib implements Evaluator<Integer> {

    /** Number of items. */
    public final Integer size;

    /** Number of constraints. */
    public final Integer m;

    /** 1-based index of the instance within the OR-Library file. */
    public final int problemIndex;

    /** Profits p[i]. */
    public final Double[] p;

    /** Weights w[k][i]. */
    public final Double[][] w;

    /** Capacities b[k]. */
    public final Double[] b;

    /** Optimal value reported in the file (0.0 if unavailable). */
    public final Double optimalFromFile;

    /** Binary variables (as Double, for compatibility with the infrastructure). */
    public final Double[] variables;

    /** Penalty factor λ. */
    public double penaltyFactor;

    // --- Accumulators for incremental evaluation ---
    private final Double[] accWeight; // sum of weights per constraint
    private Double accProfit;

    // ---------------------- Constructors ----------------------

    /**
     * Reads an OR-Library instance (mknap1 or mknapcb*) and selects the k-th (1..K).
     * λ (penaltyFactor) is automatically adjusted to the average of profits.
     */
    public MKP_ORLib(String filename, int instanceIndex) throws IOException {
        Loader out = readOrLibInstance(filename, instanceIndex);
        this.size = out.n;
        this.m = out.m;
        this.problemIndex = instanceIndex;
        this.p = out.p;
        this.w = out.w;
        this.b = out.b;
        this.optimalFromFile = out.optVal;

        this.variables = new Double[size];
        Arrays.fill(this.variables, 0.0);

        this.accWeight = new Double[m];
        Arrays.fill(this.accWeight, 0.0);
        this.accProfit = 0.0;

        this.penaltyFactor = defaultPenaltyFactor();
    }

    /**
     * Same as the previous constructor, but with λ provided.
     */
    public MKP_ORLib(String filename, int instanceIndex, double penaltyFactor) throws IOException {
        this(filename, instanceIndex);
        this.penaltyFactor = penaltyFactor;
    }

    // ------------------- Solution Evaluation -------------------

    @Override
    public Integer getDomainSize() { return size; }

    @Override
    public Double evaluate(Solution<Integer> sol) {
        setVariables(sol);
        return sol.cost = evaluateMKP();
    }

    /** Sets variables[] from sol and recalculates accumulators. */
    public void setVariables(Solution<Integer> sol) {
        resetVariables();
        resetAccumulators();
        if (!sol.isEmpty()) {
            for (Integer i : sol) {
                if (variables[i] == 0.0) {
                    variables[i] = 1.0;
                    accProfit += p[i];
                    for (int k = 0; k < m; k++) accWeight[k] += w[k][i];
                }
            }
        }
    }

    /** f(x) = profit - λ * sum of excesses. */
    public Double evaluateMKP() {
        double overSum = 0.0;
        for (int k = 0; k < m; k++) {
            double over = accWeight[k] - b[k];
            if (over > 0) overSum += over;
        }
        return accProfit - penaltyFactor * overSum;
    }

    // ---------------- Incremental Costs (O(m)) ----------------

    @Override
    public Double evaluateInsertionCost(Integer elem, Solution<Integer> sol) {
        setVariables(sol);
        return evaluateInsertionMKP(elem);
    }

    @Override
    public Double evaluateRemovalCost(Integer elem, Solution<Integer> sol) {
        setVariables(sol);
        return evaluateRemovalMKP(elem);
    }

    @Override
    public Double evaluateExchangeCost(Integer in, Integer out, Solution<Integer> sol) {
        setVariables(sol);
        return evaluateExchangeMKP(in, out);
    }

    /** Δ insert i. */
    public Double evaluateInsertionMKP(int i) {
        if (variables[i] == 1.0) return 0.0;
        double dProfit = p[i];
        double dPenalty = 0.0;
        for (int k = 0; k < m; k++) {
            double before = Math.max(0.0, accWeight[k] - b[k]);
            double after  = Math.max(0.0, accWeight[k] + w[k][i] - b[k]);
            dPenalty += (after - before);
        }
        return dProfit - penaltyFactor * dPenalty;
    }

    /** Δ remove i. */
    public Double evaluateRemovalMKP(int i) {
        if (variables[i] == 0.0) return 0.0;
        double dProfit = -p[i];
        double dPenalty = 0.0;
        for (int k = 0; k < m; k++) {
            double before = Math.max(0.0, accWeight[k] - b[k]);
            double after  = Math.max(0.0, accWeight[k] - w[k][i] - b[k]);
            dPenalty += (after - before);
        }
        return dProfit - penaltyFactor * dPenalty;
    }

    /** Δ swap out → in. */
    public Double evaluateExchangeMKP(int in, int out) {
        if (in == out) return 0.0;
        if (variables[in] == 1.0 && variables[out] == 1.0) return evaluateRemovalMKP(out);
        if (variables[in] == 0.0 && variables[out] == 0.0) return evaluateInsertionMKP(in);
        if (variables[in] == 1.0) return evaluateRemovalMKP(out);
        if (variables[out] == 0.0) return evaluateInsertionMKP(in);

        double dProfit = p[in] - p[out];
        double dPenalty = 0.0;
        for (int k = 0; k < m; k++) {
            double before = Math.max(0.0, accWeight[k] - b[k]);
            double after  = Math.max(0.0, accWeight[k] + w[k][in] - w[k][out] - b[k]);
            dPenalty += (after - before);
        }
        return dProfit - penaltyFactor * dPenalty;
    }

    // ------------------------- Utilities -------------------------

    /** Resets variables[]. */
    public void resetVariables() { Arrays.fill(variables, 0.0); }

    /** Resets accumulators. */
    private void resetAccumulators() {
        Arrays.fill(accWeight, 0.0);
        accProfit = 0.0;
    }

    /** Default λ: average of profits (good initial scale). */
    private double defaultPenaltyFactor() {
        double s = 0.0;
        for (double v : p) s += v;
        return (p.length > 0) ? s / p.length : 1.0;
    }

    /** Debug: prints the loaded instance. */
    public void printInstance() {
        System.out.println("MKP OR-Library: n=" + size + ", m=" + m + ", idx=" + problemIndex
                + ", opt(file)=" + optimalFromFile);
        System.out.println("p: " + Arrays.toString(p));
        for (int k = 0; k < m; k++) System.out.println("w[" + k + "]: " + Arrays.toString(w[k]));
        System.out.println("b: " + Arrays.toString(b));
    }

    // --------------------- OR-Library Reading ---------------------

    private static final class Loader {
        int n, m;
        Double[] p, b;
        Double[][] w;
        Double optVal;
    }

    /**
     * Reads an OR-Library file (mknap1 / mknapcb*) and returns the 'instanceIndex' (1..K) instance.
     * Format according to OR-Library documentation.
     */
    private Loader readOrLibInstance(String filename, int instanceIndex) throws IOException {
        if (instanceIndex <= 0) throw new IllegalArgumentException("instanceIndex must be >= 1");

        try (Reader r = new BufferedReader(new FileReader(filename))) {
            StreamTokenizer st = new StreamTokenizer(r);
            st.parseNumbers();

            // K: number of instances
            nextNum(st);
            int K = (int) st.nval;
            if (instanceIndex > K) {
                throw new IllegalArgumentException("instanceIndex=" + instanceIndex +
                        " > K=" + K + " in file " + filename);
            }

            Loader out = new Loader();

            // Iterates through instances until the desired one is found
            for (int k = 1; k <= K; k++) {
                nextNum(st); int n = (int) st.nval;
                nextNum(st); int m = (int) st.nval;
                nextNum(st); double opt = st.nval;

                if (k != instanceIndex) {
                    // Discards n profits
                    for (int i = 0; i < n; i++) nextNum(st);
                    // Discards m*n weights
                    for (int c = 0; c < m; c++) for (int i = 0; i < n; i++) nextNum(st);
                    // Discards m capacities
                    for (int c = 0; c < m; c++) nextNum(st);
                } else {
                    out.n = n; out.m = m; out.optVal = opt;
                    out.p = new Double[n];
                    out.w = new Double[m][n];
                    out.b = new Double[m];

                    for (int i = 0; i < n; i++) { nextNum(st); out.p[i] = st.nval; }
                    for (int c = 0; c < m; c++) {
                        for (int i = 0; i < n; i++) { nextNum(st); out.w[c][i] = st.nval; }
                    }
                    for (int c = 0; c < m; c++) { nextNum(st); out.b[c] = st.nval; }

                    return out;
                }
            }

            // If not returned, something went wrong
            throw new IOException("Failed to locate instance " + instanceIndex + " in " + filename);
        }
    }

    /** Reads the next numeric token (throws EOF if unexpected). */
    private static void nextNum(StreamTokenizer st) throws IOException {
        int tt = st.nextToken();
        if (tt == StreamTokenizer.TT_EOF)
            throw new IOException("Unexpected end of OR-Library file.");
        // If a non-numeric symbol comes, try to advance
        while (tt != StreamTokenizer.TT_NUMBER) {
            tt = st.nextToken();
            if (tt == StreamTokenizer.TT_EOF)
                throw new IOException("Unexpected end of OR-Library file.");
        }
    }

    // --------------------------- Main ---------------------------

    public static void main(String[] args) throws Exception {
        // Usage:
        // java MKP_ORLib path_to/mknapcb4.txt 17 [lambda]
        String path = (args.length >= 1) ? args[0] : "instances/orlib/mknap1.txt";
        int idx = (args.length >= 2) ? Integer.parseInt(args[1]) : 1;

        MKP_ORLib mkp = (args.length >= 3)
                ? new MKP_ORLib(path, idx, Double.parseDouble(args[2]))
                : new MKP_ORLib(path, idx);

        System.out.println("Loaded: n=" + mkp.size + ", m=" + mkp.m +
                ", idx=" + mkp.problemIndex + ", opt(file)=" + mkp.optimalFromFile);
        // Small test: random search
        Double best = Double.NEGATIVE_INFINITY;
        for (int it = 0; it < 50_000; it++) {
            Arrays.fill(mkp.variables, (Math.random() < 0.5) ? 0.0 : 1.0);
            mkp.resetAccumulators();
            for (int i = 0; i < mkp.size; i++) {
                if (mkp.variables[i] == 1.0) {
                    mkp.accProfit += mkp.p[i];
                    for (int k = 0; k < mkp.m; k++) mkp.accWeight[k] += mkp.w[k][i];
                }
            }
            double val = mkp.evaluateMKP();
            if (val > best) best = val;
        }
        System.out.println("Best (random + penalization) = " + best);
    }
}
