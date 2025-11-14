
package problems.mkp;

import metaheuristics.ls.LocalImprover;
import solutions.Solution;

/**
 * Tabu Search + Strategic Oscillation (penalty-factor control) for MKP.
 * Uses MKP_ORLib's incremental deltas for 1-flip moves.
 */
public class TabuSO_MKP implements LocalImprover<Integer> {

    private final MKP_ORLib eval;
    private final int tenure;
    private final double lmbMin;
    private final double lmbMax;
    private final double upFactor;
    private final double downFactor;
    private final double vMin;
    private final double vMax;

    public TabuSO_MKP(MKP_ORLib eval,
                      int tenure,
                      double lmbMin, double lmbMax,
                      double upFactor, double downFactor,
                      double vMin, double vMax) {
        this.eval = eval;
        this.tenure = Math.max(1, tenure);
        this.lmbMin = lmbMin;
        this.lmbMax = lmbMax;
        this.upFactor = upFactor;
        this.downFactor = downFactor;
        this.vMin = vMin;
        this.vMax = vMax;
    }

    @Override
    public Solution<Integer> improve(Solution<Integer> s, int maxSteps, long deadlineMillis) {
        // Make a working copy.
        Solution<Integer> cur = new Solution<>();
        cur.addAll(s);
        eval.evaluate(cur); // Updates accumulators.

        double bestCost = cur.cost;
        Solution<Integer> best = new Solution<>();
        best.addAll(cur);

        final int n = eval.size;
        int[] tabu = new int[n];
        int step = 0;

        while ((maxSteps <= 0 || step < maxSteps) &&
               (deadlineMillis <= 0 || System.currentTimeMillis() < deadlineMillis)) {

            // Strategic Oscillation (SO) control on penalty factor.
            double v = currentViolationOf(cur);
            if (v > vMax)      eval.penaltyFactor = Math.min(lmbMax, Math.max(lmbMin, eval.penaltyFactor * upFactor));
            else if (v < vMin) eval.penaltyFactor = Math.min(lmbMax, Math.max(lmbMin, eval.penaltyFactor * downFactor));

            // Best admissible 1-flip.
            int bestMove = -1;
            double bestDelta = Double.NEGATIVE_INFINITY;

            // Work with eval's variables synced with 'cur'.
            eval.setVariables(cur);

            // Evaluate all 1-flip moves (O(n*m) but with deltas).
            for (int i = 0; i < n; i++) {
                double delta;
                if (cur.contains(i)) {
                    delta = eval.evaluateRemovalMKP(i);
                } else {
                    delta = eval.evaluateInsertionMKP(i);
                }
                boolean isTabu = (tabu[i] > step);
                boolean aspire = (cur.cost + delta > bestCost); // Aspiration by improving best.

                if (!isTabu || aspire) {
                    if (delta > bestDelta) {
                        bestDelta = delta;
                        bestMove = i;
                    }
                }
            }

            if (bestMove < 0 || bestDelta <= 0.0) {
                // No improving move; accept best admissible (even if worsening) to keep moving.
                bestMove = argBestWorseningMove(cur, tabu, step);
                if (bestMove < 0) break; // Stuck.
                bestDelta = (cur.contains(bestMove) ? eval.evaluateRemovalMKP(bestMove)
                                                    : eval.evaluateInsertionMKP(bestMove));
            }

            // Apply move.
            if (cur.contains(bestMove)) {
                cur.remove(Integer.valueOf(bestMove));
            } else {
                cur.add(bestMove);
            }
            eval.evaluate(cur); // Update cost.
            // Tabu tenure.
            tabu[bestMove] = step + tenure;

            // Track global best.
            if (cur.cost > bestCost) {
                bestCost = cur.cost;
                best.clear();
                best.addAll(cur);
            }

            step++;
        }

        return best;
    }

    private int argBestWorseningMove(Solution<Integer> cur, int[] tabu, int step) {
        eval.setVariables(cur);
        int n = eval.size;
        int arg = -1;
        double bestVal = Double.NEGATIVE_INFINITY;
        for (int i = 0; i < n; i++) {
            if (tabu[i] > step) continue; // Keep tabu for worsening steps.
            double delta = cur.contains(i) ? eval.evaluateRemovalMKP(i) : eval.evaluateInsertionMKP(i);
            if (delta > bestVal) {
                bestVal = delta;
                arg = i;
            }
        }
        return arg;
    }

    
    /**
     * Calculates the current violation of the solution.
     * @param cur The current solution.
     * @return The total violation.
     */
     private double currentViolationOf(Solution<Integer> cur) {
        double v = 0.0;
        final int m = eval.m;
        for (int k = 0; k < m; k++) {
            double load = 0.0;
            for (int idx : cur) {
                load += eval.w[k][idx];
            }
            double over = Math.max(0.0, load - eval.b[k]);
            v += over;
        }
        return v;
     }
}

