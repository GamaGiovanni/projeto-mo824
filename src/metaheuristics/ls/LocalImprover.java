
package metaheuristics.ls;

import solutions.Solution;

/**
 * Local improver interface (e.g., Tabu Search).
 * Works at the phenotype level (Solution<F>).
 */
public interface LocalImprover<F> {
    /**
     * Improves the given solution under a time/step budget.
     * @param s starting solution (will not be mutated by contract; may return same ref)
     * @param maxSteps maximum number of move applications (<=0 => unlimited within deadline)
     * @param deadlineMillis absolute time (System.currentTimeMillis()) budget; 0 => no deadline
     * @return improved solution (may be the same if no improvement found)
     */
    Solution<F> improve(Solution<F> s, int maxSteps, long deadlineMillis);
}
