package metaheuristics.ga;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.TimeUnit;

import problems.Evaluator;
import solutions.Solution;
import utils.MetricsLogger;

/**
 * Abstract class for metaheuristic GA (Genetic Algorithms). It consider the
 * maximization of the chromosome fitness.
 * 
 * @author ccavellucci, fusberti
 * @param <G>
 *            Generic type of the chromosome element (genotype).
 * @param <F>
 *            Generic type of the candidate to enter the solution (fenotype).
 */
public abstract class AbstractGA<G extends Number, F> {

	@SuppressWarnings("serial")
	public class Chromosome extends ArrayList<G> {
	}

	@SuppressWarnings("serial")
	public class Population extends ArrayList<Chromosome> {
	}

	/**
	 * Flag that indicates whether the code should print more information on screen.
	 */
	public static boolean verbose = true;

	/**
	 * A random number generator.
	 */
	public static final Random rng = new Random(0);

	/**
	 * The objective function being optimized.
	 */
	protected Evaluator<F> ObjFunction;

	/**
	 * The size of the population.
	 */
	protected int popSize;

	/**
	 * The size of the chromosome.
	 */
	protected int chromosomeSize;

	/**
	 * The probability of performing a mutation.
	 */
	protected double mutationRate;

	/**
	 * The best solution cost.
	 */
	protected Double bestCost;

	/**
	 * The best solution.
	 */
	protected Solution<F> bestSol;

	/**
	 * The best chromosome, according to its fitness evaluation.
	 */
	protected Chromosome bestChromosome;

	/**
	 * Maximum runtime in nanoseconds. Default: 30 minutes.
	 */
	protected long maxRuntimeNanos = TimeUnit.MINUTES.toNanos(30);

	/**
	 * Stagnation factor: maximum number of iterations without improvement = stagnationFactor * chromosomeSize.
	 * Default: 10.
	 */
	protected int stagnationFactor = 10;

	/**
	 * Sets the maximum runtime in milliseconds.
	 *
	 * @param maxMillis
	 *            The maximum runtime in milliseconds.
	 */
	public void setMaxRuntimeMillis(long maxMillis) {
		this.maxRuntimeNanos = TimeUnit.MILLISECONDS.toNanos(maxMillis);
	}
	/**
	 * Sets the stagnation factor.
	 *
	 * @param stagnationFactor
	 *            The stagnation factor (minimum value is 1).
	 */
	public void setStagnationFactor(int stagnationFactor) {
		if (stagnationFactor < 1) stagnationFactor = 1;
		this.stagnationFactor = stagnationFactor;
	}


 /**
  * Creates a new solution which is empty, i.e., does not contain any
  * candidate solution element.
  *
  * @return An empty solution.
  */
	public abstract Solution<F> createEmptySol();

	/**
	 * A mapping from the genotype (domain) to the fenotype (image). In other
	 * words, it takes a chromosome as input and generates a corresponding
	 * solution.
	 * 
	 * @param chromosome
	 *            The genotype being considered for decoding.
	 * @return The corresponding fenotype (solution).
	 */
	protected abstract Solution<F> decode(Chromosome chromosome);

	/**
	 * Generates a random chromosome according to some probability distribution
	 * (usually uniform).
	 * 
	 * @return A random chromosome.
	 */
	protected abstract Chromosome generateRandomChromosome();

	/**
	 * Determines the fitness for a given chromosome. The fitness should be a
	 * function strongly correlated to the objective function under
	 * consideration.
	 * 
	 * @param chromosome
	 *            The genotype being considered for fitness evaluation.
	 * @return The fitness value for the input chromosome.
	 */
	protected abstract Double fitness(Chromosome chromosome);

	/**
	 * Mutates a given locus of the chromosome. This method should be preferably
	 * called with an expected frequency determined by the {@link #mutationRate}.
	 * 
	 * @param chromosome
	 *            The genotype being mutated.
	 * @param locus
	 *            The position in the genotype being mutated.
	 */
	protected abstract void mutateGene(Chromosome chromosome, Integer locus);

	/**
	 * The constructor for the GA class.
	 * 
	 * @param objFunction
	 *            The objective function being optimized.
	 * @param popSize
	 *            Population size.
	 * @param mutationRate
	 *            The mutation rate.
	 */
	public AbstractGA(Evaluator<F> objFunction, Integer popSize, Double mutationRate) {
		this.ObjFunction = objFunction;
		this.popSize = popSize;
		this.chromosomeSize = this.ObjFunction.getDomainSize();
		this.mutationRate = mutationRate;
	}

	/**
	 * Best known solution value (BK).
	 */
	protected double bestKnown = Double.NaN;
	
	/**
	 * Upper bound from LP relaxation.
	 */
	protected double ubLP = Double.NaN;
	
	/**
	 * Dataset identifier (e.g., "ORLIB").
	 */
	protected String datasetId = "";
	
	/**
	 * File tag (e.g., "mknapcb3").
	 */
	protected String fileTag = "";
	
	/**
	 * Instance index (e.g., 17).
	 */
	protected int instanceIdx = 0;
	
	/**
	 * Algorithm name (e.g., "GA", "ISGA", "KMeansGA").
	 */
	protected String algo = "GA";
	
	/**
	 * Algorithm variant (e.g., flags, TS configuration).
	 */
	protected String variant = "";
	
	/**
	 * Random seed used.
	 */
	protected long seed = 0L;

	protected MetricsLogger metricsLogger;

	public void setBenchmark(double bk, double ubLP) { this.bestKnown = bk; this.ubLP = ubLP; }
	public void setRunInfo(String datasetId, String fileTag, int instanceIdx,
						String algo, String variant, long seed) {
	this.datasetId = datasetId; this.fileTag = fileTag; this.instanceIdx = instanceIdx;
	this.algo = algo; this.variant = variant; this.seed = seed;
	}
	public void setMetricsLogger(MetricsLogger logger) { this.metricsLogger = logger; }

	/**
	 * Converts a chromosome to a boolean array (bit = gene != 0).
	 *
	 * @param c
	 *            The chromosome to convert.
	 * @return A boolean array representation of the chromosome.
	 */
	protected boolean[] toBool(Chromosome c) {
		boolean[] b = new boolean[c.size()];
		for (int i = 0; i < c.size(); i++) b[i] = c.get(i).intValue() != 0;
		return b;
	}
	/**
	 * Converts a population to a list of boolean arrays.
	 *
	 * @param pop
	 *            The population to convert.
	 * @return A list of boolean array representations.
	 */
	protected List<boolean[]> toBoolPop(Population pop) {
		List<boolean[]> list = new java.util.ArrayList<>(pop.size());
		for (Chromosome c : pop) list.add(toBool(c));
		return list;
	}
	/**
	 * GA mainframe with stopping criteria:
	 * - Time limit (default 30 minutes), OR
	 * - Stagnation of (stagnationFactor * chromosomeSize) iterations without improvement.
	 *
	 * The algorithm starts by initializing a population of chromosomes.
	 * It then enters a generational loop, in which each generation performs the
	 * following steps: parent selection, crossover, mutation, population update,
	 * and best solution update.
	 *
	 * @return The best feasible solution obtained.
	 */
	public Solution<F> solve() {

		// Initialize the population
		Population population = initializePopulation();

		bestChromosome = getBestChromosome(population);
		bestSol = decode(bestChromosome);
		if (verbose) System.out.println("(Gen. 0) BestSol = " + bestSol);

		final long t0 = System.nanoTime();
		final long deadline = t0 + maxRuntimeNanos;
		final int stagnationLimit = Math.max(1, stagnationFactor * chromosomeSize);
		int noImprovement = 0;
		int g = 0;

		// Diversity accumulators (averages across generations)
		double accDivHamming = 0.0, accDivEntropy = 0.0;
		int accDivCount = 0;

		// Log for generation 0
		{
			List<boolean[]> popBits = toBoolPop(population);
			double divH = utils.Diversity.meanHammingToMedoid(popBits);
			double divE = utils.Diversity.meanLocusEntropy(popBits);
			accDivHamming += divH; accDivEntropy += divE; accDivCount++;
			if (metricsLogger != null) {
			long elapsedMs = (System.nanoTime() - t0) / 1_000_000L;
			metricsLogger.logGeneration(datasetId, fileTag, instanceIdx, algo, variant, seed,
										0, elapsedMs, bestSol.cost, divH, divE);
			}
		}

		// Main generational loop
		while (System.nanoTime() < deadline && noImprovement < stagnationLimit) {
			g++;

			Population parents = selectParents(population);
			Population offsprings = crossover(parents);
			Population mutants = mutate(offsprings);
			Population newpopulation = selectPopulation(mutants);

			population = newpopulation;
			postGenerationHook(population, g);
			bestChromosome = getBestChromosome(population);

			double currentBestFit = fitness(bestChromosome);
			if (currentBestFit > bestSol.cost) {
				bestSol = decode(bestChromosome);
				noImprovement = 0;
				if (verbose) System.out.println("(Gen. " + g + ") BestSol = " + bestSol);
			} else {
				noImprovement++;
			}

			// Diversity and logging for this generation
			List<boolean[]> popBits = toBoolPop(population);
			double divH = utils.Diversity.meanHammingToMedoid(popBits);
			double divE = utils.Diversity.meanLocusEntropy(popBits);
			accDivHamming += divH; accDivEntropy += divE; accDivCount++;

			if (metricsLogger != null) {
			long elapsedMs = (System.nanoTime() - t0) / 1_000_000L;
			metricsLogger.logGeneration(datasetId, fileTag, instanceIdx, algo, variant, seed,
										g, elapsedMs, bestSol.cost, divH, divE);
			}
		}

		long totalMs = (System.nanoTime() - t0) / 1_000_000L;
		String stopReason = (noImprovement >= stagnationLimit)
			? ("stagnation_" + stagnationLimit)
			: "time_limit";

		if (verbose) {
			System.out.println("Finished after " + g + " generations.");
			System.out.println("Best solution found: " + bestSol + " in time: " + (totalMs/1000.0) + " s.");
			System.out.println("Stopping criterion: " + stopReason + ".");
		}

		// Execution summary (profit and gaps)
		double f = bestSol.cost;
		double gapLPpct = (Double.isFinite(ubLP) && ubLP > 0.0) ? 100.0 * (ubLP - f) / ubLP : Double.NaN;
		double deltaBK = (Double.isFinite(bestKnown)) ? (f - bestKnown) : Double.NaN;
		double gapBKpct = (Double.isFinite(bestKnown) && Math.abs(bestKnown) > 0.0)
			? 100.0 * (bestKnown - f) / Math.abs(bestKnown)
			: Double.NaN;

		double divHavg = (accDivCount > 0) ? (accDivHamming / accDivCount) : Double.NaN;
		double divEavg = (accDivCount > 0) ? (accDivEntropy / accDivCount) : Double.NaN;

		if (metricsLogger != null) {
			MetricsLogger.RunSummary s = new MetricsLogger.RunSummary(
			datasetId, fileTag, instanceIdx,
			algo, variant, seed,
			g, totalMs,
			f, bestKnown, ubLP,
			gapLPpct, gapBKpct, deltaBK,
			divHavg, divEavg, stopReason
			);
			metricsLogger.logRunSummary(s);
		}

		return bestSol;
	}

	/**
	 * Randomly generates an initial population to start the GA.
	 * 
	 * @return A population of chromosomes.
	 */
	protected Population initializePopulation() {

		Population population = new Population();

		while (population.size() < popSize) {
			population.add(generateRandomChromosome());
		}

		return population;

	}

	/**
	 * Given a population of chromosome, takes the best chromosome according to
	 * the fitness evaluation.
	 * 
	 * @param population
	 *            A population of chromosomes.
	 * @return The best chromosome among the population.
	 */
	protected Chromosome getBestChromosome(Population population) {

		double bestFitness = Double.NEGATIVE_INFINITY;
		Chromosome bestChromosome = null;
		for (Chromosome c : population) {
			double fitness = fitness(c);
			if (fitness > bestFitness) {
				bestFitness = fitness;
				bestChromosome = c;
			}
		}

		return bestChromosome;
	}

	/**
	 * Given a population of chromosome, takes the worst chromosome according to
	 * the fitness evaluation.
	 * 
	 * @param population
	 *            A population of chromosomes.
	 * @return The worst chromosome among the population.
	 */
	protected Chromosome getWorseChromosome(Population population) {

		double worseFitness = Double.POSITIVE_INFINITY;
		Chromosome worseChromosome = null;
		for (Chromosome c : population) {
			double fitness = fitness(c);
			if (fitness < worseFitness) {
				worseFitness = fitness;
				worseChromosome = c;
			}
		}

		return worseChromosome;
	}

	/**
	 * Selection of parents for crossover using the tournament method. Given a
	 * population of chromosomes, randomly takes two chromosomes and compare
	 * them by their fitness. The best one is selected as parent. Repeat until
	 * the number of selected parents is equal to {@link #popSize}.
	 * 
	 * @param population
	 *            The current population.
	 * @return The selected parents for performing crossover.
	 */
	protected Population selectParents(Population population) {

		Population parents = new Population();

		while (parents.size() < popSize) {
			int index1 = rng.nextInt(popSize);
			Chromosome parent1 = population.get(index1);
			int index2 = rng.nextInt(popSize);
			Chromosome parent2 = population.get(index2);
			if (fitness(parent1) > fitness(parent2)) {
				parents.add(parent1);
			} else {
				parents.add(parent2);
			}
		}

		return parents;

	}

	/**
	 * The crossover step takes the parents generated by {@link #selectParents}
	 * and recombine their genes to generate new chromosomes (offsprings). The
	 * method being used is the 2-point crossover, which randomly selects two
	 * locus for being the points of exchange (P1 and P2). For example:
	 * 
	 *                        P1            P2
	 *    Parent 1: X1 ... Xi | Xi+1 ... Xj | Xj+1 ... Xn
	 *    Parent 2: Y1 ... Yi | Yi+1 ... Yj | Yj+1 ... Yn
	 * 
	 * Offspring 1: X1 ... Xi | Yi+1 ... Yj | Xj+1 ... Xn
	 * Offspring 2: Y1 ... Yi | Xi+1 ... Xj | Yj+1 ... Yn
	 * 
	 * @param parents
	 *            The selected parents for crossover.
	 * @return The resulting offsprings.
	 */
	protected Population crossover(Population parents) {

		Population offsprings = new Population();

		for (int i = 0; i < popSize; i = i + 2) {

			Chromosome parent1 = parents.get(i);
			Chromosome parent2 = parents.get(i + 1);

			int crosspoint1 = rng.nextInt(chromosomeSize + 1);
			int crosspoint2 = crosspoint1 + rng.nextInt((chromosomeSize + 1) - crosspoint1);

			Chromosome offspring1 = new Chromosome();
			Chromosome offspring2 = new Chromosome();

			for (int j = 0; j < chromosomeSize; j++) {
				if (j >= crosspoint1 && j < crosspoint2) {
					offspring1.add(parent2.get(j));
					offspring2.add(parent1.get(j));
				} else {
					offspring1.add(parent1.get(j));
					offspring2.add(parent2.get(j));
				}
			}

			offsprings.add(offspring1);
			offsprings.add(offspring2);

		}

		return offsprings;

	}

	/**
	 * The mutation step takes the offsprings generated by {@link #crossover}
	 * and to each possible locus, perform a mutation with the expected
	 * frequency given by {@link #mutationRate}.
	 * 
	 * @param offsprings
	 *            The offsprings chromosomes generated by the
	 *            {@link #crossover}.
	 * @return The mutated offsprings.
	 */
	protected Population mutate(Population offsprings) {

		for (Chromosome c : offsprings) {
			for (int locus = 0; locus < chromosomeSize; locus++) {
				if (rng.nextDouble() < mutationRate) {
					mutateGene(c, locus);
				}
			}
		}

		return offsprings;
	}

	/**
	 * Updates the population that will be considered for the next GA
	 * generation. The method used for updating the population is the elitist,
	 * which simply takes the worse chromosome from the offsprings and replace
	 * it with the best chromosome from the previous generation.
	 * 
	 * @param offsprings
	 *            The offsprings generated by {@link #crossover}.
	 * @return The updated population for the next generation.
	 */
	protected Population selectPopulation(Population offsprings) {

		Chromosome worse = getWorseChromosome(offsprings);
		if (fitness(worse) < fitness(bestChromosome)) {
			offsprings.remove(worse);
			offsprings.add(bestChromosome);
		}

		return offsprings;
	}

	/**
	 * Optional hook that allows subclasses to modify the population after selection/elitism
	 * and before updating the generation's best solution (e.g., to run local search or TS).
	 * Default implementation is no-op.
	 *
	 * @param population
	 *            The current population.
	 * @param generation
	 *            The current generation number.
	 */
	protected void postGenerationHook(Population population, int generation) {
		// no-op by default
	}

}
