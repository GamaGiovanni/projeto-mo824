package problems.mkp.solvers;

import java.util.ArrayList;

import problems.Evaluator;
import problems.mkp.MKP_ORLib;
import solutions.Solution;
import metaheuristics.ga.AbstractGA;

public class GA_MKP extends AbstractGA<Integer, Integer> {

    /** Usa rotina de repair guloso (fenótipo), sem alterar o cromossomo. */
    private final boolean useGreedyRepair;

    public GA_MKP(Evaluator<Integer> objFunction,
                  Integer generations,
                  Integer popSize,
                  Double mutationRate,
                  boolean useGreedyRepair) {
        super(objFunction, generations, popSize, mutationRate);
        this.useGreedyRepair = useGreedyRepair;
    }

    // ---------------- AbstractGA hooks ----------------

    @Override
    public Solution<Integer> createEmptySol() {
        return new Solution<>();
    }

    @Override
    protected Solution<Integer> decode(Chromosome chromosome) {
        // Constrói solução a partir dos genes 0/1
        Solution<Integer> sol = createEmptySol();
        for (int i = 0; i < chromosomeSize; i++) {
            if (chromosome.get(i) != null && chromosome.get(i) != 0) {
                sol.add(i);
            }
        }

        // Aplica um "repair" guloso no fenótipo — não altera o cromossomo.
        if (useGreedyRepair && (ObjFunction instanceof MKP_ORLib)) {
            greedyRepairToFeasible(sol, (MKP_ORLib) ObjFunction);
        }

        // Avalia pela função-objetivo
        ObjFunction.evaluate(sol);
        return sol;
    }

    @Override
    protected Chromosome generateRandomChromosome() {
        Chromosome c = new Chromosome();
        for (int i = 0; i < chromosomeSize; i++) {
            c.add(rng.nextBoolean() ? 1 : 0);
        }
        return c;
    }

    @Override
    protected Double fitness(Chromosome chromosome) {
        // fitness = custo da solução decodificada (já calculado pela ObjFunction)
        return decode(chromosome).cost;
    }

    @Override
    protected void mutateGene(Chromosome chromosome, Integer locus) {
        int v = chromosome.get(locus);
        chromosome.set(locus, (v == 0) ? 1 : 0);
    }

    // ---------------- Greedy repair (fenótipo) ----------------
    /**
     * Torna a solução viável removendo itens de pior densidade p / (Σ_k w[k]/b[k]).
     * Não modifica o cromossomo original; apenas o fenótipo 'sol'.
     */
    private void greedyRepairToFeasible(Solution<Integer> sol, MKP_ORLib mkp) {
        if (sol.isEmpty()) return;

        final int m = mkp.m;
        final Double[] b = mkp.b;
        final Double[][] w = mkp.w;
        final Double[] p = mkp.p;

        // calcula cargas atuais
        double[] load = new double[m];
        for (int i : sol) {
            for (int k = 0; k < m; k++) load[k] += w[k][i];
        }

        // checa se já está viável
        if (isFeasible(load, b)) return;

        // cria lista mutável de itens selecionados
        ArrayList<Integer> items = new ArrayList<>(sol);

        // Enquanto houver violação, remove 1 item com pior densidade
        while (!isFeasible(load, b) && !items.isEmpty()) {
            int removeIdx = argMinDensity(items, p, w, b);
            int it = items.get(removeIdx);

            // atualiza cargas e remove
            for (int k = 0; k < m; k++) load[k] -= w[k][it];
            items.remove(removeIdx);
        }

        // reescreve a solução (fenótipo)
        sol.clear();
        sol.addAll(items);
    }

    private boolean isFeasible(double[] load, Double[] b) {
        for (int k = 0; k < load.length; k++) if (load[k] > b[k]) return false;
        return true;
    }

    /**
     * Retorna o índice (na lista 'items') do item com a pior densidade:
     * dens(i) = p[i] / (eps + Σ_k (w[k][i]/b[k])) — removemos o MENOR.
     */
    private int argMinDensity(ArrayList<Integer> items, Double[] p, Double[][] w, Double[] b) {
        final double EPS = 1e-12;
        int bestPos = 0;
        double bestVal = Double.POSITIVE_INFINITY;

        for (int pos = 0; pos < items.size(); pos++) {
            int i = items.get(pos);
            double denom = EPS;
            for (int k = 0; k < b.length; k++) denom += w[k][i] / b[k];
            double dens = p[i] / denom; // maior é melhor; queremos remover o menor
            if (dens < bestVal) {
                bestVal = dens;
                bestPos = pos;
            }
        }
        return bestPos;
    }

    // ---------------- Runner de conveniência ----------------
    public static void main(String[] args) throws Exception {
        // Exemplo de uso:
        // java problems.mkp.solvers.GA_MKP <path_orlib> <instanceIndex> <generations> <popSize> <mutationRate> <useRepair:true/false> [lambda]
        String path = (args.length >= 1) ? args[0] : "instances/mkp/mknapcb1.txt";
        int inst = (args.length >= 2) ? Integer.parseInt(args[1]) : 1;
        int generations = (args.length >= 3) ? Integer.parseInt(args[2]) : 500;
        int popSize = (args.length >= 4) ? Integer.parseInt(args[3]) : 100;
        double mut = (args.length >= 5) ? Double.parseDouble(args[4]) : 0.02;
        boolean repair = (args.length >= 6) ? Boolean.parseBoolean(args[5]) : true;

        MKP_ORLib evaluator =
            (args.length >= 7)
                ? new MKP_ORLib(path, inst, Double.parseDouble(args[6]))
                : new MKP_ORLib(path, inst); // lambda automático

        GA_MKP ga = new GA_MKP(evaluator, generations, popSize, mut, repair);

        Solution<Integer> best = ga.solve();
        System.out.println("Best (cost=" + best.cost + "): " + best);
        if (evaluator.optimalFromFile != null && evaluator.optimalFromFile > 0) {
            System.out.println("Opt (file) = " + evaluator.optimalFromFile);
        }
    }
}
