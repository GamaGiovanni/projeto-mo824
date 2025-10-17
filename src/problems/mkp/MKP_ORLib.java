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
 * Multidimensional Knapsack Problem (MKP) no formato OR-Library.
 *
 * Maximizar:    sum_i p[i] * x_i
 * s.a.:         sum_i w[k][i] * x_i <= b[k],  k = 0..m-1
 *               x_i ∈ {0,1}
 *
 * Leitura no formato de mknap1/mknapcb*: várias instâncias no mesmo arquivo
 * (ver página da OR-Library ou ./instances). Esta classe usa avaliação penalizada para permitir
 * soluções inviáveis durante a busca: f(x) = lucro(x) - λ * Σ_k over_k,
 * com over_k = max(0, carga_k - b[k]).
 */
public class MKP_ORLib implements Evaluator<Integer> {

    /** Número de itens. */
    public final Integer size;

    /** Número de restrições. */
    public final Integer m;

    /** Índice (1-based) da instância dentro do arquivo OR-Library. */
    public final int problemIndex;

    /** Lucros p[i]. */
    public final Double[] p;

    /** Pesos w[k][i]. */
    public final Double[][] w;

    /** Capacidades b[k]. */
    public final Double[] b;

    /** Valor ótimo informado no arquivo (0.0 se indisponível). */
    public final Double optimalFromFile;

    /** Variáveis binárias (como Double, p/ compatibilidade com a infra). */
    public final Double[] variables;

    /** Fator de penalidade λ. */
    public double penaltyFactor;

    // --- Acumuladores para avaliação incremental ---
    private final Double[] accWeight; // soma de pesos por restrição
    private Double accProfit;

    // ---------------------- Construtores ----------------------

    /**
     * Lê uma instância OR-Library (mknap1 ou mknapcb*) e seleciona a k-ésima (1..K).
     * λ (penaltyFactor) é ajustado automaticamente para a média dos lucros.
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
     * Mesmo que o anterior, mas com λ informado.
     */
    public MKP_ORLib(String filename, int instanceIndex, double penaltyFactor) throws IOException {
        this(filename, instanceIndex);
        this.penaltyFactor = penaltyFactor;
    }

    // ------------------- Avaliação da solução -------------------

    @Override
    public Integer getDomainSize() { return size; }

    @Override
    public Double evaluate(Solution<Integer> sol) {
        setVariables(sol);
        return sol.cost = evaluateMKP();
    }

    /** Seta variables[] a partir de sol e recalcula acumuladores. */
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

    /** f(x) = lucro - λ * soma dos excessos. */
    public Double evaluateMKP() {
        double overSum = 0.0;
        for (int k = 0; k < m; k++) {
            double over = accWeight[k] - b[k];
            if (over > 0) overSum += over;
        }
        return accProfit - penaltyFactor * overSum;
    }

    // ---------------- custos incrementais (O(m)) ----------------

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

    /** Δ inserir i. */
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

    /** Δ remover i. */
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

    /** Δ trocar out → in. */
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

    // ------------------------- Utilidades -------------------------

    /** Zera variables[]. */
    public void resetVariables() { Arrays.fill(variables, 0.0); }

    /** Zera acumuladores. */
    private void resetAccumulators() {
        Arrays.fill(accWeight, 0.0);
        accProfit = 0.0;
    }

    /** λ padrão: média dos lucros (boa escala inicial). */
    private double defaultPenaltyFactor() {
        double s = 0.0;
        for (double v : p) s += v;
        return (p.length > 0) ? s / p.length : 1.0;
    }

    /** Debug: imprime a instância carregada. */
    public void printInstance() {
        System.out.println("MKP OR-Library: n=" + size + ", m=" + m + ", idx=" + problemIndex
                + ", opt(file)=" + optimalFromFile);
        System.out.println("p: " + Arrays.toString(p));
        for (int k = 0; k < m; k++) System.out.println("w[" + k + "]: " + Arrays.toString(w[k]));
        System.out.println("b: " + Arrays.toString(b));
    }

    // --------------------- Leitura OR-Library ---------------------

    private static final class Loader {
        int n, m;
        Double[] p, b;
        Double[][] w;
        Double optVal;
    }

    /**
     * Lê um arquivo OR-Library (mknap1 / mknapcb*) e devolve a instância 'instanceIndex' (1..K).
     * Formato conforme documentação da OR-Library. :contentReference[oaicite:2]{index=2}
     */
    private Loader readOrLibInstance(String filename, int instanceIndex) throws IOException {
        if (instanceIndex <= 0) throw new IllegalArgumentException("instanceIndex deve ser >= 1");

        try (Reader r = new BufferedReader(new FileReader(filename))) {
            StreamTokenizer st = new StreamTokenizer(r);
            st.parseNumbers();

            // K: número de instâncias
            nextNum(st);
            int K = (int) st.nval;
            if (instanceIndex > K) {
                throw new IllegalArgumentException("instanceIndex=" + instanceIndex +
                        " > K=" + K + " no arquivo " + filename);
            }

            Loader out = new Loader();

            // Percorre instâncias até chegar na desejada
            for (int k = 1; k <= K; k++) {
                nextNum(st); int n = (int) st.nval;
                nextNum(st); int m = (int) st.nval;
                nextNum(st); double opt = st.nval;

                if (k != instanceIndex) {
                    // descarta n lucros
                    for (int i = 0; i < n; i++) nextNum(st);
                    // descarta m*n pesos
                    for (int c = 0; c < m; c++) for (int i = 0; i < n; i++) nextNum(st);
                    // descarta m capacidades
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

            // Se não retornou, algo deu errado
            throw new IOException("Falha ao localizar a instância " + instanceIndex + " em " + filename);
        }
    }

    /** Lê próximo token numérico (lança se EOF). */
    private static void nextNum(StreamTokenizer st) throws IOException {
        int tt = st.nextToken();
        if (tt == StreamTokenizer.TT_EOF)
            throw new IOException("Fim inesperado do arquivo OR-Library.");
        // caso venha um símbolo não-numérico, tente avançar
        while (tt != StreamTokenizer.TT_NUMBER) {
            tt = st.nextToken();
            if (tt == StreamTokenizer.TT_EOF)
                throw new IOException("Fim inesperado do arquivo OR-Library.");
        }
    }

    // --------------------------- Main ---------------------------

    public static void main(String[] args) throws Exception {
        // Uso:
        // java MKP_ORLib path_para/mknapcb4.txt 17 [lambda]
        String path = (args.length >= 1) ? args[0] : "instances/orlib/mknap1.txt";
        int idx = (args.length >= 2) ? Integer.parseInt(args[1]) : 1;

        MKP_ORLib mkp = (args.length >= 3)
                ? new MKP_ORLib(path, idx, Double.parseDouble(args[2]))
                : new MKP_ORLib(path, idx);

        System.out.println("Carregado: n=" + mkp.size + ", m=" + mkp.m +
                ", idx=" + mkp.problemIndex + ", opt(file)=" + mkp.optimalFromFile);
        // Pequeno teste: busca aleatória
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
        System.out.println("Melhor (random + penalização) = " + best);
    }
}
