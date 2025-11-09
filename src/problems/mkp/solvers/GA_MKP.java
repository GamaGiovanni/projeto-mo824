package problems.mkp.solvers;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;

import problems.Evaluator;
import problems.mkp.MKP_ORLib;
import solutions.Solution;
import utils.MetricsLogger;
import metaheuristics.ga.AbstractGA;
import metaheuristics.ls.LocalImprover;

public class GA_MKP extends AbstractGA<Integer, Integer> {

    /** Optional local improver (e.g., Tabu Search). */
    protected LocalImprover<Integer> improver = null;
    public void setImprover(LocalImprover<Integer> improver) { this.improver = improver; }


    /** Usa rotina de repair guloso (fenótipo), sem alterar o cromossomo. */
    private final boolean useGreedyRepair;

    public GA_MKP(Evaluator<Integer> objFunction,
                  Integer popSize,
                  Double mutationRate,
                  boolean useGreedyRepair) {
        super(objFunction, popSize, mutationRate);
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
        // Ex.: 
        //  Baseline:
        //  java problems.mkp.solvers.GA_MKP --path instances/mkp/mknapcb1.txt --instance 1 --pop 100 --mutation 0.02 --repair true
        //
        //  Com TS:
        //  java problems.mkp.solvers.GA_MKP --path instances/mkp/mknapcb1.txt --instance 1 --pop 100 --mutation 0.02 --repair true \
        //       --ts true --tenure 7 --ts-steps 500 --vmin 0.0 --vmax 100.0 --lmbMin 0.1 --lmbMax 10000 --up 1.2 --down 0.9

        if (hasFlag(args, "--help") || hasFlag(args, "-h")) {
            printHelp();
            return;
        }

        java.util.Map<String,String> cli = parseArgsToMap(args);

        // --- parâmetros do GA (com defaults) ---
        String path = cli.getOrDefault("path", "instances/mkp/mknapcb1.txt");

        int inst = getInt(cli, new String[]{"instance","inst"}, 1);
        int popSize = getInt(cli, new String[]{"pop","popSize"}, 100);
        double mut = getDouble(cli, new String[]{"mutation","mut"}, 0.02);
        boolean repair = getBool(cli, new String[]{"repair"}, true);

        // lambda opcional
        Double lambdaFixed = getNullableDouble(cli, new String[]{"lambda","lam","lmb"});
        MKP_ORLib evaluator = (lambdaFixed != null)
                ? new MKP_ORLib(path, inst, lambdaFixed)
                : new MKP_ORLib(path, inst);

        GA_MKP ga = new GA_MKP(evaluator, popSize, mut, repair);

        // --- TS + SO (opcional) ---
        boolean tsOn = getBool(cli, new String[]{"ts"}, false)
                || hasAny(cli, "tenure","ts-steps","vmin","vmax","lmbMin","lmbMax","up","down");

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
        String algo = cli.getOrDefault("algo", GA_MKP.class.getSimpleName());
        String variant = cli.getOrDefault("variant", buildAutoVariant(mut, repair, tsOn, cli));
        long seed = Long.parseLong(cli.getOrDefault("seed", "0"));
        String mkcbresPath = cli.get("mkcbres"); // pode vir por CLI; senão tentamos inferir abaixo

        // (opcional) setar seed se você ajustou o AbstractGA para permitir
        // AbstractGA.setSeed(seed);

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

    /* ====================== Helpers simples de CLI ====================== */

    protected static java.util.Map<String,String> parseArgsToMap(String[] args) {
        java.util.Map<String,String> map = new java.util.HashMap<>();
        for (int i = 0; i < args.length; i++) {
            String a = args[i];
            if (!a.startsWith("--")) continue;

            // --chave=valor
            int eq = a.indexOf('=');
            if (eq > 2) {
                String k = a.substring(2, eq).trim();
                String v = a.substring(eq + 1).trim();
                map.put(k, v);
                continue;
            }

            // --chave valor (se houver próximo token que não é outra flag)
            String k = a.substring(2).trim();
            String v = "true"; // valor padrão para flags booleanas "secas"
            if ((i + 1) < args.length && !args[i + 1].startsWith("--")) {
                v = args[++i].trim();
            }
            map.put(k, v);
        }
        return map;
    }

    protected static boolean hasFlag(String[] args, String flag) {
        for (String a : args) if (a.equals(flag)) return true;
        return false;
    }

    protected static boolean hasAny(java.util.Map<String,String> m, String... keys) {
        for (String k : keys) if (m.containsKey(k)) return true;
        return false;
    }

    protected static int getInt(java.util.Map<String,String> m, String[] keys, int def) {
        for (String k : keys) if (m.containsKey(k)) return Integer.parseInt(m.get(k));
        return def;
    }
    protected static double getDouble(java.util.Map<String,String> m, String[] keys, double def) {
        for (String k : keys) if (m.containsKey(k)) return Double.parseDouble(m.get(k));
        return def;
    }
    protected static Boolean getBool(java.util.Map<String,String> m, String[] keys, Boolean def) {
        for (String k : keys) if (m.containsKey(k)) return parseBool(m.get(k), def);
        return def;
    }
    protected static Double getNullableDouble(java.util.Map<String,String> m, String[] keys) {
        for (String k : keys) if (m.containsKey(k)) return Double.parseDouble(m.get(k));
        return null;
    }
    protected static Boolean parseBool(String s, Boolean def) {
        if (s == null) return def;
        switch (s.toLowerCase()) {
            case "1": case "true": case "yes": case "y": case "on": return true;
            case "0": case "false": case "no": case "n": case "off": return false;
            default: return def;
        }
    }

    protected static void printHelp() {
        System.out.println("Uso (parâmetros nomeados em qualquer ordem):");
        System.out.println("  --path <arquivo OR-Library>    (default: instances/mkp/mknapcb1.txt)");
        System.out.println("  --instance <id>                (default: 1)");
        System.out.println("  --pop|--popSize <n>            (default: 100)");
        System.out.println("  --mutation|--mut <p>           (default: 0.02)");
        System.out.println("  --repair <true/false>          (default: true)");
        System.out.println("  --lambda <val>                 (opcional; se não passar, usa lambda automático)");
        System.out.println();
        System.out.println("  --ts <true/false>              (ativa TS+SO; se omitir mas passar algum parâmetro de TS, ativa também)");
        System.out.println("  --tenure <n>                   (default: 7)");
        System.out.println("  --ts-steps|--steps <n>         (default: 500)");
        System.out.println("  --vmin <val>                   (default: 0.0)");
        System.out.println("  --vmax <val>                   (default: 100.0)");
        System.out.println("  --lmbMin <val>                 (default: 0.1)");
        System.out.println("  --lmbMax <val>                 (default: 10000.0)");
        System.out.println("  --up <fator>                   (default: 1.2)");
        System.out.println("  --down <fator>                 (default: 0.9)");
        System.out.println();
        System.out.println("Exemplos:");
        System.out.println("  java ...GA_MKP --path instances/mkp/mknapcb1.txt --instance 1 --pop 100 --mutation 0.02 --repair true");
        System.out.println("  java ...GA_MKP --path instances/mkp/mknapcb1.txt --instance 1 --pop 100 --mut 0.02 --repair true --ts true --tenure 7 --ts-steps 500 --vmin 0 --vmax 100 --lmbMin 0.1 --lmbMax 10000 --up 1.2 --down 0.9");
    }

    private static String stripExt(String s) {
        int dot = s.lastIndexOf('.');
        return (dot >= 0) ? s.substring(0, dot) : s;
    }

    /** Monta um 'variant' legível se não vier por CLI — útil nos CSVs. */
    private static String buildAutoVariant(double mut, boolean repair, boolean tsOn, java.util.Map<String,String> cli) {
        StringBuilder sb = new StringBuilder();
        sb.append("mut=").append(String.format(java.util.Locale.US, "%.4f", mut));
        sb.append(";repair=").append(repair);
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


    /** Overridable hook from AbstractGA: apply local improvement to selected individuals. */
    @Override
    protected void postGenerationHook(Population population, int generation) {
        if (improver == null) return;
        // improve 1 elite + 1 diverse (if any)
        Chromosome elite = getBestChromosome(population);
        applyImprovement(elite);

        Chromosome diverse = getMostDistantFrom(elite, population);
        if (diverse != null && diverse != elite) applyImprovement(diverse);
    }

    /** Applies the configured LocalImprover to a chromosome: decode, improve, encode-back. */
    protected void applyImprovement(Chromosome c) {
        if (c == null) return;
        Solution<Integer> s = decode(c);
        long deadline = 0L; // no strict deadline by default
        Solution<Integer> s2 = (improver != null) ? improver.improve(s, 500, deadline) : s;
        if (s2 != null && s2.cost > s.cost) {
            // encode back into genotype (0/1)
            for (int i = 0; i < chromosomeSize; i++) c.set(i, 0);
            for (int idx : s2) c.set(idx, 1);
        }
    }

    /** Finds the chromosome with maximum Hamming distance to 'ref'. */
    protected Chromosome getMostDistantFrom(Chromosome ref, Population pop) {
        if (ref == null || pop == null || pop.isEmpty()) return null;
        Chromosome best = null;
        int bestD = -1;
        for (Chromosome c : pop) {
            int d = 0;
            for (int i = 0; i < chromosomeSize; i++) {
                int a = (ref.get(i) != null && ref.get(i) != 0) ? 1 : 0;
                int b = (c.get(i)   != null && c.get(i)   != 0) ? 1 : 0;
                if (a != b) d++;
            }
            if (d > bestD) { bestD = d; best = c; }
        }
        return best;
    }
}
