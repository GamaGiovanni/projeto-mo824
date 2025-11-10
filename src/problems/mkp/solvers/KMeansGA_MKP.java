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
 * GA para MKP com escolha de pais por k-means.
 *
 * Ideia (resumo): em cada geração (ou a cada Gc gerações), clusterizamos a população
 * (k clusters, por padrão k=2) pelos genótipos 0/1 (distância euclidiana em {0,1}^n,
 * equivalente à Hamming). Depois, montamos pares escolhendo 1 pai de um cluster e 1
 * do outro (clusters distintos), com torneio por fitness dentro de cada cluster.
 *
 * Referências/metodologia:
 * - Proposta (seção “Pais por k-means: clusters distintos”): clusterização periódica,
 *   amostragem de bits (opcional) p/ custo linear em npop, e cruzamentos forçados
 *   entre clusters distintos, priorizando elites locais quando desejado. 
 *   (vide MO824_proposta_247122). 
 * - Tese (Laabadi, 2020): fixa k=2, divide a população, e, para cada indivíduo do
 *   grupo menor, o parceiro é escolhido no outro grupo via torneio — aumenta diversidade
 *   sem perder exploração (intensificação). 
 *   (vide Thèse2020.pdf). 
 *
 * CLI (mesmos args do GA_MKP; params opcionais no fim):
 *   java -cp out problems.mkp.solvers.KMeansGA_MKP <path_orlib> <instanceIndex> <popSize> <mutationRate> <useRepair:true/false> [lambda]
 *                                           [k] [tourn] [maxIter] [bitSample] [clusterEveryG]
 *
 * Defaults:
 *   k = 2                (número de clusters)
 *   tourn = 3            (tamanho do torneio em cada cluster)
 *   maxIter = 10         (iterações do k-means)
 *   bitSample = 0        (0 => usa todos os loci; >0 => amostra 'bitSample' loci p/ distância)
 *   clusterEveryG = 1    (re-executa k-means a cada G gerações; 1 = toda geração)
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

    /** Seleção de pais sobreposta: usa k-means em vez do torneio global. */
    @Override
    protected Population selectParents(Population population) {
        selectParentsCalls++;

        // Se não é “hora de clusterizar”, cai no baseline (torneio padrão).
        if ((selectParentsCalls % clusterEveryG) != 0) {
            return super.selectParents(population);
        }

        final int n = chromosomeSize;
        // Amostra de loci p/ reduzir custo (0 => usa todos)
        int[] loci = buildLociSample(n, bitSample);

        // --- K-means em {0,1}^n ---
        int[] assign = new int[popSize];
        double[][] centers = initCentersFromPopulation(population, k, n, loci);

        // Loop principal do k-means
        for (int it = 0; it < maxIter; it++) {
            boolean moved = assignToNearest(population, centers, assign, loci);

            // Recalcula centros; se algum cluster ficou vazio, repara.
            boolean emptyFixed = recomputeCenters(population, centers, assign, loci);

            if (!moved && !emptyFixed) break;
        }

        // Agrupa índices por cluster
        List<List<Integer>> byCluster = new ArrayList<>();
        for (int c = 0; c < k; c++) byCluster.add(new ArrayList<>());
        for (int i = 0; i < popSize; i++) byCluster.get(assign[i]).add(i);

        // Monta pais: pares entre clusters distintos, priorizando o menor cluster.
        Population parents = new Population();
        int targetParents = popSize; // mesmo contrato do framework

        // Estratégia: sempre escolher 1 pai do cluster menor e o 2º de outro cluster,
        // com torneio por fitness dentro de cada um.
        while (parents.size() < targetParents) {
            int cA = indexOfSmallestNonEmpty(byCluster);
            if (cA < 0) { // todos vazios? improvável, mas previne loop
                return super.selectParents(population);
            }
            int cB = pickDifferentNonEmptyCluster(byCluster, cA);
            if (cB < 0) { // sobrou cluster único não-vazio — fallback
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

            // Remoção opcional para “consumir” indivíduos do cluster menor primeiro:
            removeIndexOnce(byCluster.get(cA), population.indexOf(pA));
            removeIndexOnce(byCluster.get(cB), population.indexOf(pB));
        }

        return parents;
    }

    // ----------------- Helpers de K-means / torneio -----------------

    private int[] buildLociSample(int n, int sample) {
        if (sample <= 0 || sample >= n) {
            int[] all = new int[n];
            for (int i = 0; i < n; i++) all[i] = i;
            return all;
        }
        // sample aleatória sem reposição
        int[] idx = new int[n];
        for (int i = 0; i < n; i++) idx[i] = i;
        // Fisher-Yates parcial
        for (int i = 0; i < sample; i++) {
            int j = i + rng.nextInt(n - i);
            int tmp = idx[i]; idx[i] = idx[j]; idx[j] = tmp;
        }
        return Arrays.copyOf(idx, sample);
    }

    private double[][] initCentersFromPopulation(Population pop, int k, int n, int[] loci) {
        double[][] centers = new double[k][n];
        // sementes = k indivíduos aleatórios distintos
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
                // cluster vazio: re-semeia com um indivíduo aleatório
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

    private int indexOfSmallestNonEmpty(List<List<Integer>> byCluster) {
        int best = -1, bestSize = Integer.MAX_VALUE;
        for (int c = 0; c < byCluster.size(); c++) {
            int sz = byCluster.get(c).size();
            if (sz > 0 && sz < bestSize) { best = c; bestSize = sz; }
        }
        return best;
    }

    private int pickDifferentNonEmptyCluster(List<List<Integer>> byCluster, int avoid) {
        // tenta escolher o maior cluster != avoid, caso contrário o primeiro não vazio
        int best = -1, bestSize = -1;
        for (int c = 0; c < byCluster.size(); c++) {
            if (c == avoid) continue;
            int sz = byCluster.get(c).size();
            if (sz > bestSize) { best = c; bestSize = sz; }
        }
        return (bestSize > 0) ? best : -1;
    }

    private void removeIndexOnce(List<Integer> list, int idx) {
        for (int i = 0; i < list.size(); i++) {
            if (list.get(i) == idx) { list.remove(i); return; }
        }
    }

    /** Torneio restrito a um conjunto de índices (cluster). */
    private Chromosome tournamentWithin(Population pop, List<Integer> indices, int kTourn) {
        Chromosome best = null;
        double bestFit = Double.NEGATIVE_INFINITY;
        int n = indices.size();
        // Se o cluster tem poucos, degrade p/ torneio no tamanho possível
        int draws = Math.min(kTourn, Math.max(1, n));
        for (int r = 0; r < draws; r++) {
            int pos = indices.get(rng.nextInt(n));
            Chromosome cand = pop.get(pos);
            double fit = fitness(cand);
            if (fit > bestFit) { bestFit = fit; best = cand; }
        }
        // fallback (não deveria ocorrer)
        return (best != null) ? best : pop.get(indices.get(rng.nextInt(n)));
    }

   // ---------------- Runner com parâmetros nomeados ----------------
    public static void main(String[] args) throws Exception {
        // Exemplos:
        //  KMeans puro:
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

        // --- parâmetros base do GA (mesmos do GA_MKP) ---
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

        // --- parâmetros do k-means GA ---
        int k = getInt(cli, new String[]{"k"}, 2);
        int tourn = getInt(cli, new String[]{"tourn"}, 3);
        int maxIter = getInt(cli, new String[]{"maxIter"}, 10);
        int bitSample = getInt(cli, new String[]{"bitSample"}, 0);
        int clusterEveryG = getInt(cli, new String[]{"clusterEveryG"}, 1);

        KMeansGA_MKP ga = new KMeansGA_MKP(
                evaluator, popSize, mut, repair,
                k, tourn, maxIter, bitSample, clusterEveryG
        );

        // --- Tabu Search + Strategic Oscillation (opcional) ---
        boolean tsOn = getBool(cli, new String[]{"ts"}, false)
                || hasAny(cli, "tenure","ts-steps","steps","vmin","vmax","lmbMin","lmbMax","up","down");
        if (tsOn) {
            int tenure = getInt(cli, new String[]{"tenure"}, 7);
            int tsSteps = getInt(cli, new String[]{"ts-steps","steps"}, 500); // se quiser usar, crie um setter no GA
            double vmin = getDouble(cli, new String[]{"vmin"}, 0.0);
            double vmax = getDouble(cli, new String[]{"vmax"}, 100.0);
            double lmbMin = getDouble(cli, new String[]{"lmbMin"}, 0.1);
            double lmbMax = getDouble(cli, new String[]{"lmbMax"}, 10000.0);
            double up = getDouble(cli, new String[]{"up"}, 1.2);
            double down = getDouble(cli, new String[]{"down"}, 0.9);

            problems.mkp.TabuSO_MKP ts = new problems.mkp.TabuSO_MKP(
                    evaluator, tenure, lmbMin, lmbMax, up, down, vmin, vmax);
            ga.setImprover(ts);
            // opcional: ga.setLocalSearchSteps(tsSteps);
        }

        // === parâmetros de execução/registro (opcionais por CLI) ===
        String resultsDir = cli.getOrDefault("results-dir", "results");
        String algo = cli.getOrDefault("algo", KMeansGA_MKP.class.getSimpleName());
        String variant = cli.getOrDefault("variant", buildAutoVariant(mut, repair, tsOn, k, tourn, maxIter, bitSample, clusterEveryG, cli));
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

    private static String stripExt(String s) {
        int dot = s.lastIndexOf('.');
        return (dot >= 0) ? s.substring(0, dot) : s;
    }

    /** Monta um 'variant' legível se não vier por CLI — útil nos CSVs. */
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

    protected static void printHelp() {
        System.out.println("Uso (parâmetros nomeados em qualquer ordem):");
        System.out.println("  --path <arquivo OR-Library>          (default: instances/mkp/mknapcb1.txt)");
        System.out.println("  --instance|--inst <id>               (default: 1)");
        System.out.println("  --pop|--popSize <n>                  (default: 100)");
        System.out.println("  --mutation|--mut <p>                 (default: 0.02)");
        System.out.println("  --repair <true/false>                (default: true)");
        System.out.println("  --lambda|--lam|--lmb <val>           (opcional; se ausente, lambda automático)");
        System.out.println();
        System.out.println("  --k <n>                              (default: 2)");
        System.out.println("  --tourn <n>                          (default: 3)");
        System.out.println("  --maxIter <n>                        (default: 10)");
        System.out.println("  --bitSample <n>                      (default: 0  => usa todos os loci)");
        System.out.println("  --clusterEveryG <n>                  (default: 1  => clusteriza toda geração)");
        System.out.println();
        System.out.println("  --ts <true/false>                    (ativa TS+SO; também ativa se qualquer parâmetro de TS for passado)");
        System.out.println("  --tenure <n>                         (default: 7)");
        System.out.println("  --ts-steps|--steps <n>               (default: 500)");
        System.out.println("  --vmin <val>                         (default: 0.0)");
        System.out.println("  --vmax <val>                         (default: 100.0)");
        System.out.println("  --lmbMin <val>                       (default: 0.1)");
        System.out.println("  --lmbMax <val>                       (default: 10000.0)");
        System.out.println("  --up <fator>                         (default: 1.2)");
        System.out.println("  --down <fator>                       (default: 0.9)");
    }

}
