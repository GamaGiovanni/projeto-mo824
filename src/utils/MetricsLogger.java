package utils;

import java.io.*;
import java.nio.file.*;
import java.time.*;
import java.util.*;

public final class MetricsLogger implements AutoCloseable {
  private final PrintWriter runsOut;
  private final PrintWriter gensOut;
  private final boolean append;

  public static final class RunSummary {
    public final String datasetId, fileTag, algo, variant;
    public final int instanceIdx, generations;
    public final long seed, timeMs;
    public final double f, bk, ublp, gapLPpct, gapBKpct, deltaBK;
    public final double divHammingAvg, divEntropyAvg;
    public final String stopReason;
    public RunSummary(String datasetId, String fileTag, int instanceIdx,
                      String algo, String variant, long seed,
                      int generations, long timeMs,
                      double f, double bk, double ublp,
                      double gapLPpct, double gapBKpct, double deltaBK,
                      double divHammingAvg, double divEntropyAvg,
                      String stopReason) {
      this.datasetId = datasetId; this.fileTag = fileTag; this.instanceIdx = instanceIdx;
      this.algo = algo; this.variant = variant; this.seed = seed;
      this.generations = generations; this.timeMs = timeMs;
      this.f = f; this.bk = bk; this.ublp = ublp;
      this.gapLPpct = gapLPpct; this.gapBKpct = gapBKpct; this.deltaBK = deltaBK;
      this.divHammingAvg = divHammingAvg; this.divEntropyAvg = divEntropyAvg;
      this.stopReason = stopReason;
    }
  }

  public MetricsLogger(Path runsCsv, Path gensCsv, boolean append) throws IOException {
    this.append = append;
    boolean runsExists = Files.exists(runsCsv);
    boolean gensExists = Files.exists(gensCsv);
    this.runsOut = new PrintWriter(new BufferedWriter(new FileWriter(runsCsv.toFile(), append)));
    this.gensOut = new PrintWriter(new BufferedWriter(new FileWriter(gensCsv.toFile(), append)));
    if (!append || !runsExists) {
      runsOut.println(String.join(",",
        "timestamp","dataset_id","file_tag","instance_idx","algo","variant","seed",
        "generations","time_ms",
        "f","bk","ublp","gapLP_pct","gapBK_pct","deltaBK",
        "div_hamming_avg","div_entropy_avg","stop_reason"));
      runsOut.flush();
    }
    if (!append || !gensExists) {
      gensOut.println(String.join(",",
        "timestamp","dataset_id","file_tag","instance_idx","algo","variant","seed",
        "generation","elapsed_ms","best_f",
        "div_hamming","div_entropy"));
      gensOut.flush();
    }
  }

  public void logGeneration(String datasetId, String fileTag, int instanceIdx,
                            String algo, String variant, long seed,
                            int generation, long elapsedMs, double bestF,
                            double divHamming, double divEntropy) {
    gensOut.printf(Locale.US,
      "%s,%s,%s,%d,%s,%s,%d,%d,%d,%.6f,%.6f,%.6f%n",
      Instant.now(), datasetId, fileTag, instanceIdx, algo, variant, seed,
      generation, elapsedMs, bestF, divHamming, divEntropy);
    gensOut.flush();
  }

  public void logRunSummary(RunSummary s) {
    runsOut.printf(Locale.US,
      "%s,%s,%s,%d,%s,%s,%d,%d,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%s%n",
      Instant.now(), s.datasetId, s.fileTag, s.instanceIdx,
      s.algo, s.variant, s.seed, s.generations, s.timeMs,
      s.f, s.bk, s.ublp, s.gapLPpct, s.gapBKpct, s.deltaBK,
      s.divHammingAvg, s.divEntropyAvg, s.stopReason);
    runsOut.flush();
  }

  @Override public void close() { gensOut.close(); runsOut.close(); }
}
