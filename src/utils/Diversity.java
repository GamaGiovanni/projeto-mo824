package utils;

import java.util.List;

public final class Diversity {

  /** Normalized Hamming distance [0,1] between two binary vectors. */
  public static double hamming01(boolean[] a, boolean[] b) {
    int d = 0;
    for (int i = 0; i < a.length; i++) if (a[i] != b[i]) d++;
    return d / (double) a.length;
  }

  /** Mean Hamming distance to medoid: simple O(n_pop^2); for large populations, sample pairs. */
  public static double meanHammingToMedoid(List<boolean[]> pop) {
    int n = pop.size();
    double bestSum = Double.POSITIVE_INFINITY;
    for (int i = 0; i < n; i++) {
      double sum = 0.0;
      for (int j = 0; j < n; j++) if (i != j) sum += hamming01(pop.get(i), pop.get(j));
      if (sum < bestSum) bestSum = sum;
    }
    return bestSum / (n - 1);
  }

  /** Mean entropy per locus (normalized by log(2)). */
  public static double meanLocusEntropy(List<boolean[]> pop) {
    int n = pop.size(), L = pop.get(0).length;
    double acc = 0.0;
    for (int i = 0; i < L; i++) {
      int ones = 0;
      for (boolean[] ind : pop) if (ind[i]) ones++;
      double p = ones / (double) n;
      if (p == 0.0 || p == 1.0) continue;
      double h = -(p * Math.log(p) + (1 - p) * Math.log(1 - p)) / Math.log(2);
      acc += h;
    }
    return acc / L;
  }
}
