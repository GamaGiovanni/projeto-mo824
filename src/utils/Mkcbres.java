package utils;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

public final class Mkcbres {
  public static record Ref(double bk, double ublp) {}
  private final List<Ref> refs = new ArrayList<>();

  public Mkcbres(Path path) throws IOException {
    try (var br = Files.newBufferedReader(path)) {
      for (String ln; (ln = br.readLine()) != null; ) {
        ln = ln.trim();
        if (ln.isEmpty()) continue;
        String[] t = ln.split("\\s+");
        double bk   = Double.parseDouble(t[t.length-2]);
        double ublp = Double.parseDouble(t[t.length-1]);
        refs.add(new Ref(bk, ublp));
      }
    }
  }

  /** Instances numbered 1..30 within each mknapcbX; X ∈ {1..9} */
  public Ref get(String fileTag, int idx) {
    // fileTag = "mknapcb1"|"mknapcb2"|...; idx = 1..30
    int x = Integer.parseInt(fileTag.replaceAll("\\D+", ""));
    int global = (x - 1) * 30 + (idx - 1);
    return refs.get(global);
  }
    
}
