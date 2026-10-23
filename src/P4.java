import java.io.BufferedReader;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Scanner;

public class P4 {

    private static final int[] TOPOLOGY = {400, 16, 8, 1};
    private static final String ENSEMBLE_DIR = "pesos";
    private static final String THRESHOLD_FILE = "threshold.txt";

    public static void main(String[] args) throws IOException {
        Scanner sc = new Scanner(System.in);
        sc.useDelimiter("[,;\\s]+");
        List<Double> values = readAllInput(sc);

        if (values.isEmpty()) throw new IOException("No feature values supplied.");
        if (values.size() % TOPOLOGY[0] != 0)
            throw new IOException("Input size is not a multiple of 400.");

        int m = values.size() / TOPOLOGY[0];
        Matrix X = new Matrix(m, TOPOLOGY[0]);
        for (int idx = 0; idx < values.size(); idx++)
            X.data[idx / TOPOLOGY[0]][idx % TOPOLOGY[0]] = values.get(idx);

        List<ModelParams> models = loadModels();
        if (models.isEmpty()) throw new IOException("No models found in weights directory.");

        double threshold = loadThreshold();
        double[] avgProb = new double[m];
        for (ModelParams model : models) {
            Matrix output = forward(X, model.weights, model.biases);
            for (int i = 0; i < m; i++) avgProb[i] += output.get(i, 0);
        }

        double inv = 1.0 / models.size();
        for (int i = 0; i < m; i++) {
            double p = avgProb[i] * inv;
            int digit = (p < threshold) ? 2 : 3;
            System.out.println(digit);
        }
    }

    private static Matrix forward(Matrix X, Matrix[] weights, Matrix[] biases) {
        Matrix current = X;
        for (int l = 0; l < weights.length; l++)
            current = current.dot(weights[l]).addRowVector(biases[l]).apply(P4::sigmoidStable);
        return current;
    }

    private static double sigmoidStable(double z) {
        if (z >= 0) {
            double e = Math.exp(-z);
            return 1.0 / (1.0 + e);
        } else {
            double e = Math.exp(z);
            return e / (1.0 + e);
        }
    }

    private static Matrix loadMatrix(Path path) throws IOException {
        List<double[]> rows = new ArrayList<>();
        try (BufferedReader reader = Files.newBufferedReader(path, StandardCharsets.UTF_8)) {
            String line;
            while ((line = reader.readLine()) != null) {
                if (line.trim().isEmpty()) continue;
                String[] parts = line.split(",");
                double[] row = new double[parts.length];
                for (int i = 0; i < parts.length; i++) row[i] = Double.parseDouble(parts[i]);
                rows.add(row);
            }
        }
        return new Matrix(rows.toArray(new double[0][]));
    }

    private static List<Double> readAllInput(Scanner sc) {
        List<Double> values = new ArrayList<>();
        Double value;
        while ((value = nextDoubleOrNull(sc)) != null) values.add(value);
        return values;
    }

    private static Double nextDoubleOrNull(Scanner sc) {
        while (sc.hasNext()) {
            String tok = sc.next().trim();
            if (tok.isEmpty()) continue;
            tok = tok.replace(',', '.');
            return Double.parseDouble(tok);
        }
        return null;
    }

    private static List<ModelParams> loadModels() throws IOException {
        Path base = resolveEnsembleDir();
        List<Path> ensembleDirs = new ArrayList<>();
        try (var stream = Files.list(base)) {
            stream.filter(p -> Files.isDirectory(p) && p.getFileName().toString().startsWith("model_"))
                  .forEach(ensembleDirs::add);
        }
        ensembleDirs.sort(Comparator.comparing(p -> p.getFileName().toString()));
        if (ensembleDirs.isEmpty()) ensembleDirs = Collections.singletonList(base);

        List<ModelParams> models = new ArrayList<>();
        for (Path dir : ensembleDirs) models.add(loadModelFromDir(dir));
        return models;
    }

    private static ModelParams loadModelFromDir(Path dir) throws IOException {
        Matrix[] weights = new Matrix[TOPOLOGY.length - 1];
        Matrix[] biases = new Matrix[TOPOLOGY.length - 1];
        for (int i = 0; i < weights.length; i++) {
            Path wPath = dir.resolve("w_layer" + (i + 1) + ".csv");
            Path bPath = dir.resolve("b_layer" + (i + 1) + ".csv");
            if (!Files.exists(wPath) || !Files.exists(bPath))
                throw new IOException("Missing weights in " + dir.toString());
            weights[i] = loadMatrix(wPath);
            biases[i] = loadMatrix(bPath);
        }
        return new ModelParams(weights, biases);
    }

    private static double loadThreshold() throws IOException {
        Path tPath = resolveEnsembleDir().resolve(THRESHOLD_FILE);
        if (!Files.exists(tPath)) return 0.5;
        List<String> lines = Files.readAllLines(tPath, StandardCharsets.UTF_8);
        if (lines.isEmpty()) return 0.5;
        try { return Double.parseDouble(lines.get(0).trim()); }
        catch (NumberFormatException ex) { return 0.5; }
    }

    private static Path resolveEnsembleDir() throws IOException {
        List<Path> candidates = new ArrayList<>();
        candidates.add(Paths.get(ENSEMBLE_DIR));
        candidates.add(Paths.get("src", ENSEMBLE_DIR));
        candidates.add(Paths.get("temp_submit", "src", ENSEMBLE_DIR));

        try {
            Path codePath = Paths.get(P4.class.getProtectionDomain().getCodeSource().getLocation().toURI()).getParent();
            if (codePath != null) candidates.add(codePath.resolve(ENSEMBLE_DIR));
        } catch (Exception ignored) {}

        for (Path candidate : candidates)
            if (Files.exists(candidate)) return candidate;

        throw new IOException("Weights directory not found. Checked: " + candidates);
    }

    private static final class ModelParams {
        final Matrix[] weights;
        final Matrix[] biases;
        ModelParams(Matrix[] weights, Matrix[] biases) {
            this.weights = weights;
            this.biases = biases;
        }
    }
}
