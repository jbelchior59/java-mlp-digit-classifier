import java.io.BufferedReader;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Locale;
import java.util.Random;

public class DigitTrainer {

    private static final String DATASET_PATH = "data/dataset.csv";
    private static final String LABELS_PATH = "data/labels.csv";
    private static final String METRICS_PATH = "data/training_metrics.csv";
    private static final String CURVE_PATH = "data/training_curve.csv";
    private static final String ENSEMBLE_DIR = "pesos";
    private static final String THRESHOLD_FILE = "threshold.txt";

    private static final int[] TOPOLOGY = {400, 16, 8, 1};
    private static final int IMG_SIZE = 20;

    private static final double LEARNING_RATE = 0.01;
    private static final int MAX_EPOCHS = 500;
    private static final int EARLY_STOP_PATIENCE = 15;
    private static final double IMPROVEMENT_TOL = 1e-6;
    private static final int BATCH_SIZE = 32;

    private static final double FIXED_WEIGHT_DECAY = 1e-2;
    private static final double THRESH_MIN = 0.30;
    private static final double THRESH_MAX = 0.70;
    private static final double THRESH_STEP = 0.01;

    private static final int ENSEMBLE_SIZE = 10;
    private static final int HOLDOUT_REPEATS = 1;
    private static final int RANDOM_SEED = 42;

    public static void main(String[] args) throws IOException {
        double[][] features = loadDataset(Paths.get(DATASET_PATH));
        double[][] labels = loadLabels(Paths.get(LABELS_PATH));

        if (features.length != labels.length)
            throw new IllegalStateException("Mismatch between dataset and label counts.");

        List<Double> singleAccs = new ArrayList<>();
        List<Double> ensembleAccs = new ArrayList<>();
        List<Double> ensThresholds = new ArrayList<>();
        List<Double> singleThresholds = new ArrayList<>();

        for (int rep = 0; rep < HOLDOUT_REPEATS; rep++) {
            long seed = RANDOM_SEED + rep * 17L;

            StratifiedSplit holdout = stratifiedSplit(features, labels, 0.85, seed);
            StratifiedSplit inner = stratifiedSplit(holdout.trainFeatures, holdout.trainLabels, 0.85, seed + 1);
            System.out.println("epochs = " + MAX_EPOCHS + ", batch size = " + BATCH_SIZE);

            // ---- Single model ----
            boolean logCurve = (rep == 0);
            TrainedModel single = trainModel(inner.trainFeatures, inner.trainLabels,
                    inner.valFeatures, inner.valLabels, FIXED_WEIGHT_DECAY, seed, logCurve);

            if (logCurve && single.metricsHistory != null) {
                saveCurveMetrics(single.metricsHistory, Paths.get(CURVE_PATH));
            }

            double singleThresh = pickBestThreshold(single.valProbabilities, single.valLabels);
            double singleTestAcc = accuracy(flattenVector(single.mlp.predict(new Matrix(holdout.valFeatures))),
                    holdout.valLabels, singleThresh);

            // ---- Ensemble ----
            List<TrainedModel> ensemble = trainEnsemble(inner.trainFeatures, inner.trainLabels,
                    inner.valFeatures, inner.valLabels, FIXED_WEIGHT_DECAY, seed);

            double[] ensValMean = meanEnsembleValProbs(ensemble);
            double ensThreshold = pickBestThreshold(ensValMean, inner.valLabels);

            double[] ensTestMean = meanEnsembleProbsOnFeatures(ensemble, holdout.valFeatures);
            double ensembleTestAcc = accuracy(ensTestMean, holdout.valLabels, ensThreshold);

            singleAccs.add(singleTestAcc);
            ensembleAccs.add(ensembleTestAcc);
            ensThresholds.add(ensThreshold);
            singleThresholds.add(singleThresh);

            System.out.printf(Locale.US,
                    "[rep=%d] single=%.4f thr=%.2f | ensemble=%.4f thr=%.2f%n",
                    rep, singleTestAcc, singleThresh, ensembleTestAcc, ensThreshold);
        }

        double singleMean = meanD(singleAccs);
        double singleStd = stdD(singleAccs, singleMean);
        double ensMean = meanD(ensembleAccs);
        double ensStd = stdD(ensembleAccs, ensMean);

        double ensThrFinal = medianD(ensThresholds);
        double singleThrFinal = medianD(singleThresholds);

        saveThreshold(ensThrFinal, Paths.get(ENSEMBLE_DIR, THRESHOLD_FILE));
        saveTrainingMetrics(FIXED_WEIGHT_DECAY, singleThrFinal, ensThrFinal,
                singleMean, ensMean, singleStd, ensStd, Paths.get(METRICS_PATH));

        System.out.printf(Locale.US,
                "Mean acc over %d reps -> single=%.4f (std=%.4f) | ensemble=%.4f (std=%.4f) | thr(median)=%.2f%n",
                HOLDOUT_REPEATS, singleMean, singleStd, ensMean, ensStd, ensThrFinal);

        // Se queres sempre gravar os pesos finais, descomenta a linha:
        saveFinalEnsemble(features, labels, FIXED_WEIGHT_DECAY, ensThrFinal);
    }

    // -------------------- FINAL ensemble saved to "pesos/model_XX" --------------------

    private static void saveFinalEnsemble(double[][] features, double[][] labels,
                                          double weightDecay, double threshold) throws IOException {

        // Split só para early stopping (treina quase tudo)
        StratifiedSplit split = stratifiedSplit(features, labels, 0.95, RANDOM_SEED + 999);

        for (int i = 0; i < ENSEMBLE_SIZE; i++) {
            long seed = RANDOM_SEED + 1000L + i * 31L;

            AugmentedData aug = augmentDataset(split.trainFeatures, split.trainLabels, seed);

                TrainedModel tm = trainModel(aug.features, aug.labels,
                    split.valFeatures, split.valLabels, weightDecay, seed, false);

            Path modelDir = Paths.get(ENSEMBLE_DIR, String.format(Locale.US, "model_%02d", i));
            saveParameters(tm.weights, tm.biases, modelDir);
        }

        saveThreshold(threshold, Paths.get(ENSEMBLE_DIR, THRESHOLD_FILE));
        System.out.printf(Locale.US, "Saved FINAL ensemble (%d models) to '%s' with threshold %.2f%n",
                ENSEMBLE_SIZE, ENSEMBLE_DIR, threshold);
    }

    // -------------------- Ensemble training --------------------

    private static List<TrainedModel> trainEnsemble(double[][] trainF, double[][] trainL,
                                                    double[][] valF, double[][] valL,
                                                    double weightDecay, long baseSeed) throws IOException {
        List<TrainedModel> ensemble = new ArrayList<>();

        for (int i = 0; i < ENSEMBLE_SIZE; i++) {
            long seed = baseSeed + i * 31L;

            AugmentedData aug = augmentDataset(trainF, trainL, seed);
            TrainedModel model = trainModel(aug.features, aug.labels, valF, valL, weightDecay, seed, false);
            ensemble.add(model);
        }

        return ensemble;
    }

    // -------------------- Core training (early stopping + MSE) --------------------

    private static TrainedModel trainModel(double[][] trainF, double[][] trainL,
                                           double[][] valF, double[][] valL,
                                           double weightDecay, long seed, boolean keepCurve) {
        Matrix trainX = new Matrix(trainF);
        Matrix trainY = new Matrix(trainL);
        Matrix valX = new Matrix(valF);
        Matrix valY = new Matrix(valL);

        MLP mlp = new MLP(TOPOLOGY, buildActivations(TOPOLOGY.length - 1), (int) seed);
        mlp.setWeightDecay(weightDecay);

        Matrix[] bestW = null;
        Matrix[] bestB = null;
        double bestValMse = Double.POSITIVE_INFINITY;
        int bestEpoch = 0;

                List<EpochMetric> history = keepCurve ? new ArrayList<>() : null;

        int patienceCounter = 0;

        for (int epoch = 1; epoch <= MAX_EPOCHS; epoch++) {
            double trainMse = trainOneEpochMiniBatch(mlp, trainX, trainY, LEARNING_RATE, BATCH_SIZE, seed + epoch);

            double valMse = meanSquaredError(mlp.predict(valX), valY);

            if (history != null) {
                history.add(new EpochMetric(epoch, trainMse, valMse));
            }

            if (valMse < bestValMse - IMPROVEMENT_TOL) {
                bestValMse = valMse;
                bestW = mlp.getWeights();
                bestB = mlp.getBiases();
                bestEpoch = epoch;
                patienceCounter = 0;
            } else {
                patienceCounter++;
            }

            if (patienceCounter >= EARLY_STOP_PATIENCE)
                break;
        }

        // fallback: se nunca melhorou, usa o estado final
        if (bestW == null || bestB == null) {
            bestW = mlp.getWeights();
            bestB = mlp.getBiases();
            bestValMse = meanSquaredError(mlp.predict(valX), valY);
            bestEpoch = MAX_EPOCHS;
        }

        MLP evalMlp = new MLP(TOPOLOGY, buildActivations(TOPOLOGY.length - 1), (int) seed);
        evalMlp.setWeightDecay(weightDecay);
        evalMlp.setParameters(bestW, bestB);

        double[] valProb = flattenVector(evalMlp.predict(valX));

        // devolvemos evalMlp (já está nos best params)
        return new TrainedModel(evalMlp, bestW, bestB, bestValMse, bestEpoch, valProb, valL, history);
    }


    
    private static double trainOneEpochMiniBatch(MLP mlp, Matrix X, Matrix y, double lr, int batchSize, long seed) {
        int n = X.rows();
        int[] indices = shuffledIndices(n, seed);
        double sumLoss = 0.0;
        int batches = 0;
        for (int start = 0; start < n; start += batchSize) {
            int end = Math.min(n, start + batchSize);
            int[] batchIdx = Arrays.copyOfRange(indices, start, end);
            Matrix bx = X.selectRows(batchIdx);
            Matrix by = y.selectRows(batchIdx);
            sumLoss += mlp.trainEpoch(bx, by, lr); // returns MSE
            batches++;
        }
        return sumLoss / batches;
    }

    // -------------------- MSE --------------------

    private static double meanSquaredError(Matrix predicted, Matrix expected) {
        Matrix diff = expected.sub(predicted);
        Matrix squared = diff.mult(diff);
        return squared.sum() / (predicted.rows() * predicted.cols());
    }

    // -------------------- Threshold / Accuracy --------------------

    private static double accuracy(double[] probs, double[][] labels, double threshold) {
        int correct = 0;
        for (int i = 0; i < probs.length; i++) {
            int pred = probs[i] >= threshold ? 1 : 0;
            int truth = labels[i][0] >= 0.5 ? 1 : 0;
            if (pred == truth) correct++;
        }
        return (double) correct / probs.length;
    }

    private static double pickBestThreshold(double[] probs, double[][] labels) {
        double bestAcc = -1.0;
        double bestT = 0.5;
        for (double t = THRESH_MIN; t <= THRESH_MAX + 1e-9; t += THRESH_STEP) {
            double acc = accuracy(probs, labels, t);
            if (acc > bestAcc + 1e-6) {
                bestAcc = acc;
                bestT = t;
            }
        }
        return bestT;
    }

    // -------------------- Mean ensemble probabilities --------------------

    private static double[] meanEnsembleValProbs(List<TrainedModel> ensemble) {
        if (ensemble.isEmpty()) return new double[0];
        int n = ensemble.get(0).valProbabilities.length;
        double[] sum = new double[n];
        for (TrainedModel m : ensemble) {
            for (int i = 0; i < n; i++) sum[i] += m.valProbabilities[i];
        }
        for (int i = 0; i < n; i++) sum[i] /= ensemble.size();
        return sum;
    }

    private static double[] meanEnsembleProbsOnFeatures(List<TrainedModel> ensemble, double[][] features) {
        if (ensemble.isEmpty()) return new double[0];
        int n = features.length;
        double[] sum = new double[n];
        Matrix X = new Matrix(features);
        for (TrainedModel tm : ensemble) {
            double[] p = flattenVector(tm.mlp.predict(X));
            for (int i = 0; i < n; i++) sum[i] += p[i];
        }
        for (int i = 0; i < n; i++) sum[i] /= ensemble.size();
        return sum;
    }

    // -------------------- Augmentation (mantive o teu, mas podes trocar depois) --------------------

    private static AugmentedData augmentDataset(double[][] features, double[][] labels, long seed) {
        Random rnd = new Random(seed);
        List<double[]> augF = new ArrayList<>();
        List<double[]> augL = new ArrayList<>();

        for (int i = 0; i < features.length; i++) {
            double[] base = features[i];
            double label = labels[i][0];

            augF.add(base);
            augL.add(new double[]{label});

            int shiftMag = rnd.nextDouble() < 0.65 ? 1 : 2;
            augF.add(applyShift(base, shiftMag, 0));
            augL.add(new double[]{label});

            augF.add(applyShift(base, -shiftMag, 0));
            augL.add(new double[]{label});

            augF.add(applyShift(base, 0, shiftMag));
            augL.add(new double[]{label});

            augF.add(applyShift(base, 0, -shiftMag));
            augL.add(new double[]{label});

            double[] noisy = applyNoise(base, 0.01 + rnd.nextDouble() * 0.01, rnd);
            augF.add(noisy);
            augL.add(new double[]{label});

            if (rnd.nextDouble() < 0.5) {
                double angle = (rnd.nextDouble() * 10.0 - 5.0); // [-5,5]
                augF.add(applyRotation(base, angle));
                augL.add(new double[]{label});
            }

            // BUGFIX opcional (se quiseres manter): usar 1 random só
            double r = rnd.nextDouble();
            if (r < 0.25) {
                augF.add(applyDilation(base));
                augL.add(new double[]{label});
            } else if (r < 0.50) {
                augF.add(applyErosion(base));
                augL.add(new double[]{label});
            }
        }

        return new AugmentedData(augF.toArray(new double[0][]), augL.toArray(new double[0][]));
    }

    private static double[] applyNoise(double[] input, double sigma, Random rnd) {
        double[] out = new double[input.length];
        for (int i = 0; i < input.length; i++) {
            double v = input[i] + rnd.nextGaussian() * sigma;
            out[i] = clamp01(v);
        }
        return out;
    }

    private static double[] applyShift(double[] input, int dx, int dy) {
        double[][] grid = toGrid(input);
        double[][] shifted = new double[IMG_SIZE][IMG_SIZE];
        for (int y = 0; y < IMG_SIZE; y++) {
            for (int x = 0; x < IMG_SIZE; x++) {
                int srcX = x - dx;
                int srcY = y - dy;
                if (srcX >= 0 && srcX < IMG_SIZE && srcY >= 0 && srcY < IMG_SIZE)
                    shifted[y][x] = grid[srcY][srcX];
            }
        }
        return flatten(shifted);
    }

    private static double[] applyRotation(double[] input, double angleDeg) {
        double[][] grid = toGrid(input);
        double[][] rotated = new double[IMG_SIZE][IMG_SIZE];
        double angleRad = Math.toRadians(angleDeg);
        double cos = Math.cos(angleRad);
        double sin = Math.sin(angleRad);
        double cx = (IMG_SIZE - 1) / 2.0;
        double cy = (IMG_SIZE - 1) / 2.0;

        for (int y = 0; y < IMG_SIZE; y++) {
            for (int x = 0; x < IMG_SIZE; x++) {
                double dx = x - cx;
                double dy = y - cy;
                double srcX = cos * dx + sin * dy + cx;
                double srcY = -sin * dx + cos * dy + cy;
                int sx = (int) Math.round(srcX);
                int sy = (int) Math.round(srcY);
                if (sx >= 0 && sx < IMG_SIZE && sy >= 0 && sy < IMG_SIZE)
                    rotated[y][x] = grid[sy][sx];
            }
        }
        return flatten(rotated);
    }

    private static double[] applyDilation(double[] input) {
        double[][] grid = toGrid(input);
        double[][] out = new double[IMG_SIZE][IMG_SIZE];
        for (int y = 0; y < IMG_SIZE; y++) {
            for (int x = 0; x < IMG_SIZE; x++) {
                double max = 0.0;
                for (int dy = -1; dy <= 1; dy++) {
                    for (int dx = -1; dx <= 1; dx++) {
                        int sx = x + dx;
                        int sy = y + dy;
                        if (sx >= 0 && sx < IMG_SIZE && sy >= 0 && sy < IMG_SIZE)
                            max = Math.max(max, grid[sy][sx]);
                    }
                }
                out[y][x] = max;
            }
        }
        return flatten(out);
    }

    private static double[] applyErosion(double[] input) {
        double[][] grid = toGrid(input);
        double[][] out = new double[IMG_SIZE][IMG_SIZE];
        for (int y = 0; y < IMG_SIZE; y++) {
            for (int x = 0; x < IMG_SIZE; x++) {
                double min = 1.0;
                for (int dy = -1; dy <= 1; dy++) {
                    for (int dx = -1; dx <= 1; dx++) {
                        int sx = x + dx;
                        int sy = y + dy;
                        if (sx >= 0 && sx < IMG_SIZE && sy >= 0 && sy < IMG_SIZE)
                            min = Math.min(min, grid[sy][sx]);
                    }
                }
                out[y][x] = min;
            }
        }
        return flatten(out);
    }

    // -------------------- Load dataset --------------------

    private static double[][] loadDataset(Path path) throws IOException {
        List<double[]> rows = new ArrayList<>();
        try (BufferedReader reader = Files.newBufferedReader(path, StandardCharsets.UTF_8)) {
            String line;
            while ((line = reader.readLine()) != null) {
                if (line.trim().isEmpty()) continue;
                String[] parts = line.split(",");
                if (parts.length != TOPOLOGY[0])
                    throw new IllegalStateException("Each sample must have 400 values.");
                double[] row = new double[parts.length];
                for (int i = 0; i < parts.length; i++) {
                    double raw = Double.parseDouble(parts[i]);
                    row[i] = normalizePixel(raw);
                }
                rows.add(row);
            }
        }
        return rows.toArray(new double[0][]);
    }

    private static double[][] loadLabels(Path path) throws IOException {
        List<double[]> rows = new ArrayList<>();
        try (BufferedReader reader = Files.newBufferedReader(path, StandardCharsets.UTF_8)) {
            String line;
            while ((line = reader.readLine()) != null) {
                String trimmed = line.trim();
                if (trimmed.isEmpty()) continue;
                int digit = Integer.parseInt(trimmed);
                if (digit != 2 && digit != 3)
                    throw new IllegalStateException("Labels must be 2 or 3 only.");
                rows.add(new double[]{digit == 2 ? 0.0 : 1.0});
            }
        }
        return rows.toArray(new double[0][]);
    }

    // -------------------- Split --------------------

    private static StratifiedSplit stratifiedSplit(double[][] features, double[][] labels, double trainRatio, long seed) {
        List<Integer> class0 = new ArrayList<>();
        List<Integer> class1 = new ArrayList<>();
        for (int i = 0; i < labels.length; i++) {
            if (labels[i][0] == 0.0) class0.add(i);
            else class1.add(i);
        }
        Collections.shuffle(class0, new Random(seed));
        Collections.shuffle(class1, new Random(seed + 1));

        int train0 = (int) Math.round(class0.size() * trainRatio);
        int train1 = (int) Math.round(class1.size() * trainRatio);

        List<Integer> trainIdx = new ArrayList<>();
        List<Integer> valIdx = new ArrayList<>();

        trainIdx.addAll(class0.subList(0, Math.min(train0, class0.size())));
        valIdx.addAll(class0.subList(Math.min(train0, class0.size()), class0.size()));
        trainIdx.addAll(class1.subList(0, Math.min(train1, class1.size())));
        valIdx.addAll(class1.subList(Math.min(train1, class1.size()), class1.size()));

        int[] tIdx = trainIdx.stream().mapToInt(Integer::intValue).toArray();
        int[] vIdx = valIdx.stream().mapToInt(Integer::intValue).toArray();

        double[][] trainF = subset(features, tIdx);
        double[][] trainL = subset(labels, tIdx);
        double[][] valF = subset(features, vIdx);
        double[][] valL = subset(labels, vIdx);

        return new StratifiedSplit(trainF, trainL, valF, valL);
    }

    private static double[][] subset(double[][] data, int[] idx) {
        double[][] out = new double[idx.length][data[0].length];
        for (int i = 0; i < idx.length; i++)
            out[i] = Arrays.copyOf(data[idx[i]], data[idx[i]].length);
        return out;
    }

    // -------------------- Utils --------------------

    private static int[] shuffledIndices(int n, long seed) {
        int[] idx = new int[n];
        for (int i = 0; i < n; i++) idx[i] = i;
        Random rnd = new Random(seed);
        for (int i = n - 1; i > 0; i--) {
            int j = rnd.nextInt(i + 1);
            int tmp = idx[i];
            idx[i] = idx[j];
            idx[j] = tmp;
        }
        return idx;
    }

    private static IDifferentiableFunction[] buildActivations(int numLayers) {
        IDifferentiableFunction[] activations = new IDifferentiableFunction[numLayers];
        for (int i = 0; i < numLayers; i++)
            activations[i] = new Sigmoid();
        return activations;
    }

    private static double normalizePixel(double value) { return value; }

    private static double clamp01(double v) {
        if (v < 0.0) return 0.0;
        if (v > 1.0) return 1.0;
        return v;
    }

    private static double[][] toGrid(double[] flat) {
        if (flat.length != IMG_SIZE * IMG_SIZE)
            throw new IllegalArgumentException("Expected flattened 20x20 vector.");
        double[][] grid = new double[IMG_SIZE][IMG_SIZE];
        for (int i = 0; i < flat.length; i++)
            grid[i / IMG_SIZE][i % IMG_SIZE] = flat[i];
        return grid;
    }

    private static double[] flatten(double[][] grid) {
        double[] out = new double[grid.length * grid[0].length];
        int idx = 0;
        for (double[] row : grid)
            for (double v : row)
                out[idx++] = clamp01(v);
        return out;
    }

    private static double[] flattenVector(Matrix m) {
        double[] out = new double[m.rows()];
        for (int i = 0; i < m.rows(); i++)
            out[i] = m.get(i, 0);
        return out;
    }

    private static double meanD(List<Double> values) {
        if (values.isEmpty()) return 0.0;
        double sum = 0.0;
        for (double v : values) sum += v;
        return sum / values.size();
    }

    private static double stdD(List<Double> values, double mean) {
        if (values.size() <= 1) return 0.0;
        double ss = 0.0;
        for (double v : values) {
            double d = v - mean;
            ss += d * d;
        }
        return Math.sqrt(ss / values.size());
    }

    private static double medianD(List<Double> values) {
        if (values.isEmpty()) return 0.5;
        List<Double> copy = new ArrayList<>(values);
        copy.sort(Comparator.naturalOrder());
        int n = copy.size();
        if (n % 2 == 1) return copy.get(n / 2);
        return 0.5 * (copy.get(n / 2 - 1) + copy.get(n / 2));
    }

    // -------------------- Saving --------------------

    private static void saveParameters(Matrix[] weights, Matrix[] biases, Path targetDir) throws IOException {
        Files.createDirectories(targetDir);
        for (int i = 0; i < weights.length; i++)
            writeMatrix(weights[i], targetDir.resolve(String.format(Locale.US, "w_layer%d.csv", i + 1)));
        for (int i = 0; i < biases.length; i++)
            writeMatrix(biases[i], targetDir.resolve(String.format(Locale.US, "b_layer%d.csv", i + 1)));
    }

    private static void saveThreshold(double threshold, Path path) throws IOException {
        Files.createDirectories(path.getParent());
        Files.write(path, Collections.singletonList(String.format(Locale.US, "%.4f", threshold)),
                StandardCharsets.UTF_8);
    }

    private static void saveCurveMetrics(List<EpochMetric> metrics, Path outputPath) throws IOException {
        List<String> lines = new ArrayList<>();
        lines.add("epoch,train_mse,val_mse");
        for (EpochMetric m : metrics) {
            lines.add(String.format(Locale.US, "%d,%.10f,%.10f", m.epoch, m.trainMse, m.valMse));
        }

        Path parent = outputPath.getParent();
        if (parent != null) Files.createDirectories(parent);
        Files.write(outputPath, lines, StandardCharsets.UTF_8);
    }

    private static void saveTrainingMetrics(double weightDecay, double singleThresh, double ensThresh,
                                            double singleMeanAcc, double ensembleMeanAcc,
                                            double singleStdAcc, double ensembleStdAcc,
                                            Path outputPath) throws IOException {
        List<String> lines = new ArrayList<>();
        lines.add("weight_decay,single_threshold,ensemble_threshold,single_mean_acc,ensemble_mean_acc,single_std_acc,ensemble_std_acc,repeats");
        lines.add(String.format(Locale.US, "%.5f,%.4f,%.4f,%.6f,%.6f,%.6f,%.6f,%d",
                weightDecay, singleThresh, ensThresh, singleMeanAcc, ensembleMeanAcc,
                singleStdAcc, ensembleStdAcc, HOLDOUT_REPEATS));

        Path parent = outputPath.getParent();
        if (parent != null) Files.createDirectories(parent);
        Files.write(outputPath, lines, StandardCharsets.UTF_8);
    }

    private static void writeMatrix(Matrix matrix, Path path) throws IOException {
        List<String> rows = new ArrayList<>();
        for (int r = 0; r < matrix.rows(); r++) {
            StringBuilder builder = new StringBuilder();
            for (int c = 0; c < matrix.cols(); c++) {
                if (c > 0) builder.append(',');
                builder.append(String.format(Locale.US, "%.10f", matrix.get(r, c)));
            }
            rows.add(builder.toString());
        }
        Files.write(path, rows, StandardCharsets.UTF_8);
    }

    // -------------------- Data structs --------------------

    private static final class AugmentedData {
        final double[][] features;
        final double[][] labels;
        AugmentedData(double[][] features, double[][] labels) {
            this.features = features;
            this.labels = labels;
        }
    }

    private static final class TrainedModel {
        final MLP mlp;
        final Matrix[] weights;
        final Matrix[] biases;
        final double valMse;
        final int bestEpoch;
        final double[] valProbabilities;
        final double[][] valLabels;
        final List<EpochMetric> metricsHistory;

        TrainedModel(MLP mlp, Matrix[] weights, Matrix[] biases, double valMse, int bestEpoch,
                     double[] valProbabilities, double[][] valLabels, List<EpochMetric> metricsHistory) {
            this.mlp = mlp;
            this.weights = weights;
            this.biases = biases;
            this.valMse = valMse;
            this.bestEpoch = bestEpoch;
            this.valProbabilities = valProbabilities;
            this.valLabels = valLabels;
            this.metricsHistory = metricsHistory;
        }
    }

    private static final class EpochMetric {
        final int epoch;
        final double trainMse;
        final double valMse;

        EpochMetric(int epoch, double trainMse, double valMse) {
            this.epoch = epoch;
            this.trainMse = trainMse;
            this.valMse = valMse;
        }
    }

    private static final class StratifiedSplit {
        final double[][] trainFeatures;
        final double[][] trainLabels;
        final double[][] valFeatures;
        final double[][] valLabels;

        StratifiedSplit(double[][] trainFeatures, double[][] trainLabels,
                        double[][] valFeatures, double[][] valLabels) {
            this.trainFeatures = trainFeatures;
            this.trainLabels = trainLabels;
            this.valFeatures = valFeatures;
            this.valLabels = valLabels;
        }
    }
}
