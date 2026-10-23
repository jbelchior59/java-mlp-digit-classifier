import java.util.function.BiFunction;
import java.util.function.Function;
import java.util.Random;

public class Matrix {

    double[][] data;
    int rows, cols;

    public Matrix(int rows, int cols) {
        data = new double[rows][cols];
        this.rows = rows;
        this.cols = cols;
    }

    public Matrix(double[][] data) {
        if (data == null)
            throw new IllegalArgumentException("Matrix data cannot be null.");

        if (data.length == 0) {
            this.rows = 0;
            this.cols = 0;
            this.data = new double[0][0];
            return;
        }

        this.rows = data.length;
        this.cols = data[0].length;

        for (int i = 1; i < rows; i++) {
            if (data[i].length != this.cols) {
                throw new IllegalArgumentException("All rows must have the same number of columns.");
            }
        }

        this.data = new double[rows][cols];
        for (int i = 0; i < rows; i++)
            System.arraycopy(data[i], 0, this.data[i], 0, cols);

    }

    public double get(int row, int col) { return data[row][col]; }
    public int rows() { return rows; }
    public int cols() { return cols; }

    static public Matrix Rand(int rows, int cols, int seed) {
        Matrix out = new Matrix(rows, cols);

        if (seed < 0)
            seed = (int) System.currentTimeMillis();

        Random rand = new Random(seed);
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                out.data[i][j] = (rand.nextDouble() * 2.0) - 1.0;

        return out;
    }

    static public Matrix RandXavier(int fanIn, int fanOut, int seed) {
        Matrix out = new Matrix(fanIn, fanOut);

        if (seed < 0)
            seed = (int) System.currentTimeMillis();

        Random rand = new Random(seed);
        double limit = Math.sqrt(6.0 / (fanIn + fanOut));

        for (int i = 0; i < fanIn; i++)
            for (int j = 0; j < fanOut; j++)
                out.data[i][j] = (rand.nextDouble() * 2.0 - 1.0) * limit;

        return out;
    }

    private Matrix traverse(Function<Double, Double> fnc) {
        Matrix result = new Matrix(rows, cols);

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result.data[i][j] = fnc.apply(data[i][j]);
            }
        }
        return result;
    }

    public Matrix apply(Function<Double, Double> fnc) { return traverse(fnc); }

    public Matrix mult(double scalar) {
        return traverse(v -> v * scalar);
    }

    public Matrix add(double scalar) {
        return traverse(v -> v + scalar);
    }

    public Matrix subFromScalar(double scalar) {
        return traverse(v -> scalar - v);
    }

    public Matrix elementWise(Matrix other, BiFunction<Double, Double, Double> fnc) {
        if (this.rows != other.rows || this.cols != other.cols) {
            throw new IllegalArgumentException("Incompatible matrix sizes for element wise.");
        }

        Matrix result = new Matrix(rows, cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++)
                result.data[i][j] = fnc.apply(this.data[i][j], other.data[i][j]);
        }
        return result;
    }

    public Matrix add(Matrix other) {
        return this.elementWise(other, (a, b) -> a + b);
    }

    public Matrix mult(Matrix other) {
        return this.elementWise(other, (a, b) -> a * b);
    }

    public Matrix sub(Matrix other) {
        return this.elementWise(other, (a, b) -> a - b);
    }

    public double sum() {
        double total = 0.0;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                total += data[i][j];
            }
        }
        return total;
    }

    public Matrix sumColumns() {
        Matrix result = new Matrix(1, this.cols);

        for (int j = 0; j < this.cols; j++) {
            double sum = 0.0;
            for (int i = 0; i < this.rows; i++)
                sum += this.data[i][j];
            result.data[0][j] = sum;
        }
        return result;
    }

    public Matrix addRowVector(Matrix rowVector) {
        if (rowVector.rows() != 1 || rowVector.cols() != this.cols) {
            throw new IllegalArgumentException("Incompatible sizes for adding row vector.");
        }

        Matrix result = new Matrix(this.rows, this.cols);
        for (int i = 0; i < this.rows; i++) {
            for (int j = 0; j < this.cols; j++) {
                result.data[i][j] = this.data[i][j] + rowVector.data[0][j];
            }
        }
        return result;
    }

    public Matrix dot(Matrix other) {
        if (this.cols != other.rows) {
            throw new IllegalArgumentException("Incompatible matrix sizes for multiplication.");
        }

        Matrix result = new Matrix(this.rows, other.cols);
        for (int i = 0; i < this.rows; i++) {
            for (int j = 0; j < other.cols; j++) {
                for (int k = 0; k < this.cols; k++) {
                    result.data[i][j] += this.data[i][k] * other.data[k][j];
                }
            }
        }
        return result;
    }

    public Matrix transpose() {
        Matrix result = new Matrix(cols, rows);

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result.data[j][i] = data[i][j];
            }
        }
        return result;
    }

    public Matrix selectRows(int[] rowIndices) {
        Matrix result = new Matrix(rowIndices.length, this.cols);
        for (int i = 0; i < rowIndices.length; i++) {
            int r = rowIndices[i];
            System.arraycopy(this.data[r], 0, result.data[i], 0, this.cols);
        }
        return result;
    }
}
