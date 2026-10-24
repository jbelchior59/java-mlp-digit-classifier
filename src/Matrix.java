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

    //accessors
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
                out.data[i][j] = (rand.nextDouble() * 2.0) - 1.0; // intervalo [-1, 1]

        return out;
    }

    // Xavier/Glorot initialization - melhor para Sigmoid/Tanh
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


    //Apply Function<Double, Double> to all elements of the matrix
    //store the result in matrix result
    private Matrix traverse(Function<Double, Double> fnc) {
        Matrix result = new Matrix(rows, cols);

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result.data[i][j] = fnc.apply(data[i][j]);
            }
        }
        return result;
    }

    //Apply a function to all elements
    public Matrix apply(Function<Double, Double> fnc) { return traverse(fnc); }

    //multiply matrix by scalar
    public Matrix mult(double scalar) {

        // insert code here to multiply all elements by scalar
        return traverse(v -> v * scalar);
    }

    //add scalar to matrix
    public Matrix add(double scalar) {
        // insert code here to add to all elements the scalar
        return traverse(v -> v + scalar);
    }

    //sub matrix from scalar:   scalar - M
    public Matrix subFromScalar(double scalar) {

        // insert code here to subtract from the scalar all the elements  
        return traverse(v -> scalar - v);
    }

    //Element wise operation
    public Matrix elementWise(Matrix other, BiFunction<Double, Double, Double> fnc) {
        if (this.rows != other.rows || this.cols != other.cols) {
            throw new IllegalArgumentException("Incompatible matrix sizes for element wise.");
        }

        Matrix result = new Matrix(rows, cols);

        //add element by element
        //store the result in matrix result
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++)
                result.data[i][j] = fnc.apply(this.data[i][j], other.data[i][j]);
        }
        return result;
    }

    //add two matrices 
    public Matrix add(Matrix other) {
        return this.elementWise(other, (a, b) -> a + b);
    }

    //multiply two matrices (element-wise)
    public Matrix mult(Matrix other) {

        // insert code here to multiply element-wise     
        return this.elementWise(other, (a, b) -> a * b);
    }

    //subtract two matrices (element-wise)
    public Matrix sub(Matrix other) {

        // insert code here to subtract (this-other) element-wise     
        return this.elementWise(other, (a, b) -> a - b);
    }

    //sum all elements of the matrix
    public double sum() {

        // insert code here to return the sum of all elements 
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

    //Add row vector to each row of the matrix
    public Matrix addRowVector(Matrix rowVector) {
        if (rowVector.rows() != 1 || rowVector.cols() != this.cols) {

            throw new IllegalArgumentException("Incompatible sizes for adding row vector.");
        }

        Matrix result = new Matrix(this.rows, this.cols);

        //add row vector to each row
        for (int i = 0; i < this.rows; i++) {
            for (int j = 0; j < this.cols; j++) {
                result.data[i][j] = this.data[i][j] + rowVector.data[0][j];
            }
        }
        return result;
    }

    //multiply two matrices (dot product)
    public Matrix dot(Matrix other) {
        if (this.cols != other.rows) {
            throw new IllegalArgumentException("Incompatible matrix sizes for multiplication.");
        }

        Matrix result = new Matrix(this.rows, other.cols);

        //multiply 2 matrices
        //store the result in matrix result
        for (int i = 0; i < this.rows; i++) {
            for (int j = 0; j < other.cols; j++) {
                for (int k = 0; k < this.cols; k++) {
                    result.data[i][j] += this.data[i][k] * other.data[k][j];
                }
            }
        }
        return result;
    }

    //transpose matrix
    public Matrix transpose() {
        Matrix result = new Matrix(cols, rows);

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result.data[j][i] = data[i][j];
            }
        }
        return result;
    }

    // select subset of rows by index array
    public Matrix selectRows(int[] rowIndices) {
        Matrix result = new Matrix(rowIndices.length, this.cols);
        for (int i = 0; i < rowIndices.length; i++) {
            int r = rowIndices[i];
            System.arraycopy(this.data[r], 0, result.data[i], 0, this.cols);
        }
        return result;
    }

}
