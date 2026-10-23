public class MLP {

    private Matrix[] w;
    private Matrix[] b;
    private Matrix[] yp;
    private IDifferentiableFunction[] act;
    private int numLayers;

    private double weightDecay = 0.0;

    public MLP(int[] layerSizes, IDifferentiableFunction[] act, int seed) {
        if (seed < 0)
            seed = (int) System.currentTimeMillis();

        numLayers = layerSizes.length;
        this.act = act;
        yp = new Matrix[numLayers];
        w = new Matrix[numLayers-1];
        b = new Matrix[numLayers-1];

        for (int i = 0; i < numLayers-1; i++) {
            w[i] = Matrix.RandXavier(layerSizes[i], layerSizes[i + 1], seed + i);
            b[i] = new Matrix(1, layerSizes[i + 1]);
        }
    }

    public void setWeightDecay(double lambda) {
        this.weightDecay = Math.max(0.0, lambda);
    }

    public Matrix predict(Matrix X) {
        yp[0] = X;
        for (int l = 0; l < numLayers-1; l++)
            yp[l+1] = yp[l].dot(w[l]).addRowVector(b[l]).apply(act[l].fnc());
        return yp[numLayers-1];
    }

    public double[] train(Matrix X, Matrix y, double learningRate, int epochs) {
        double[] mse = new double[epochs];

        for (int epoch = 0; epoch < epochs; epoch++) {
            mse[epoch] = trainEpoch(X, y, learningRate);
        }
        return mse;
    }

    public double trainEpoch(Matrix X, Matrix y, double learningRate) {
        Matrix predictions = predict(X);
        Matrix diff = y.sub(predictions);
        Matrix squared = diff.mult(diff);

        double mse = squared.sum() / (predictions.rows() * predictions.cols());
        backPropagation(X, y, learningRate);
        return mse;
    }

    public Matrix[] getWeights() {
        Matrix[] copy = new Matrix[w.length];
        for (int i = 0; i < w.length; i++)
            copy[i] = new Matrix(w[i].data);
        return copy;
    }

    public Matrix[] getBiases() {
        Matrix[] copy = new Matrix[b.length];
        for (int i = 0; i < b.length; i++)
            copy[i] = new Matrix(b[i].data);
        return copy;
    }

    public void setParameters(Matrix[] newW, Matrix[] newB) {
        if (newW.length != w.length || newB.length != b.length)
            throw new IllegalArgumentException("Shape mismatch on parameters");

        for (int i = 0; i < w.length; i++)
            w[i] = new Matrix(newW[i].data);

        for (int i = 0; i < b.length; i++)
            b[i] = new Matrix(newB[i].data);
    }

    public Matrix backPropagation(Matrix X, Matrix y, double lr) {
        Matrix e = null;
        Matrix delta = null;

        for (int l = numLayers-2; l >= 0; l--) {
            if (l == numLayers-2)
                e = y.sub(yp[l+1]);
            else
                e = delta.dot(w[l+1].transpose());

            Matrix dy = yp[l+1].apply(act[l].derivative());
            delta = e.mult(dy);

            Matrix gradW = yp[l].transpose().dot(delta).mult(lr);

            if (weightDecay > 0.0)
                gradW = gradW.sub(w[l].mult(weightDecay * lr));

            w[l] = w[l].add(gradW);
            b[l] = b[l].add(delta.sumColumns().mult(lr));
        }
        return e;
    }
}
