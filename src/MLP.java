public class MLP {

    private Matrix[] w;  //weights for each layer
    private Matrix[] b;  //biases for each layer
    private Matrix[] yp; //outputs for each layer
    private IDifferentiableFunction[] act; //activation fnc for each layer
    private int numLayers;

    private double weightDecay = 0.0; // L2 regularization factor (lambda)

    /* PRE: layerSizes.length >= 2
     * PRE: act.length == layerSizes.length - 1
     */
    public MLP(int[] layerSizes, IDifferentiableFunction[] act, int seed) {
        if (seed < 0)
            seed = (int) System.currentTimeMillis();

        numLayers = layerSizes.length;
        this.act = act;   //setup activation by layer

        //create output storage for each layer but the input layer
        yp = new Matrix[numLayers];

        //create weights and biases for each layer
        w = new Matrix[numLayers-1];
        b = new Matrix[numLayers-1];

        for (int i = 0; i < numLayers-1; i++) {
            // Xavier initialization for Sigmoid
            w[i] = Matrix.RandXavier(layerSizes[i], layerSizes[i + 1], seed + i);
            b[i] = new Matrix(1, layerSizes[i + 1]); // biases initialized to zero
        }
    }

    // Optional: enable/disable L2 regularization
    public void setWeightDecay(double lambda) {
        this.weightDecay = Math.max(0.0, lambda);
    }

    // Feed forward propagation
    // yp[l+1] = Sigmoid( yp[l] * w[l] + b[l] )
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
        Matrix predictions = predict(X); // sets yp for backprop
        Matrix diff = y.sub(predictions);
        Matrix squared = diff.mult(diff);

        // Mean Squared Error
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

        // back propagation using generalised delta rule
        for (int l = numLayers-2; l >= 0; l--) {
            if (l == numLayers-2)                       // output layer
                e = y.sub(yp[l+1]);                     // e = y − yp[l+1]
            else {                                      // propagate error

                // insert code here to compute the error of layer l:
                //
                //      e = delta * w[l+1]^T
                e = delta.dot(w[l+1].transpose());
            }

            // insert code here to compute Delta:
            //
            //      dy = yp[l+1] .* (1 − yp[l+1])
            //      Note: to compute dy use Sigmoid class derivative
            //
            //      delta = e .* dy
            Matrix dy = yp[l+1].apply(act[l].derivative());
            delta = e.mult(dy);

            // insert code here to update the weights
            //
            //      w[l] += yp[l]^T * delta * lr
            Matrix gradW = yp[l].transpose().dot(delta).mult(lr);

            // L2 regularization (weight decay)
            if (weightDecay > 0.0)
                gradW = gradW.sub(w[l].mult(weightDecay * lr));

            w[l] = w[l].add(gradW);

            // update biases
            b[l] = b[l].add(delta.sumColumns().mult(lr));
        }
        return e;
    }
}
