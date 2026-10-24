import java.util.function.Function;

public class Step implements IDifferentiableFunction {

    static private double threshold = 0.5;

    @Override
    public Function<Double, Double> fnc() {

        // insert code here to return a lambda function that returns  
        // 0 if its argument is < 0 and 1 otherwise 
        return (z) -> z < threshold ? 0.0 : 1.0;
    }

    @Override
    public Function<Double, Double> derivative() {
        throw new UnsupportedOperationException("Step function is not differentiable.");
    }

}
