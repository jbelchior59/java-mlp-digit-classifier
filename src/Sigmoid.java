import java.util.function.Function;

public class Sigmoid implements IDifferentiableFunction  {

    @Override
    public Function<Double, Double> fnc() {
        return (z) -> 1.0 / (1.0 + Math.exp(-z));
    }

    @Override
    public Function<Double, Double> derivative() {
        return (y) -> y * (1.0 - y);
    }
}
