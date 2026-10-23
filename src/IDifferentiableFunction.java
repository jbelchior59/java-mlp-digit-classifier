import java.util.function.Function;

public interface IDifferentiableFunction {
    Function<Double, Double> fnc();
    Function<Double, Double> derivative();
}
