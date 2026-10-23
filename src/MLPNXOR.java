import java.util.Scanner;

public class MLPNXOR {

    public static void main(String[] args) {

        double lr    = 0.1;
        int   epochs = 20000;
        int[] topology = {2, 2, 1};

        Matrix trX = new Matrix(
                new double[][] {
                        {0,0},
                        {0,1},
                        {1,0},
                        {1,1}
                });

        Matrix trY = new Matrix(
                new double[][] {
                        {1},
                        {0},
                        {0},
                        {1}
                });

        Scanner sc = new Scanner(System.in);
        sc.useLocale(java.util.Locale.US);

        int n = sc.nextInt();
        double[][] input = new double[n][2];

        for (int i = 0; i < n; i++) {
            input[i][0] = sc.nextDouble();
            input[i][1] = sc.nextDouble();
        }

        Matrix evX = new Matrix(input);

        MLP mlp = new MLP(
                topology,
                new IDifferentiableFunction[] {
                        new Sigmoid(),
                        new Sigmoid()
                },
                -1
        );

        mlp.train(trX, trY, lr, epochs);

        Matrix pred = mlp.predict(evX);
        pred = pred.apply(new Step().fnc());

        for (int i = 0; i < pred.rows(); i++) {
            System.out.println((int) pred.get(i, 0));
        }

        sc.close();
    }
}
