import java.util.*;
import java.io.*;

public class Main {

    private static Scanner scanner = new Scanner(System.in);
    private static ML_Library.Data_Manipulation.Data_Frame df = null;
    private static Object model = null;
    private static ML_Library.SplitResult split = null;
    private static boolean isNormalized = false;
    private static ArrayList<Object> targetValues = new ArrayList<>();
    private static double testSize = 0;

    public static void main(String[] args) {
        System.out.println("Welcome to the Machine Learning CLI App!");

        while (true) {
            displayMainMenu();
            int mainChoice = getValidIntInput("Choose an option: ");

            if (mainChoice == 5 && confirmExit()) {
                System.out.println("Exiting...");
                scanner.close();
                return;
            }

            switch (mainChoice) {
                case 1: loadData(); break;
                case 2: loadModel(); break;
                case 3: normalizeData(); break;
                case 4: denormalizeData(); break;
                default: continue;
            }

            handleModelOperations();
        }
    }

    private static void displayMainMenu() {
        System.out.println("\nMain Menu:");
        System.out.println("1. Load Data");
        System.out.println("2. Load Model");
        System.out.println("3. Normalize Data");
        System.out.println("4. Denormalize Data");
        System.out.println("5. Exit");
    }

    private static boolean confirmExit() {
        System.out.print("Are you sure you want to exit? (y/n): ");
        String confirmation = scanner.next().trim().toLowerCase();
        scanner.nextLine(); // Consume newline
        return confirmation.equals("y") || confirmation.equals("yes");
    }

    private static int getValidIntInput(String prompt) {
        while (true) {
            System.out.print(prompt);
            try {
                int input = scanner.nextInt();
                scanner.nextLine(); // Consume newline
                return input;
            } catch (InputMismatchException e) {
                System.out.println("Please enter a valid number.");
                scanner.nextLine(); // Consume invalid input
            }
        }
    }

    private static double getValidDoubleInput(String prompt) {
        while (true) {
            System.out.print(prompt);
            try {
                double input = scanner.nextDouble();
                scanner.nextLine(); // Consume newline
                return input;
            } catch (InputMismatchException e) {
                System.out.println("Please enter a valid number.");
                scanner.nextLine(); // Consume invalid input
            }
        }
    }

    private static void loadData() {
        System.out.print("Enter the full path or filename of the CSV file: ");
        String filePath = scanner.nextLine().trim();

        File file = new File(filePath);
        if (!file.exists()) {
            System.out.println("File not found. Please enter a valid path.");
            return;
        }

        try {
            df = ML_Library.Data_Manipulation.read_csv(filePath);
        } catch (Exception e) {
            System.out.println("Error processing file: " + e.getMessage());
            return;
        }

        // Display columns and select target
        System.out.println("Columns in dataset:");
        for (int i = 0; i < df.colSize; i++) {
            System.out.println(i + ": " + df.getCol(i).get(0));
        }

        int targetColumn = getValidIntInput("Enter the index of the target column: ");

        if (targetColumn < 0 || targetColumn >= df.size) {
            System.out.println("Invalid column index. Please try again.");
            return;
        }

        targetValues = df.values(targetColumn);
        df = df.drop(targetColumn);

        double testPercentage = getValidDoubleInput("Enter the percentage of data to use for testing (0-100): ");
        testSize = testPercentage / 100.0;

        if (testSize < 0 || testSize > 1) {
            System.out.println("Test size must be between 0% and 100%. Please try again.");
            return;
        }

        split = ML_Library.trainTestSplit(df, targetValues, testSize);
        System.out.println("Data split: " + (1 - testSize) * 100 + "% training, " + testSize * 100 + "% testing.");
    }

    private static void loadModel() {
        System.out.print("Enter the filename of the saved model: ");
        String modelFilename = scanner.nextLine().trim();

        try (BufferedReader reader = new BufferedReader(new FileReader(modelFilename))) {
            String modelType = reader.readLine().trim();

            if ("KNN".equals(modelType)) {
                model = new ML_Library.classifiers.K_Nearest_Neighbor();
                ((ML_Library.classifiers.K_Nearest_Neighbor) model).loadModel(modelFilename);
                System.out.println("KNN Model loaded successfully from " + modelFilename);
            } else if ("LinearRegression".equals(modelType)) {
                model = new ML_Library.Regressors.Linear_Regression(0.01, 1000);
                ((ML_Library.Regressors.Linear_Regression) model).loadModel(modelFilename);
                System.out.println("Linear Regression Model loaded successfully from " + modelFilename);
            } else {
                System.out.println("Error: Unknown model type in file.");
            }
        } catch (IOException e) {
            System.err.println("Error loading model: " + e.getMessage());
        }
    }

    private static void normalizeData() {
        if (df == null) {
            System.out.println("There is no data to normalize.");
            return;
        }

        if (isNormalized) {
            System.out.println("Data has already been normalized.");
            return;
        }

        df.Z_Score_Normalize();
        isNormalized = true;
        System.out.println("Data has been normalized.");
    }

    private static void denormalizeData() {
        if (df == null) {
            System.out.println("There is no data to denormalize.");
            return;
        }

        if (!isNormalized) {
            System.out.println("Data hasn't been normalized.");
            return;
        }

        df.DeNormalize();
        isNormalized = false;
        System.out.println("Data has been denormalized.");
    }

    private static void handleModelOperations() {
        while (true) {
            displayModelMenu();
            int modelChoice = getValidIntInput("Choose an option: ");

            if (modelChoice == 4) break;

            switch (modelChoice) {
                case 1: trainNewModel(); break;
                case 2: predictUsingModel(); break;
                case 3: saveModel(); break;
                default: continue;
            }
        }
    }

    private static void displayModelMenu() {
        System.out.println("\nModel Menu:");
        System.out.println("1. Train New Model");
        System.out.println("2. Predict Using Model");
        System.out.println("3. Save Model");
        System.out.println("4. Back to Main Menu");
    }

    private static void trainNewModel() {
        if (df == null) {
            System.out.println("No data provided, please enter data then try again.");
            return;
        }

        split = ML_Library.trainTestSplit(df, targetValues, testSize);

        System.out.println("Select a model type:");
        System.out.println("1. Classification (KNN)");
        System.out.println("2. Regression (Linear Regression)");
        int modelType = getValidIntInput("Enter choice: ");

        try {
            if (modelType == 1) {
                trainKNNModel();
            } else if (modelType == 2) {
                trainLinearRegressionModel();
            } else {
                System.out.println("Invalid model type selection.");
                return;
            }
            System.out.println("Training complete! Model is ready.");
        } catch (Exception e) {
            System.out.println("Error training model: " + e.getMessage());
        }
    }

    private static void trainKNNModel() {
        int k = getValidIntInput("Enter number of neighbors (k): ");
        ML_Library.classifiers.K_Nearest_Neighbor knn = new ML_Library.classifiers.K_Nearest_Neighbor();
        knn.train(split.X_train, split.y_train, k);
        model = knn;
    }

    private static void trainLinearRegressionModel() {
        double learningRate = getValidDoubleInput("Enter learning rate: ");
        int iterations = getValidIntInput("Enter number of iterations: ");

        ML_Library.Regressors.Linear_Regression lr = new ML_Library.Regressors.Linear_Regression(learningRate, iterations);
        lr.train(split.X_train, ML_Library.DoubleArrayConverter(split.y_train));
        model = lr;
    }

    private static void predictUsingModel() {
        if (model == null) {
            System.out.println("No model trained or loaded, please train or load a model to start.");
            return;
        }

        System.out.println("Choose prediction type:");
        System.out.println("1. Predict Specific Data");
        System.out.println("2. Predict from Test Data");
        int predictChoice = getValidIntInput("Enter choice: ");

        try {
            if (predictChoice == 1) {
                predictSpecificData();
            } else if (predictChoice == 2 && split != null) {
                predictFromTestData();
            } else {
                System.out.println("Invalid choice or missing test data.");
            }
        } catch (Exception e) {
            System.out.println("Error making prediction: " + e.getMessage());
        }
    }

    private static void predictSpecificData() throws Exception {
        System.out.println("Enter test data row (comma-separated values):");
        String input = scanner.nextLine();
        String[] values = input.split(",");
        ArrayList<Double> testData = new ArrayList<>();

        for (String value : values) {
            testData.add(Double.parseDouble(value.trim()));
        }

        if (model instanceof ML_Library.classifiers.K_Nearest_Neighbor) {
            Object prediction = ((ML_Library.classifiers.K_Nearest_Neighbor) model).predictSingle(testData);
            System.out.println("Predicted class: " + prediction);
        } else if (model instanceof ML_Library.Regressors.Linear_Regression) {
            double prediction = ((ML_Library.Regressors.Linear_Regression) model).predict(testData);
            System.out.println("Predicted value: " + prediction);
        }
    }

    private static void predictFromTestData() throws Exception {
        if (model instanceof ML_Library.classifiers.K_Nearest_Neighbor) {
            ArrayList<Object> predictions = ((ML_Library.classifiers.K_Nearest_Neighbor) model).predict(split.X_test);
            double accuracy = ((ML_Library.classifiers.K_Nearest_Neighbor) model).accuracy_score(split.y_test, predictions);
            System.out.println("Predictions: " + predictions);
            System.out.println("Accuracy: " + accuracy);
        } else if (model instanceof ML_Library.Regressors.Linear_Regression) {
            ArrayList<Double> predictions = new ArrayList<>();
            for (ArrayList<Double> test_row : split.X_test) {
                predictions.add(((ML_Library.Regressors.Linear_Regression) model).predict(test_row));
            }
            double r2 = ((ML_Library.Regressors.Linear_Regression) model).r2Score(split.X_test, ML_Library.DoubleArrayConverter(split.y_test));
            System.out.println("Predictions: " + predictions);
            System.out.println("R^2 Score: " + r2);
        }
    }

    private static void saveModel() {
        if (model == null) {
            System.out.println("No model trained or loaded, please train or load a model to start.");
            return;
        }

        System.out.print("Enter the filename to save the model: ");
        String modelFilename = scanner.nextLine().trim();

        try {
            if (model instanceof ML_Library.classifiers.K_Nearest_Neighbor) {
                ((ML_Library.classifiers.K_Nearest_Neighbor) model).saveModel(modelFilename);
            } else if (model instanceof ML_Library.Regressors.Linear_Regression) {
                ((ML_Library.Regressors.Linear_Regression) model).saveModel(modelFilename);
            }
            System.out.println("Model saved as " + modelFilename);
        } catch (Exception e) {
            System.out.println("Error saving model: " + e.getMessage());
        }
    }

    public class ML_Library {
        public static class SplitResult {
            public ArrayList<ArrayList<Double>> X_train, X_test;
            public ArrayList<Object> y_train, y_test;

            public SplitResult(ArrayList<ArrayList<Double>> X_train, ArrayList<ArrayList<Double>> X_test,
                               ArrayList<Object> y_train, ArrayList<Object> y_test) {
                this.X_train = X_train;
                this.X_test = X_test;
                this.y_train = y_train;
                this.y_test = y_test;
            }
        }

        public static SplitResult trainTestSplit(ArrayList<ArrayList<Double>> X, ArrayList<Object> y, double testSize) {
            if (X == null || y == null) {
                throw new IllegalArgumentException("X or y cannot be null");
            }
            if (X.size() != y.size()) {
                throw new IllegalArgumentException("X and y must have the same number of rows");
            }
            if (testSize < 0 || testSize > 1) {
                throw new IllegalArgumentException("Test size must be between 0 and 1.");
            }

            int totalSize = X.size();
            int testSizeCount = (int) (totalSize * testSize);
            int trainSizeCount = totalSize - testSizeCount;
            List<Integer> indices = new ArrayList<>();

            for (int i = 0; i < totalSize; i++) {
                indices.add(i);
            }
            Collections.shuffle(indices);

            ArrayList<ArrayList<Double>> X_train = new ArrayList<>();
            ArrayList<ArrayList<Double>> X_test = new ArrayList<>();
            ArrayList<Object> y_train = new ArrayList<>();
            ArrayList<Object> y_test = new ArrayList<>();

            for (int i = 0; i < trainSizeCount; i++) {
                X_train.add(X.get(indices.get(i)));
                y_train.add(y.get(indices.get(i)));
            }
            for (int i = 0; i < testSizeCount; i++) {
                X_test.add(X.get(indices.get(trainSizeCount + i)));
                y_test.add(y.get(indices.get(trainSizeCount + i)));
            }

            return new SplitResult(X_train, X_test, y_train, y_test);
        }

        public static SplitResult trainTestSplit(Data_Manipulation.Data_Frame X, ArrayList<Object> y, double testSize) {
            ArrayList<ArrayList<Double>> x = new ArrayList<>();
            for (int i = 1; i < X.size; i++) {
                x.add(DoubleArrayConverter(X.getRow(i)));
            }
            return trainTestSplit(x,y,testSize);
        }

        public static ArrayList<Double> DoubleArrayConverter(ArrayList<?> x) {
            ArrayList<Double> row = new ArrayList<>();
            for (Object value : x) {
                if (value instanceof Double) {
                    row.add((Double) value);
                } else if (value instanceof Integer) {
                    row.add(((Integer) value).doubleValue());
                } else {
                    throw new IllegalArgumentException("Not a number in the input");
                }
            }
            return row;
        }

        public static class Regressors {
            public static class Linear_Regression {
                private ArrayList<Double> weights;
                private double bias;
                private double learningRate;
                private int iterations;
                Linear_Regression(double learningRate,int iterations){
                    this.learningRate = learningRate;
                    this.iterations = iterations;
                }
                public void train(ArrayList<ArrayList<Double>> X, ArrayList<Double> y) {
                    int m = X.size();
                    int n = X.get(0).size();
                    weights = new ArrayList<>(n);
                    for (int i = 0; i < n; i++) {
                        weights.add(0.0);
                    }
                    bias = 0.0;
                    for (int iter = 0; iter < iterations; iter++) {
                        ArrayList<Double> gradients = new ArrayList<>(n);
                        for (int i = 0; i < n; i++) {
                            gradients.add(0.0);
                        }

                        // updating gradients with the derivative of the cost function imputing with the current values of weights and bias
                        double biasGradient = 0.0;
                        for (int i = 0; i < m; i++) {
                            double prediction = predict(X.get(i));
                            double error = prediction - y.get(i);
                            for (int j = 0; j < n; j++) {
                                gradients.set(j, gradients.get(j) + (error * X.get(i).get(j)) / m);
                            }
                            biasGradient += (error) / m;
                        }
                        // setting each weight to its new value
                        for (int j = 0; j < n; j++) {
                            if (Double.isNaN(gradients.get(j)) || Double.isInfinite(gradients.get(j))) {
                                throw new RuntimeException("Warning: Gradient exploded at iteration " + iter);
                            }
                            weights.set(j, weights.get(j) - learningRate * gradients.get(j));
                        }
                        bias -= learningRate * biasGradient;
                        if (iter % 100 == 0) {
                            System.out.println("Iteration " + iter + " | Cost: " + computeCost(X, y));
                        }
                    }
                }
                public double predict(ArrayList<Double> x) {
                    double prediction = bias;
                    for (int i = 0; i < weights.size(); i++) {
                        prediction += weights.get(i) * x.get(i);
                    }
                    return prediction;
                }
                public ArrayList<Double> predictArray(ArrayList<ArrayList<Double>> x) {
                    ArrayList<Double> predictions = new ArrayList<>();
                    for (ArrayList<Double> doubles : x) {
                        predictions.add(predict(doubles));
                    }
                    return predictions;
                }
                public double computeCost(ArrayList<ArrayList<Double>> X, ArrayList<Double> y) {
                    int m = X.size();
                    double cost = 0.0;
                    for (int i = 0; i < m; i++) {
                        double prediction = predict(X.get(i));
                        double e = prediction - y.get(i);
                        cost += e * e;
                    }
                    return cost / (2 * m);
                }
                public void printFunctionParams() {
                    System.out.println("Weights: " + weights);
                    System.out.println("Bias: " + bias);
                }
                public double r2Score(ArrayList<ArrayList<Double>> X, ArrayList<Double> y) {
                    double sum = 0.0;
                    for (double value : y) {
                        sum += value;
                    }
                    double meanY = sum / y.size();
                    double totalVariation = 0.0;
                    double explainedVariation = 0.0;

                    for (int i = 0; i < y.size(); i++) {
                        double prediction = predict(X.get(i));
                        totalVariation += Math.pow(y.get(i) - meanY, 2);
                        explainedVariation += Math.pow(y.get(i) - prediction, 2);
                    }
                    if (totalVariation == 0) return 1;
                    return 1 - (explainedVariation / totalVariation);
                }
                public void saveModel(String filename) {
                    try (BufferedWriter writer = new BufferedWriter(new FileWriter(filename))) {
                        writer.write("LinearRegression\n");
                        for (double weight : weights) {
                            writer.write(weight + " ");
                        }
                        writer.newLine();
                        writer.write(bias + "");
                        System.out.println("Model saved to " + filename);
                    } catch (IOException e) {
                        System.err.println("Error saving model: " + e.getMessage());
                    }
                }

                public void loadModel(String filename) {
                    try (BufferedReader reader = new BufferedReader(new FileReader(filename))) {
                        reader.readLine();
                        String[] weightStr = reader.readLine().split(" ");
                        weights = new ArrayList<>();
                        for (String w : weightStr) {
                            weights.add(Double.parseDouble(w));
                        }
                        bias = Double.parseDouble(reader.readLine());
                    } catch (IOException e) {
                        System.err.println("Error loading model: " + e.getMessage());
                    }
                }

            }
        }
        public class classifiers {
            public static class K_Nearest_Neighbor {
                ArrayList<ArrayList<Double>> x;ArrayList<Object> y;int k;
                K_Nearest_Neighbor(Data_Manipulation.Data_Frame x, ArrayList<Object> y, int k) throws IllegalArgumentException {
                    checks(x, y, k);
                    this.x = new ArrayList<>();
                    for (int i = 0; i < x.size; i++) {
                        this.x.add(DoubleArrayConverter(x.getRow(i)));
                    }

                    this.y = new ArrayList<>(y);
                    this.k = k;

                    if (this.k <= 0 || this.k > this.x.size()) {
                        throw new IllegalArgumentException("Invalid value for k.");
                    }
                }

                K_Nearest_Neighbor(ArrayList<ArrayList<Double>> x, ArrayList<Object> y, int k) throws IllegalArgumentException {
                    checks(x, y, k);

                    this.x = new ArrayList<>();
                    for (ArrayList<?> row : x) {
                        this.x.add(DoubleArrayConverter(row));
                    }

                    this.y = new ArrayList<>(y);
                    this.k = k;
                }

                K_Nearest_Neighbor(){}
                private double euclideanDistance(ArrayList<Double> p1, ArrayList<Double> p2) {
                    double sum = 0;
                    for (int i = 0; i < p1.size(); i++) {
                        sum += Math.pow(p1.get(i) - p2.get(i), 2);
                    }
                    return Math.sqrt(sum);
                }

                public void train(Data_Manipulation.Data_Frame x, ArrayList<Object> y, int k) throws IllegalArgumentException {
                    checks(x, y, k);

                    ArrayList<ArrayList<Double>> X = new ArrayList<>();
                    for (int i = 0; i < x.size; i++) {
                        X.add(DoubleArrayConverter(x.getRow(i)));
                    }

                    this.x = X;
                    this.y = new ArrayList<>(y);
                    this.k = k;
                }

                public void train(ArrayList<ArrayList<Double>> x, ArrayList<Object> y, int k) throws IllegalArgumentException {
                    checks(x, y, k);

                    this.x = x;
                    this.y = new ArrayList<>(y);
                    this.k = k;
                }

                private void checks(Data_Manipulation.Data_Frame x, ArrayList<Object> y, int k) {
                    if (x == null || y == null || x.size == 0 || y.isEmpty()) {
                        throw new IllegalArgumentException("Training data cannot be null or empty.");
                    }
                    if (k <= 0 || k > x.size) {
                        throw new IllegalArgumentException("Invalid value for k.");
                    }
                }

                private void checks(ArrayList<ArrayList<Double>> x, ArrayList<Object> y, int k) {
                    if (x == null || y == null || x.isEmpty() || y.isEmpty()) {
                        throw new IllegalArgumentException("Training data cannot be null or empty.");
                    }
                    if (k <= 0 || k > x.size()) {
                        throw new IllegalArgumentException("Invalid value for k.");
                    }
                }
                class Pair implements Comparable<Pair> {
                    int value;
                    double distance;

                    public Pair(int value, double distance) {
                        this.value = value;
                        this.distance = distance;
                    }

                    @Override
                    public int compareTo(Pair other) {
                        return Double.compare(this.distance, other.distance);
                    }
                }


                public Object predictSingle(ArrayList<Double> x) {
                    PriorityQueue<Pair> pq = new PriorityQueue<>((a, b) -> Double.compare(b.distance, a.distance));
                    for (int i = 0; i < this.x.size(); i++) {
                        ArrayList<Double> row = this.x.get(i);
                        double dist = euclideanDistance(row,x);
                        pq.add(new Pair(i, dist));
                        if(pq.size() > this.k){
                            pq.poll();
                        }
                    }
                    Object element = getElement(pq);
                    return element;
                }
                public ArrayList<Object> predict(ArrayList<ArrayList<Double>> x) {
                    ArrayList<Object> y = new ArrayList<>();
                    for (int i = 0; i < x.size(); i++) {
                        ArrayList<Double> row = x.get(i);
                        y.add(predictSingle(row));
                    }
                    return y;
                }
                public ArrayList<Object> predict(Data_Manipulation.Data_Frame x) {
                    ArrayList<Object> y = new ArrayList<>();
                    for (int i = 0; i < x.size; i++) {
                        ArrayList<Double> row = DoubleArrayConverter(x.getRow(i));
                        y.add(predictSingle(row));
                    }
                    return y;
                }

                private Object getElement(PriorityQueue<Pair> pq) {
                    Map<Object, Integer> mp = new HashMap<>();
                    ArrayList<Pair> NN = new ArrayList<>(pq);
                    for(Pair p : NN){
                        Object label = this.y.get(p.value);
                        mp.put(label, mp.getOrDefault(label, 0) + 1);
                    }
                    int maxi = 0;
                    Object element = null;
                    for (Map.Entry<Object,Integer> entry : mp.entrySet()) {
                        if(entry.getValue() > maxi){
                            maxi = entry.getValue();
                            element = entry.getKey();
                        }
                    }
                    return element;
                }

                public double accuracy_score(ArrayList<Object> yTrue,ArrayList<Object> yPredict) {
                    int right = 0;
                    for (int i = 0; i < yTrue.size(); i++) {
                        if(yTrue.get(i).equals(yPredict.get(i))){
                            right++;
                        }
                    }
                    return (double) right / (double) yTrue.size();
                }
                public void saveModel(String filename) {
                    try (BufferedWriter writer = new BufferedWriter(new FileWriter(filename))) {
                        writer.write("KNN\n");
                        writer.write(k + "\n");

                        for (ArrayList<Double> row : x) {
                            for (Double value : row) {
                                writer.write(value + " ");
                            }
                            writer.newLine();
                        }

                        for (Object label : y) {
                            writer.write(label.toString() + "\n");
                        }

                        System.out.println("Model saved to " + filename);
                    } catch (IOException e) {
                        System.err.println("Error saving model: " + e.getMessage());
                    }
                }

                public void loadModel(String filename) {
                    try (BufferedReader reader = new BufferedReader(new FileReader(filename))) {
                        reader.readLine();
                        this.k = Integer.parseInt(reader.readLine().trim());
                        this.x = new ArrayList<>();
                        String line;
                        while ((line = reader.readLine()) != null && !line.matches("[0-9]+")) {
                            String[] values = line.trim().split(" ");
                            ArrayList<Double> row = new ArrayList<>();
                            for (String value : values) {
                                row.add(Double.parseDouble(value));
                            }
                            this.x.add(row);
                        }
                        this.y = new ArrayList<>();
                        while (line != null) {
                            this.y.add(line.trim());
                            line = reader.readLine();
                        }
                    } catch (IOException e) {
                        System.err.println("Error loading model: " + e.getMessage());
                    }
                }

            }
        }
        public static class Data_Manipulation{
            public static class Data_Frame {

                private ArrayList<ArrayList<Object>> data;
                public int size,colSize;

                public Data_Frame() {
                    this.data = new ArrayList<>();
                    this.size = 0;
                    this.colSize = 0;
                }
                @SafeVarargs
                public <T> Data_Frame(T[]... columns) {
                    this();
                    this.colSize = columns.length;
                    for (T[] column : columns){
                        size = Math.max(size,column.length);
                    }
                    for (T[] column : columns) {
                        ArrayList<Object> colList = new ArrayList<>();
                        for (T value : column) {
                            colList.add(value);
                        }
                        while(colList.size() < size){
                            colList.add(null);
                        }
                        data.add(colList);
                    }
                }
                public void add_column(ArrayList<Object> column) {
                    data.add(column);
                    size = Math.max(size,column.size());
                    while (column.size() < size) {
                        column.add(null);
                    }
                    for (int i = 0; i < data.size(); i++) {
                        while (data.get(i).size() < size) {
                            data.get(i).add(null);
                        }
                    }
                    colSize++;
                }

                public static void checkType(Object obj) {
                    if (obj instanceof Integer) {
                        System.out.println("It's an Integer!");
                    } else if (obj instanceof Double) {
                        System.out.println("It's a Double!");
                    } else if (obj instanceof String) {
                        System.out.println("It's a String!");
                    } else {
                        System.out.println("Unknown type!");
                    }
                }
                public Data_Frame drop(int idx){
                    Data_Frame frame = new Data_Frame();
                    for (int i = 0; i < data.size(); i++) {
                        if(i == idx) continue;
                        frame.add_column(data.get(i));
                    }
                    return frame;
                }
                public Data_Frame drop(String name){
                    Data_Frame frame = new Data_Frame();
                    for (int i = 0; i < data.size(); i++) {
                        if(data.get(i).get(0).equals(name)) continue;
                        frame.add_column(data.get(i));
                    }
                    return frame;
                }
                public void print() {
                    if (data.isEmpty()) {
                        System.out.println("DataFrame is empty!");
                        return;
                    }
                    for (int i = 0; i < size; i++) {
                        for (ArrayList<Object> column : data) {
                            System.out.print(column.get(i) + "\t");
                        }
                        System.out.println();
                    }
                }
                public ArrayList<Object> getRow(int idx) {
                    ArrayList<Object> row = new ArrayList<>();
                    for (int i = 0; i < data.size(); i++) {
                        row.add(data.get(i).get(idx));
                    }
                    return row;
                }
                public ArrayList<Object> getCol(int idx) {
                    return data.get(idx);
                }
                public ArrayList<Object> getCol(String colName) {
                    for (int i = 0; i < data.size(); i++) {
                        if(colName.equals(data.get(i).get(0))){
                            return data.get(i);
                        }
                    }
                    return null;
                }
                public ArrayList<Object> values(int colIdx) {
                    ArrayList<Object> col = new ArrayList<>(data.get(colIdx));
                    col.removeFirst();
                    return col;
                }
                public ArrayList<Object> values(String name){
                    ArrayList<Object> col = new ArrayList<>();
                    for (int i = 0; i < data.size(); i++) {
                        if(data.get(i).get(0).equals(name)) {
                            col = data.get(i);
                            col.removeFirst();
                            return col;
                        }
                    }
                    return null;
                }
                private ArrayList<Double> means = new ArrayList<>();
                private ArrayList<Double> stds = new ArrayList<>();
                public Data_Frame Z_Score_Normalize() {
                    int numCols = data.size();
                    int numRows = size;
                    means = new ArrayList<>(Collections.nCopies(numCols, 0.0));
                    stds = new ArrayList<>(Collections.nCopies(numCols, 0.0));
                    // Step 1: Compute Mean for Numeric Columns
                    for (int j = 0; j < numCols; j++) {
                        double sum = 0.0;
                        int count = 0;
                        for (int i = 1; i < numRows; i++) {
                            Object value = data.get(j).get(i);
                            if (value instanceof Number) {
                                sum += ((Number) value).doubleValue();
                                count++;
                            }
                        }
                        if (count > 0) {
                            means.set(j, sum / count);
                        }
                    }
                    // Step 2: Compute Standard Deviation for Numeric Columns
                    for (int j = 0; j < numCols; j++) {
                        double sum = 0.0;
                        int count = 0;
                        for (int i = 1; i < numRows; i++) {
                            Object value = data.get(j).get(i);
                            if (value instanceof Number) {
                                sum += Math.pow(((Number) value).doubleValue() - means.get(j), 2);
                                count++;
                            }
                        }
                        if (count > 0) {
                            stds.set(j, Math.sqrt(sum / count));
                        }
                    }
                    // Step 3: Normalize Numeric Columns
                    for (int j = 0; j < numCols; j++) {
                        for (int i = 1; i < numRows; i++) {
                            Object value = data.get(j).get(i);
                            if (value instanceof Number) {
                                double std = stds.get(j);
                                if (std != 0) {
                                    data.get(j).set(i, (((Number) value).doubleValue() - means.get(j)) / std);
                                } else {
                                    data.get(j).set(i, 0.0);
                                }
                            }
                        }
                    }
                    return this;
                }
                public Data_Frame DeNormalize() {
                    if (means == null || stds == null || means.isEmpty() || stds.isEmpty()) {
                        throw new IllegalStateException("Data has not been normalized yet!");
                    }

                    int numCols = data.size();
                    int numRows = size;

                    for (int j = 0; j < numCols; j++) {
                        for (int i = 1; i < numRows; i++) {
                            Object value = data.get(j).get(i);
                            if (value instanceof Number) {
                                double originalValue = ((Number) value).doubleValue() * stds.get(j) + means.get(j);
                                data.get(j).set(i, originalValue);
                            }
                        }
                    }
                    return this;
                }
            }

            private static Object tryParse(String value) {
                try { return Integer.parseInt(value); }
                catch (NumberFormatException e1) {
                    try { return Double.parseDouble(value); }
                    catch (NumberFormatException e2) {
                        return value;
                    }
                }
            }
            public static Data_Frame read_csv(String filename) {
                Data_Frame df = new Data_Frame();
                ArrayList<ArrayList<Object>> rows = new ArrayList<>();

                try (BufferedReader reader = new BufferedReader(new FileReader(filename))) {
                    String line;

                    while ((line = reader.readLine()) != null) {
                        String[] values = line.split(",");
                        ArrayList<Object> row = new ArrayList<>();

                        for (String value : values) {
                            row.add(tryParse(value.trim()));
                        }

                        rows.add(row);
                    }
                } catch (IOException e) {
                    throw new RuntimeException("Error reading file: " + e.getMessage());
                }
                if (!rows.isEmpty()) {
                    int numColumns = rows.get(0).size();
                    for (int col = 0; col < numColumns; col++) {
                        ArrayList<Object> column = new ArrayList<>();
                        for (ArrayList<Object> row : rows) {
                            if (col < row.size()) {
                                column.add(row.get(col));
                            } else {
                                column.add(null);
                            }
                        }
                        df.add_column(column);
                    }
                }

                return df;
            }
        }
    }
}