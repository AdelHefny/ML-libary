public class Main {
    public static void main(String[] args) {
        ML_Libary.Data_Manipulation data = new ML_Libary.Data_Manipulation();
        Integer[] col1 = {1, 2,3};
        String[] col2 = {"A", "B", "C"};
        Double[] col3 = {1.1,2.2, 3.3};
        ML_Libary.Data_Manipulation.Data_Frame df = ML_Libary.Data_Manipulation.read_csv("C:\\Me\\college\\software engineering\\ML libary\\src\\telecom_churn_clean.csv");
        df.print();
    }
}