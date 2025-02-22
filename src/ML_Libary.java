import javax.xml.crypto.Data;
import java.util.*;
import java.io.*;

public class ML_Libary {
    public static ArrayList<Object> train_test_split(Data_Manipulation.Data_Frame x,ArrayList<Object> y){
        return new ArrayList<>();
    }
    class Regressors {
        public static class Linear_Regression {
            public void fit(){

            }
        }
    }
    class classifiers {
        public static class K_Nearest_Neighbor {
            public void fit(){

            }

        }
        public static class Logistic_Regression {
            public void fit(){

            }
        }
    }
    public static class Data_Manipulation{
        public static class Data_Frame {
            private ArrayList<ArrayList<Object>> data;
            private int maxi;

            public Data_Frame() {
                this.data = new ArrayList<>();
                this.maxi = 0;
            }
            @SafeVarargs
            public <T> Data_Frame(T[]... columns) {
                this();
                for (T[] column : columns){
                    maxi = Math.max(maxi,column.length);
                }
                for (T[] column : columns) {
                    ArrayList<Object> colList = new ArrayList<>();
                    for (T value : column) {
                        colList.add(value);
                    }
                    while(colList.size() < maxi){
                        colList.add(null);
                    }
                    data.add(colList);
                }
            }
            public void add_column (ArrayList<Object> column){
                while(column.size() > maxi){
                    for (int i = 0; i < data.size(); i++) {
                        data.get(i).add(null);
                    }
                    maxi++;
                }
                data.add(column);
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
            public void print() {
                if (data.isEmpty()) {
                    System.out.println("DataFrame is empty!");
                    return;
                }
                for (int i = 0; i < data.get(0).size(); i++) {
                    System.out.println(data.get(0).get(i).getClass());
                }
                for (int i = 0; i < maxi; i++) {
                    for (ArrayList<Object> column : data) {
                        System.out.print(column.get(i) + "\t");
                    }
                    System.out.println();
                }
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
