package clasesTest;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.PrintWriter;
import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.CSVLoader;
import weka.core.converters.LibSVMLoader;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

/**
 *
 * @author Pablo Torrijos Arenas
 */
public class csvToArff {
    
    public static void main(String[] args) throws Exception {
        //String ruta = "C:\\Users\\pablo\\Desktop\\";
        String ruta = "/home/pablo/Escritorio/";

        /*
        FileWriter fw = new FileWriter(ruta + "secom2.csv");
        PrintWriter pw = new PrintWriter(fw);

        String cadena;
        FileReader f = new FileReader(ruta + "secom.csv");
        BufferedReader b = new BufferedReader(f);
        
        String cadena2;
        FileReader f2 = new FileReader(ruta + "secom_labels.csv");
        BufferedReader b2 = new BufferedReader(f2);
        
        while((cadena = b.readLine())!=null && (cadena2 = b2.readLine())!=null) {
            pw.println(cadena + " " + cadena2);
        }
        
        b.close();
        fw.close();
        */
        
        // CARGAR LA BASE DE DATOS 
        CSVLoader loader = new CSVLoader();        
        loader.setFieldSeparator(",");
        
        String name = "TCGA-PANCAN";
        
        loader.setSource(new File(ruta + name + ".csv"));
        
        
        //LibSVMLoader loader = new LibSVMLoader();
        //loader.setSource(new File(ruta + "rcv1.libsvm"));
        
        
        //loader.setNominalAttributes("first-last");
        loader.setNominalAttributes("last");
        loader.setNoHeaderRowPresent(false);
        Instances data = loader.getDataSet();
        
        /*Remove rm = new Remove();
        String[] opts = new String[]{"-R", "1"};
        //String[] opts = new String[]{"-R", "1,2"};
        rm.setOptions(opts);
        rm.setInputFormat(data);
        data = Filter.useFilter(data, rm);
        data.setClassIndex(data.numAttributes()-1);*/
        
        ArffSaver saver = new ArffSaver();
        saver.setInstances(data);
        saver.setFile(new File(ruta + name + ".arff"));
        saver.writeBatch();

    }
}