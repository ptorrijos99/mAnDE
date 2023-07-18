package clasesTest;

/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */


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
 * @author pablo
 */
public class csvToArff {
    
    public static void main(String[] args) throws Exception {
        String ruta = "C:\\Users\\pablo\\Documents\\- 5. Máster\\4. TFG\\src\\test\\";
        
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
        
        loader.setSource(new File(ruta + "Brain_GSE50161.csv"));
        
        
        //LibSVMLoader loader = new LibSVMLoader();
        //loader.setSource(new File(ruta + "rcv1.libsvm"));
        
        
        //loader.setNominalAttributes("3");
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
        saver.setFile(new File(ruta + "Brain_GSE50161.arff"));
        saver.writeBatch();

    }
}