package clasesTest;

import java.io.File;
import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.CSVLoader;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.unsupervised.attribute.Transpose;

/**
 *
 * @author Pablo Torrijos Arenas
 */
public class geoMatrixToArff {
    public static void main(String[] args) throws Exception {
        String ruta = "/home/pablo/Documentos/- 5. Máster/4. TFG/ASD/";
        
        String name = "GSE27044_series_matrix";

        // CARGAR LA BASE DE DATOS 
        CSVLoader loader = new CSVLoader();        
        loader.setFieldSeparator("\t");
        loader.setSource(new File(ruta + name + ".txt"));
        loader.setNoHeaderRowPresent(true);
        
        Instances data = loader.getDataSet();

        System.out.println("LOADER");
        
        // TRANSPONER INSTANCIAS
        Transpose transpose = new Transpose();
        transpose.setInputFormat(data);
        data = Filter.useFilter(data, transpose);
        
        System.out.println("TRANSPOSE");
        
        // ELIMINAR INDICE
        Remove rm = new Remove();
        String[] opts = new String[]{"-R", "1"};
        rm.setOptions(opts);
        rm.setInputFormat(data);
        data = Filter.useFilter(data, rm);
        
        System.out.println("REMOVE");
        
        ArffSaver saver = new ArffSaver();
        saver.setInstances(data);
        saver.setFile(new File(ruta + name + ".arff"));
        saver.writeBatch();
    }
}
