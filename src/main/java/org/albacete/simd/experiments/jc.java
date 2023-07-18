/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package org.albacete.simd.experiments;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.util.Arrays;
import org.albacete.simd.mAnDE.mAnDE;
import weka.classifiers.bayes.AveragedNDependenceEstimators.A1DE;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.meta.FilteredClassifier;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;
import weka.core.converters.CSVLoader;
import weka.filters.unsupervised.attribute.Discretize;

/**
 *
 * @author pablo
 */
public class jc {

    public static void main(String[] args) throws Exception {
        A1DE model = new A1DE();
        
        
        String path = "/home/pablo/Descargas/preferences/libras/";

        Discretize disc = new Discretize();
        disc.setUseEqualFrequency(false);
        disc.setBins(4);
        
        
        FilteredClassifier fc = new FilteredClassifier();
        fc.setFilter(disc);
        fc.setClassifier(model);
        
        for (int i = 0; i < 50; i++){
            // List all the files in the train directory
            File train = new File(path + i + "/train/");

            // Create a directory named "predictions" for each iteration
            File predictions = new File(path + i + "/predictions/");
            predictions.mkdir();
            
            
            File[] listOfFiles = train.listFiles();

            // Train a model for each file with the A1DE model
            for (File file : listOfFiles) {
                if (file.isFile()) {
                    File test = new File(path + i + "/test/" + file.getName());

                    // READ THE CSV FILE
                    CSVLoader loader = new CSVLoader();
                    loader.setSource(file);

                    Instances data = loader.getDataSet();
                    disc.setInputFormat(data);

                    // Specify that the features are categorical because they are already ordinal

                    data.setClassIndex(data.numAttributes()-1);

                    fc.buildClassifier(data);
                    
                    CSVLoader loader2 = new CSVLoader();
                    loader2.setSource(test);

                    Instances dataTest = loader2.getDataSet();
                    dataTest.setClassIndex(dataTest.numAttributes()-1);
                    
                    // Create a CSV file with the same name as the test file
                    BufferedWriter csvWriter = new BufferedWriter(new FileWriter(path + i + "/predictions/" + file.getName()));


                    // Write the header
                    String header = model.m_Instances.attribute("class").toString(); // The header is the same for all the files
                    header = header.substring(header.indexOf("{")+1, header.indexOf("}"));
                    header = header += "\n";
                    csvWriter.append(header);


                    double[][] pred = fc.distributionsForInstances(dataTest);
                    
                    // Write the predictions
                    for (int j = 0; j < dataTest.numInstances(); j++){
                        double[] prediction = pred[j];
                        String line = prediction[0] + "," + prediction[1] + "," + prediction[2] + "\n";
                        csvWriter.append(line);
                    }
                    
                    csvWriter.flush();
                    csvWriter.close();

                }
            }   
        }
    }
}

