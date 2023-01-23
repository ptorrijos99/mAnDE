/*
 *  The MIT License (MIT)
 *  
 *  Copyright (c) 2022 Universidad de Castilla-La Mancha, España
 *  
 *  Permission is hereby granted, free of charge, to any person obtaining a copy
 *  of this software and associated documentation files (the "Software"), to deal
 *  in the Software without restriction, including without limitation the rights
 *  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 *  copies of the Software, and to permit persons to whom the Software is
 *  furnished to do so, subject to the following conditions:
 *  
 *  The above copyright notice and this permission notice shall be included in all
 *  copies or substantial portions of the Software.
 *  
 *  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 *  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 *  SOFTWARE.
 */

/**
 *    cvExperiment.java
 *    Copyright (C) 2023 Universidad de Castilla-La Mancha, España
 *
 * @author Pablo Torrijos Arenas
 *
 */
package org.albacete.simd.experiments;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.util.Arrays;
import java.util.Random;

import weka.core.Instances;
import weka.core.converters.ConverterUtils;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.Evaluation;

import weka.classifiers.evaluation.NominalPrediction;
import weka.classifiers.evaluation.Prediction;

import org.albacete.simd.mAnDE.mAnDE;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.meta.Bagging;
import weka.classifiers.meta.FilteredClassifier;
import weka.classifiers.sklearn.ScikitLearnClassifier;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.filters.supervised.attribute.Discretize;

public class cvExperiment {
    
    static Random random;
    static int folds;
    static Instances data;
    static String[] args;
    static String [] params;
    
    

    public static void main(String[] args) throws Exception {
        cvExperiment.args = args;

        // Reading arguments
        int index = Integer.parseInt(args[0]);
        String paramsFile = args[1];

        // Reading parameters
        try (BufferedReader br = new BufferedReader(new FileReader(paramsFile))) {
            // Reading index line
            String line;
            for (int i = 0; i < index; i++) {
                br.readLine();
            }
            line = br.readLine();

            //Splitting params
            params = line.split(" ");
        } catch (FileNotFoundException e) {
            System.out.println(e);
        }

        // Getting params from line: bbdd, algorithm, seed, folds, discretized, 
        // nTrees, featureSelection, baseClas, (n, ensemble, boosting, RF, bagSize)
        String bbdd = params[0];

        random = new Random(42);  
        
        folds = Integer.parseInt(params[2]);

        // Read data
        data = readData(bbdd);

        launchExperiment();
    }
    
    public static Instances readData(String bbdd) throws Exception {
        // Read bbdd
        ConverterUtils.DataSource loader = new ConverterUtils.DataSource("res/bbdd/" + bbdd);

        Instances data = loader.getDataSet();
        data.setClassIndex(data.numAttributes()-1);
        
        // Shuffle data
        data.randomize(random);
        
        return data;
    }
    
    public static void launchExperiment() throws Exception {
        String EXPERIMENTS_FOLDER = "results/";
        String savePath;
        
        // Set name of the save file
        savePath = "experiment_results_" 
                + params[0] + "_" + params[1] + "_" + params[2] + "_" 
                + params[3] + ".csv";
        
        System.out.println(savePath + "\n");
        
        savePath = EXPERIMENTS_FOLDER + savePath;

        File file = new File(savePath);
        
        // Leave-one-out CV
        int foldsExterna = data.numInstances();
        
        if(file.length() == 0) {
            double init = System.currentTimeMillis();
            
            int numValClass = data.classAttribute().numValues();

            double probAciertos = 0;
            double probFallos0 = 0;
            double probFallos1 = 0;
            
            double briefScore = 0;
            int[][] matriz = new int[numValClass][numValClass];

            for (int i = 0; i < foldsExterna; i++) {
                Instances train = data.trainCV(foldsExterna, i);
                Instances test = data.testCV(foldsExterna, i);

                AbstractClassifier bestModel = cvInterna(train);
                
                bestModel.buildClassifier(train);
                double[] prediction = bestModel.distributionForInstance(test.firstInstance());
                int posReal = (int)(test.firstInstance().classValue());
                
                // maxAt: Posición de la máxima probabilidad
                int maxAt = 0;
                double maxProb = -1;
                for (int j = 0; j < numValClass; j++) {
                    if (prediction[j] > maxProb) {
                        maxAt = j;
                        maxProb = prediction[j];
                    }
                }
                
                // Introducir predicción en la matriz. 
                //  Primer elemento -> Posición predicha
                //  Segundo elemento -> Posición real
                matriz[maxAt][posReal]++;
                
                // Cálculo del briefScore
                for (int j = 0; j < numValClass; j++) {
                    // Acierto
                    if (j == posReal) {
                        briefScore += Math.pow((1 - prediction[j]), 2);
                    } 
                    // Fallo
                    else {
                        briefScore += Math.pow((0 - prediction[j]), 2);
                    }
                }
                
                // probAciertos y probFallos

                if (maxAt == posReal) {
                    probAciertos += prediction[maxAt];
                } else {
                    probFallos0 += prediction[maxAt];
                    probFallos1 += prediction[posReal];
                }

            }
            
            double time = ((System.currentTimeMillis() - init) / foldsExterna) / 1000;

            briefScore /= foldsExterna;
            
            double pctCorrect = 0;
            for (int i = 0; i < numValClass; i++) {
                pctCorrect += matriz[i][i];
            }
            pctCorrect /= foldsExterna;
            
            System.out.println("\nMatriz: ");
            // Precision: Verdaderos positivos predichos / Total positivos predichos
            // Recall: Verdaderos positivos predichos / Total positivos reales
            double fm = 0;
            double precision = 0;
            double recall = 0;
            for (int i = 0; i < numValClass; i++) {
                System.out.println(Arrays.toString(matriz[i]));
                
                double truePosPred = matriz[i][i];
                
                double totalPosPred = 0;
                for (int j = 0; j < numValClass; j++) {
                    totalPosPred += matriz[j][i];
                }
                
                double totalPosReal = 0;
                for (int j = 0; j < numValClass; j++) {
                    totalPosReal += matriz[i][j];
                }
                
                double pres = 0, rec = 0;
                if (totalPosPred != 0)
                    pres = truePosPred / totalPosPred;
                
                if (totalPosReal != 0)
                    rec = truePosPred / totalPosReal;
                
                System.out.println(pres + ", " + rec + ", " + (2 * pres * rec / (pres + rec)));
                
                if ((pres + rec) != 0)
                    fm += 2 * pres * rec / (pres + rec);
                precision += pres;
                recall += rec;
            }

            fm = fm / numValClass;
            precision = precision / numValClass;
            recall = recall / numValClass;
                        
            System.out.println("\nF-Score -> " + fm + "\n");
            System.out.println("Execution time: " + time + "\n");

            BufferedWriter csvWriter = new BufferedWriter(new FileWriter(savePath, true));
            
            String header = "bbdd,algorithm,folds,n,score,fm,precision,recall,probAciertos,probPredFallos,probRealFallos,briefScore,time(s)\n";
            csvWriter.append(header);
            
            String output = params[0] + "," + params[1] + "," + params[2] + "," + params[3] + ","
                    + pctCorrect + "," + fm + "," + precision + ","
                    + recall + "," + probAciertos + ',' + probFallos0 + ','
                    + probFallos1 + ',' + briefScore + ',' + time + "\n";
            csvWriter.append(output);

            System.out.println("Results saved at: " + savePath);
            
            csvWriter.flush();
            csvWriter.close();
        } else{
            System.out.println("results already existing in " + savePath);
        }
    }
    
    public static AbstractClassifier cvInterna(Instances train) throws Exception {
        int[] nTrees = {50, 100, 150, 200};
        double[] porNBs = {0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4};
        
        String algorithm = params[1];
        
        if (folds == 0) folds = train.numInstances();
        
        AbstractClassifier result = null;
        double bestScore = -1;
        
        // Parallel execution as posible, and seed
        String[] options = new String[4];
        options[0] = "-num-slots";
        options[1] = "0";
        options[2] = "-S";
        options[3] = ""+42;
        
        String[] optionsSkL = new String[2];
        optionsSkL[0] = "-learner";
        
        switch (algorithm) {
            case "mAnDE":
                for (int nTree : nTrees) {
                    for (double porNB : porNBs) {
                        mAnDE clas = new mAnDE();
                        clas.setAddNB(porNB);
                        clas.setnTrees(nTree);
                        clas.setN(Integer.parseInt(params[2]));
                        
                        Evaluation evaluation = new Evaluation(train);
                        evaluation.crossValidateModel(clas, train, folds, random, new Object[]{});
                        
                        if (evaluation.pctCorrect() > bestScore) {
                            bestScore = evaluation.pctCorrect();
                            result = clas;
                        }
                    }
                }   
                break;
            case "mAnDE-0":
                for (int nTree : nTrees) {
                    mAnDE clas = new mAnDE();
                    clas.setAddNB(0);
                    clas.setnTrees(nTree);
                    clas.setN(Integer.parseInt(params[2]));

                    Evaluation evaluation = new Evaluation(train);
                    evaluation.crossValidateModel(clas, train, folds, random, new Object[]{});

                    if (evaluation.pctCorrect() > bestScore) {
                        bestScore = evaluation.pctCorrect();
                        result = clas;
                    }
                }   
                break;
            case "Bagging":
                for (int nTree : nTrees) {
                    Bagging clas = new Bagging();
                    clas.setOptions(options);
                    clas.setClassifier(new J48());
                    clas.setNumIterations(nTree);

                    Evaluation evaluation = new Evaluation(train);
                    evaluation.crossValidateModel(clas, train, folds, random, new Object[]{});

                    if (evaluation.pctCorrect() > bestScore) {
                        bestScore = evaluation.pctCorrect();
                        result = clas;
                    }
                    
                    FilteredClassifier fc = new FilteredClassifier();
                    Discretize discretizer = new Discretize();

                    fc.setFilter(discretizer);
                    fc.setClassifier(clas);

                    evaluation = new Evaluation(train);
                    evaluation.crossValidateModel(fc, train, folds, random, new Object[]{});

                    if (evaluation.pctCorrect() > bestScore) {
                        bestScore = evaluation.pctCorrect();
                        result = fc;
                    }
                }   
                break;
            case "RF":
                for (int nTree : nTrees) {
                    RandomForest clas = new RandomForest();
                    clas.setOptions(options);
                    clas.setNumIterations(nTree);

                    Evaluation evaluation = new Evaluation(train);
                    evaluation.crossValidateModel(clas, train, folds, random, new Object[]{});

                    if (evaluation.pctCorrect() > bestScore) {
                        bestScore = evaluation.pctCorrect();
                        result = clas;
                    }
                    
                    FilteredClassifier fc = new FilteredClassifier();
                    Discretize discretizer = new Discretize();

                    fc.setFilter(discretizer);
                    fc.setClassifier(clas);

                    evaluation = new Evaluation(train);
                    evaluation.crossValidateModel(fc, train, folds, random, new Object[]{});

                    if (evaluation.pctCorrect() > bestScore) {
                        bestScore = evaluation.pctCorrect();
                        result = fc;
                    }
                }   
                break;
            case "NB":
                return new NaiveBayes();
            default:
                break;
        }
        
        return result;
    }
}
