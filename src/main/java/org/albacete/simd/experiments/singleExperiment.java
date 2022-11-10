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
 *    singleExperiment.java
 *    Copyright (C) 2022 Universidad de Castilla-La Mancha, España
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
import java.util.Random;
import org.albacete.simd.mAnDE.mAnDE;
import weka.attributeSelection.ASEvaluation;
import weka.attributeSelection.ASSearch;
import weka.attributeSelection.CfsSubsetEval;
import weka.attributeSelection.FCBFSearch;
import weka.attributeSelection.GainRatioAttributeEval;
import weka.attributeSelection.GreedyStepwise;
import weka.attributeSelection.IWSS;
import weka.attributeSelection.IWSSembeddedNB;
import weka.attributeSelection.InfoGainAttributeEval;
import weka.attributeSelection.Ranker;
import weka.attributeSelection.ReliefFAttributeEval;
import weka.attributeSelection.SVMAttributeEval;
import weka.attributeSelection.SymmetricalUncertAttributeSetEval;
import weka.attributeSelection.WrapperSubsetEval;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.AveragedNDependenceEstimators.A1DE;
import weka.classifiers.bayes.AveragedNDependenceEstimators.A2DE;
import weka.classifiers.bayes.BayesNet;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.bayes.net.search.local.TAN;
import weka.classifiers.functions.LibSVM;
import weka.classifiers.functions.SimpleLogistic;
import weka.classifiers.lazy.IBk;
import weka.classifiers.meta.AdaBoostM1;
import weka.classifiers.meta.AttributeSelectedClassifier;
import weka.classifiers.meta.Bagging;
import weka.classifiers.meta.FilteredClassifier;
import weka.classifiers.sklearn.ScikitLearnClassifier;
import weka.classifiers.trees.DecisionStump;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.LMT;
import weka.classifiers.trees.REPTree;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;
import weka.filters.supervised.attribute.Discretize;

public class singleExperiment {
    
    static Random random;
    static int folds;

    public static void main(String[] args) throws Exception {
        // Reading arguments
        int index = Integer.parseInt(args[0]);
        String paramsFile = args[1];

        // Reading parameters
        String[] params = null;

        try (BufferedReader br = new BufferedReader(new FileReader(paramsFile))) {
            // Reading index line
            String line;
            for (int i = 0; i < index; i++) {
                br.readLine();
            }
            line = br.readLine();

            // Printing params
            System.out.println(line);

            //Splitting params
            params = line.split(" ");
        } catch (FileNotFoundException e) {
            System.out.println(e);
        }

        // Getting params from line: bbdd, algorithm, seed, folds, discretized, 
        // nTrees, featureSelection, baseClas, (n, ensemble, boosting, RF, bagSize)
        String bbdd = params[0];
        
        String alg = params[1];

        int seed = Integer.parseInt(params[2]);
        random = new Random(seed);  
        
        folds = Integer.parseInt(params[3]);
      
        boolean discretized = Boolean.parseBoolean(params[4]);
        
        int nTrees = Integer.parseInt(params[5]); 
        
        String featureSelection = params[6];

        // Read data
        Instances data = readData(bbdd);
        
        // Leave-one-out Cross Validation
        folds = data.numInstances();
        
        // Parallel execution as posible, and seed
        String[] options = new String[4];
        options[0] = "-num-slots";
        options[1] = "0";
        options[2] = "-S";
        options[3] = ""+seed;
        
        // Launch the algorithm specified
        AbstractClassifier clas;
        
        String[] optionsSkL = new String[2];
        optionsSkL[0] = "-learner";
        String optionsLearner;

        switch (alg) {
            // Bayes algorithms
            case "mAnDE":
                clas = new mAnDE();
                ((mAnDE)clas).setBaseClass(params[7]);
                ((mAnDE)clas).setN(Integer.parseInt(params[8]));
                ((mAnDE)clas).setEnsemble(params[9]);
                ((mAnDE)clas).setBagSize(Double.parseDouble(params[10]));
                
                ((mAnDE)clas).setnTrees(nTrees);
                break;
                
            case "NB":
                clas = new NaiveBayes();
                break;
  
            case "A1DE":
                clas = new A1DE();
                break;
            
            case "A2DE":
                clas = new A2DE();
                break;
                
            case "TAN":
                clas = new BayesNet();
                TAN algorithm = new TAN();
                ((BayesNet)clas).setSearchAlgorithm(algorithm);
                break;
            
            // Tree algorithms
            case "J48":
                clas = new J48();
                break;
                
            case "REPTree":
                clas = new REPTree();
                break;
                
            // Ensemble algorithms
            case "Bagging":
                clas = new Bagging();
                ((Bagging)clas).setOptions(options);
                
                switch (params[7]) {
                    case "REPTree":
                        ((Bagging)clas).setClassifier(new REPTree());
                        break;
                    case "Stump":
                        ((Bagging)clas).setClassifier(new DecisionStump());
                        break;
                    default:
                        ((Bagging)clas).setClassifier(new J48());
                        break;
                }

                ((Bagging)clas).setNumIterations(nTrees);
                break;

            case "RandomForest":
                clas = new RandomForest();
                ((RandomForest)clas).setOptions(options);
                
                ((RandomForest)clas).setNumIterations(nTrees);
                break;
                
            case "AdaBoost":
                clas = new AdaBoostM1();
                ((AdaBoostM1)clas).setOptions(options);
                
                switch (params[7]) {
                    case "REPTree":
                        ((AdaBoostM1)clas).setClassifier(new REPTree());
                        break;
                    case "J48":
                        ((AdaBoostM1)clas).setClassifier(new J48());
                        break;
                    default:
                        ((AdaBoostM1)clas).setClassifier(new DecisionStump());
                        break;
                }
                
                ((AdaBoostM1)clas).setNumIterations(nTrees);
                break;
                
            case "LMT":
                clas = new LMT();
                break;
                
            // Linear regreesion algorithms
            case "SVM":
                clas = new LibSVM();
                break;
                
            case "LogitBoost":
                clas = new SimpleLogistic();
                ((SimpleLogistic)clas).setNumBoostingIterations(nTrees);
                break;   
                
            // Scikit-learn algorithms
            case "XGB":
                clas = new ScikitLearnClassifier();
                optionsSkL[1] = "XGBClassifier";
                ((ScikitLearnClassifier)clas).setOptions(optionsSkL);
                break;  
                
            case "GradientBoosting":
                clas = new ScikitLearnClassifier();
                optionsSkL[1] = "GradientBoostingClassifier";
                ((ScikitLearnClassifier)clas).setOptions(optionsSkL);
                
                optionsLearner = "n_estimators=" + nTrees + ", random_state=" + random;
                ((ScikitLearnClassifier)clas).setLearnerOpts(optionsLearner);
                break;  
                
            case "ExtraTree":
                clas = new ScikitLearnClassifier();
                optionsSkL[1] = "ExtraTreeClassifier";
                ((ScikitLearnClassifier)clas).setOptions(optionsSkL);
                
                optionsLearner = "random_state=" + random;
                ((ScikitLearnClassifier)clas).setLearnerOpts(optionsLearner);
                break;  
                
            case "ExtraTrees":
                clas = new ScikitLearnClassifier();
                optionsSkL[1] = "ExtraTreesClassifier";
                ((ScikitLearnClassifier)clas).setOptions(optionsSkL);
                
                optionsLearner = "n_estimators=" + nTrees + 
                        ", n_jobs=-1" + ", random_state=" + random;
                ((ScikitLearnClassifier)clas).setLearnerOpts(optionsLearner);
                break;  

            case "Ridge":
                clas = new ScikitLearnClassifier();
                optionsSkL[1] = "RidgeClassifier";
                ((ScikitLearnClassifier)clas).setOptions(optionsSkL);
                
                optionsLearner = "random_state=" + random;
                ((ScikitLearnClassifier)clas).setLearnerOpts(optionsLearner);
                break;  
                
            case "GaussianNB":
                clas = new ScikitLearnClassifier();
                optionsSkL[1] = "GaussianNB";
                ((ScikitLearnClassifier)clas).setOptions(optionsSkL);
                break;  
                
            case "MultinomialNB":
                clas = new ScikitLearnClassifier();
                optionsSkL[1] = "MultinomialNB";
                ((ScikitLearnClassifier)clas).setOptions(optionsSkL);
                break; 
                
            case "LDA":
                clas = new ScikitLearnClassifier();
                optionsSkL[1] = "LDA";
                ((ScikitLearnClassifier)clas).setOptions(optionsSkL);
                break; 
                
            

            // Other algorithms
            case "KNN":
                clas = new IBk();
                break;
                
            default:
                throw new Exception("Algorithm not mAnDE, A1DE, A2DE, J48, REPTree, "
                    + "Bagging, RandomForest, AdaBoost, "
                    + "NB, TAN or SVM...\n value of alg: " + alg);
        }
        
        // Create a pipeline that first discretize the data, and then execute the algorithm
        if (discretized) {
            FilteredClassifier fc = new FilteredClassifier();
            Discretize discretizer = new Discretize();
            
            fc.setFilter(discretizer);
            fc.setClassifier(clas);
            
            clas = fc;
        }
        
        AttributeSelectedClassifier asc = null;
        ASEvaluation evaluator = null;
        ASSearch search = null;
        
        switch (featureSelection) {
            case "CFS": // Va lento
                asc = new AttributeSelectedClassifier();
                evaluator = new CfsSubsetEval();
                search = new GreedyStepwise();
                break;
            
            case "FCBF":
                asc = new AttributeSelectedClassifier();
                evaluator = new SymmetricalUncertAttributeSetEval();
                search = new FCBFSearch();
                break;
            
            case "IWSS": // Pablo Bermejo, va lento
                asc = new AttributeSelectedClassifier();
                evaluator = new WrapperSubsetEval();
                search = new IWSS();
                break;
                
            case "IWSS_NB": // Pablo Bermejo
                asc = new AttributeSelectedClassifier();
                evaluator = new WrapperSubsetEval();
                search = new IWSSembeddedNB();
                break;
                
            // INTERACT (no lo encuentro en WEKA)
                
            case "InfoGain":
                asc = new AttributeSelectedClassifier();
                evaluator = new InfoGainAttributeEval();
                search = new Ranker();
                break;
                
            case "GainRatio":
                asc = new AttributeSelectedClassifier();
                evaluator = new GainRatioAttributeEval();
                search = new Ranker();
                break;
                
            case "ReliefF":
                asc = new AttributeSelectedClassifier();
                evaluator = new ReliefFAttributeEval();
                search = new Ranker();
                break;
                
            //mRMR (no está en WEKA)

            case "SVM-RFE": // Va lento
                asc = new AttributeSelectedClassifier();
                evaluator = new SVMAttributeEval();
                search = new Ranker();
                break;
        }
        
        if (asc != null) {
            asc.setClassifier(clas);
            asc.setEvaluator(evaluator);
            asc.setSearch(search);
            
            // Si es Ranked, limitar el número de variables a escoger
            if (search instanceof Ranker) {
                String[] optionsFS = new String[2];
                optionsFS[0] = "-N";
                optionsFS[1] = "50";
                search.setOptions(optionsFS);
            }   
            clas = asc;
        }
        
        launchExperiment(alg, clas, data, args, params);
    }
    
    public static Instances readData(String bbdd) throws Exception {
        // Read bbdd
        //ConverterUtils.DataSource loader = new ConverterUtils.DataSource("/res/bbdd/" + bbdd + ".arff");
        ConverterUtils.DataSource loader = new ConverterUtils.DataSource("res/bbdd/" + bbdd);

        Instances data = loader.getDataSet();
        data.setClassIndex(data.numAttributes()-1);
        
        // Shuffle data
        data.randomize(random);
        
        return data;
    }
    
    public static void launchExperiment(String name, AbstractClassifier clas, Instances data, String[] args, String[] params) throws Exception {
        String EXPERIMENTS_FOLDER = "results/";
        String savePath;
        
        // Set name of the save file
        savePath = EXPERIMENTS_FOLDER  + "experiment_results_" 
                + params[0] + "_" + params[1] + "_" + params[2] 
                + "_" + params[3] + "_" + params[4] + "_" + params[5] 
                + "_" + params[6] + "_" + params[7] + "_" + params [8]
                + "_" + params[9] + "_" + params[10] + ".csv";

        File file = new File(savePath);
        
        if(file.length() == 0) {
            System.out.println("   ---   " + name + "   ---   ");
            double init = System.currentTimeMillis();

            Evaluation evaluation = new Evaluation(data);
            evaluation.crossValidateModel(clas, data, folds, random, new Object[]{});

            double time = ((System.currentTimeMillis() - init) / folds) / 1000;

            System.out.println(evaluation.toSummaryString());
            System.out.println(evaluation.toMatrixString());

            double fm = 0;
            double precision = 0;
            double recall = 0;
            for (int i = 0; i < data.classAttribute().numValues(); i++) {
                System.out.println(i + " -> " + evaluation.fMeasure(i));
                if (!Double.isNaN(evaluation.fMeasure(i))) {
                    fm += evaluation.fMeasure(i);
                } if (!Double.isNaN(evaluation.precision(i))) {
                    precision += evaluation.precision(i);
                } if (!Double.isNaN(evaluation.recall(i))) {
                    recall += evaluation.recall(i);
                }
            }
            fm = fm / data.classAttribute().numValues();
            precision = precision / data.classAttribute().numValues();
            recall = recall / data.classAttribute().numValues();

            System.out.println("\nF-Score -> " + fm + "\n");
            System.out.println("Execution time: " + time + "\n");

            BufferedWriter csvWriter = new BufferedWriter(new FileWriter(savePath, true));
            
            String header = "bbdd,algorithm,seed,folds,discretized,nTrees,featureSelection,baseClass,n,ensemble,boosting,RF,bagSize,score,fm,precision,recall,time(s)\n";
            csvWriter.append(header);
            
            String output = params[0] + "," + params[1] + "," + params[2] + ","
                    + folds + "," + params[4] + "," + params[5] + ","
                    + params[6] + "," + params[7] + "," + params[8] + ","
                    + params[9] + "," + params[10] + ","
                    + evaluation.pctCorrect() + "," + fm + "," + precision + ","
                    + recall + "," + time + "\n";
            csvWriter.append(output);

            System.out.println("Results saved at: " + savePath);
            
            csvWriter.flush();
            csvWriter.close();
        } else{
            System.out.println("results already existing in " + savePath);
        }
    }
}
