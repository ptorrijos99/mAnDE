/**
 *
 * @author Pablo Torrijos Arenas
 */
package org.albacete.simd.mAnDE;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.HashMap;
import java.util.List;
import java.util.ArrayList;
import java.util.Enumeration;
import java.util.HashSet;
import java.util.Vector;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.REPTree;
import weka.core.Capabilities;
import weka.core.Drawable;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.OptionHandler;
import weka.core.Utils;

import static weka.classifiers.AbstractClassifier.runClassifier;
import weka.classifiers.bayes.BayesNet;
import weka.core.Option;

public class mAnDE extends AbstractClassifier implements
        OptionHandler {

    /**
     * For serialisation.
     */
    private static final long serialVersionUID = 3545430914549890589L;

    // Auxiliary variables //
    /**
     * Instances.
     */
    protected static Instances data;

    /**
     * The discretisation filter.
     */
    protected weka.filters.supervised.attribute.Discretize discretizer = null;

    /**
     * Discretisation filter for "discretize4".
     */
    protected Discretize discretizerNS = null;

    /**
     * HashMap containing the mSPnDEs.
     */
    private HashMap<Integer, Object> mSPnDEs;

    /**
     * HashMap to convert from variable name to index.
     */
    public static HashMap<String, Integer> nToI;

    /**
     * HashMap to convert from variable index to name.
     */
    public static HashMap<Integer, String> iToN;

    /**
     * Number of values per variable.
     */
    public static int[] varNumValues;

    /**
     * Number of values of the class.
     */
    public static int classNumValues;

    /**
     * Index of the class.
     */
    public static int y;

    /**
     * Number of instances.
     */
    public static int numInstances;

    /**
     * Variable to check if any value is zero.
     */
    private static int[] zeros;

    /**
     * Naive Bayes for NB mode.
     */
    private static NaiveBayes nb;

    /**
     * Indicates whether Naive Bayes mode is enabled.
     */
    private boolean modeNB = false;

    /**
     * Indicates whether discretised.
     */
    private boolean discretized = false;

    // mAnDE parameters //
    /**
     * Performs Chow-Liu trees instead of decision trees.
     */
    private boolean chowLiu = false;

    /**
     * Performs REPTree trees instead of J48.
     */
    private boolean repTree = false;

    /**
     * Prune decision trees.
     */
    private boolean pruning = true;

    /**
     * Complete mAnDE with mSPnDEs that have not been added.
     */
    private boolean complete = false;

    /**
     * Execute a tree, and for each variable in the tree a new tree in which that variable acts as a class.
     */
    private boolean variousTrees = false;

    /**
     * Runs a set of decision trees instead of a single tree.
     */
    private boolean ensemble = false;

    /**
     * Runs a Random Forest instead of Bagging.
     */
    private boolean randomForest = true;

    /**
     * Percentage of instances to be used to make each tree in the assembly.
     */
    private double bagSize = 100;

    /**
     * n of the mAnDE.
     */
    private int n = 1;

    /**
     * Discretize before executing the tree.
     */
    private boolean discretizeBefore = true;

    /**
     * Discretize non-discretized variables into 4 intervals.
     */
    private boolean discretize4 = false;

    /**
     * Uses majority instead of sum of probabilities.
     */
    private boolean majority = false;

    /**
     * Create the structure of the classifier taking into account the established parameters.
     *
     * @param instances Instances to classify.
     * @throws java.lang.Exception
     */
    @Override
    public void buildClassifier(Instances instances) throws Exception {
        // Can the classifier work with this data?
        getCapabilities().testWithFail(instances);

        // Delete instances with no class
        instances.deleteWithMissingClass();

        // We check if we need to discretise now or later.
        if (isDiscretizeBefore()) {
            data = new Instances(instances);
            discretize(instances);
            // Free up the data space by parameter
            instances.delete();
        } else {
            data = instances;
        }

        // We initialize the index dictionaries
        initializeNameToIndex();

        // Create the mSPnDE's
        try {
            build_mSPnDEs();
        } catch (Exception ex) {
        }

        // If we have not created mSPnDE's
        if (mSPnDEs.isEmpty()) {
            boolean starts1 = false;
            if (!isEnsemble()) {
                starts1 = true;
            }
            while (true) {
                if (!isEnsemble() && !starts1) {
                    // If we cannot execute at all, we run Naive Bayes.
                    System.out.println("CAN'T RUN. WE RUN NAIVE BAYES.");
                    modeNB = true;
                    nb = new NaiveBayes();
                    nb.buildClassifier(data);
                    break;
                }
                if (!isEnsemble()) {
                    setEnsemble(true);
                    setRandomForest(true);
                } else if (isEnsemble()) {
                    bagSize *= 5;
                    if (getBagSize() > 100) {
                        if (isRandomForest()) {
                            setRandomForest(false);
                            bagSize = 100;
                        } else {
                            setEnsemble(false);
                            starts1 = false;
                        }
                    }
                }
                try {
                    build_mSPnDEs();
                } catch (Exception ex) {
                }

                // If we have created mSPnDEs, we end up with
                if (!mSPnDEs.isEmpty()) {
                    break;
                }
            }
        }
        System.out.println("  mSPnDEs created");

        // We discretise if we have not done so at the beginning.
        if (!isDiscretizeBefore()) {
            discretize(instances);
            // Free the data space per parameter
            instances.delete();
        }

        // If we have not run Naive Bayes, we calculate the mAnDE tables.
        if (!modeNB) {
            // Define global variables
            y = data.classIndex();
            classNumValues = data.classAttribute().numValues();
            varNumValues = new int[data.numAttributes()];
            for (int i = 0; i < varNumValues.length; i++) {
                varNumValues[i] = data.attribute(i).numValues();
            }
            numInstances = data.numInstances();

            calculateTables_mSPnDEs();
        }

        // We free up the discretised data space
        data.delete();

        nSPnDEs_variables();
    }

    /**
     * Calculates the probabilities of class membership for the provided Test Instance.
     *
     * @param instance Instance to classify.
     * @return Probability distribution of predicted class membership.
     * @throws java.lang.Exception
     */
    @Override
    public double[] distributionForInstance(Instance instance) throws Exception {
        double[] res = new double[classNumValues];

        final Instance instance_d;
        // If we have discretised any instances with the Unsupervised discretizer
        if (isDiscretize4() && isDiscretized() && (zeros != null)) {
            discretizerNS.input(instance);
            discretizer.input(discretizerNS.output());
            instance_d = discretizer.output();
        } else {
            discretizer.input(instance);
            instance_d = discretizer.output();
        }

        if (modeNB) {
            return nb.distributionForInstance(instance_d);
        }

        // Add up all the probabilities of the mSPnDEs
        mSPnDEs.forEach((id, spode) -> {
            double[] temp = new double[0];
            if (getN() == 1) {
                temp = ((mSP1DE) spode).probsForInstance(instance_d);
            } else if (getN() == 2) {
                temp = ((mSP2DE) spode).probsForInstance(instance_d);
            }

            if (isMajority()) {
                res[Utils.maxIndex(temp)]++;
            } else {
                for (int i = 0; i < res.length; i++) {
                    res[i] += temp[i];
                }
            }
        });

        /* Normalize the result. If the sum is 0 (Utils.normalize will return 
         * an IllegalArgumentException), we set the same value in each 
         * possible value of the class.
         */
        try {
            Utils.normalize(res);
        } catch (IllegalArgumentException ex) {
            for (int i = 0; i < res.length; i++) {
                res[i] = 1.0 / (varNumValues.length - 1);
            }
        }

        return res;
    }

    /**
     * Function that performs the discretisation, checking the "discretize4" parameter to perform the discretisation or not.
     *
     * @param instances Instances to be discretised
     * * @throws Exception
     */
       
    private void discretize(Instances instances) throws Exception {
        // If we are going to discretise in 4 chunks the variables that stay
        // integers in an interval.
        if (isDiscretize4()) {
            // Discretize instances if necessary.
            discretizer = new weka.filters.supervised.attribute.Discretize();
            discretizer.setInputFormat(instances);
            instances = weka.filters.Filter.useFilter(instances, discretizer);

            // We add the variables with only one interval.
            ArrayList<Integer> temp = new ArrayList();
            for (int i = 0; i < instances.numAttributes(); i++) {
                if (instances.attribute(i).numValues() == 1) {
                    temp.add(i);
                }
            }

            // If there is, we have to discretise by equal width.
            if (!temp.isEmpty()) {
                zeros = new int[temp.size()];
                for (int i = 0; i < zeros.length; i++) {
                    zeros[i] = temp.get(i);
                }

                // First we discretise the variables by equal width (4 bins).
                discretizerNS = new Discretize();
                discretizerNS.setBins(4);
                discretizerNS.setInputFormat(data);
                discretizerNS.setAttributeIndicesArray(zeros);
                data = weka.filters.Filter.useFilter(data, discretizerNS);
                setDiscretized(true);

                // And then the rest of the variables
                discretizer = new weka.filters.supervised.attribute.Discretize();
                discretizer.setInputFormat(data);
                data = weka.filters.Filter.useFilter(data, discretizer);
            } else {
                // Otherwise, the discretization is "instances" directly
                data = new Instances(instances);
            }
        } // If we're only going to discretize once
        else {
            discretizer = new weka.filters.supervised.attribute.Discretize();
            discretizer.setInputFormat(data);
            data = weka.filters.Filter.useFilter(data, discretizer);
        }
    }

     /**
     * Create the necessary mSPnDE's, by running the trees set in the options...
     */
    private void build_mSPnDEs() throws Exception {
        mSPnDEs = new HashMap<>();

        if (chowLiu) {
            useChowLiu();
        } else {
            if (variousTrees) {
                variousTrees();
            } else {
                Classifier[] trees;

                J48 j48 = new J48();
                Bagging bagging;
                REPTree repT = new REPTree();

                // We define whether we want pruning or not.
                if (!pruning) {
                    j48.setUnpruned(true);
                    repT.setNoPruning(true);
                }

                String[] options = new String[2];
                options[0] = "-num-slots";
                options[1] = "0";

                if (ensemble) {
                    if (randomForest) {
                        System.out.println("Run Random Forest");
                        RandomForest rf = new RandomForest();
                        // Set the number of parallel wires to 0 (automatic)
                        rf.setOptions(options);
                        rf.setNumIterations(10);
                        rf.setBagSizePercent(bagSize);
                        rf.buildClassifier(data);
                        trees = rf.getClassifiers();
                        for (Classifier tree : trees) {
                            graphToSPnDE(treeParser(tree));
                        }
                    } else {
                        System.out.println("Ejecute Bagging");
                        bagging = new Bagging();
                        if (repTree) {
                            bagging.setClassifier(repT);
                        } else {
                            bagging.setClassifier(j48);
                        }
                        // Set the number of parallel wires to 0 (automatic)
                        bagging.setOptions(options);
                        bagging.setNumIterations(10);
                        bagging.setBagSizePercent(bagSize);
                        bagging.buildClassifier(data);
                        trees = bagging.getClassifiers();
                        for (Classifier tree : trees) {
                            graphToSPnDE(treeParser(tree));
                        }
                    }
                } else {
                    if (repTree) {
                        repT.buildClassifier(data);
                        graphToSPnDE(treeParser(repT));
                    } else {
                        j48.buildClassifier(data);
                        graphToSPnDE(treeParser(j48));
                    }
                }
            }

            // If we want to add mSP1DE's with variables not in the tree
            if (isComplete()) {
                if (getN() == 1) {
                    completeAll_mSP1DEs();
                } else {
                    System.out.println("Complete can only be used with n = 1.");
                }
            }
        }
    }

    /**
     * Reads the classifier passed by parameter and returns a HashMap<String,Node> with the classifier data.
     *
     * @param classifier Classifier to be parsed
     */
    private HashMap<String, Node> treeParser(Classifier clasificador) {
        String[] lines = new String[0];
        try {
            lines = ((Drawable) clasificador).graph().split("\r\n|\r|\n");
        } catch (Exception ex) {
        }

        HashMap<String, Node> nodes = new HashMap();
        int init = -1, fin = -1;

        for (int i = 0; i < lines.length; i++) {
            if (lines[i].contains(" [label")) {
                init = i;
                break;
            }
        }

        for (int i = 0; i < lines.length; i++) {
            if (lines[i].contains(" [label")) {
                fin = i;
            }
        }

        if (init != -1) {
            // We add the father
            //N0 [label="petallength" ]             J48
            //N58640b4a [label="1: petalwidth"]     RandomTree
            String id = lines[init].substring(0, lines[init].indexOf(" [label"));
            String name = "", finLabel;
            if (lines[init].contains(": ")) {
                // RandomTree and REPTree
                try {
                    name = lines[init].substring(lines[init].indexOf(": ") + 2, lines[init].indexOf("\"]"));
                } catch (Exception ex) {
                }
            } else {
                // J48
                try {
                    name = lines[init].substring(lines[init].indexOf("=\"") + 2, lines[init].indexOf("\" ]"));
                } catch (Exception ex) {
                }
            }

            if (!"".equals(name)) {
                if (!nToI.containsKey(name)) {
                    name = name.replace("\\", "");
                }
                Node father = new Node(id, name);
                nodes.put(id, father);
            }

            for (int i = init + 1; i < fin; i++) {
                //N0->N1 [label="= \'(-inf-2.6]\'"]                     J48
                //N529c603a->N6f5883ef [label=" = \'(5.45-5.75]\'"]     RandomTree and REPTree
                if (lines[i].contains("->")) {
                    String id1 = lines[i].substring(0, lines[i].indexOf("->"));
                    String id2 = lines[i].substring(lines[i].indexOf("->") + 2, lines[i].indexOf(" [label"));

                    if (nodes.get(id2) == null) {
                        nodes.put(id2, new Node(id2, nodes.get(id1)));
                    }
                    nodes.get(id1).addChild(nodes.get(id2));
                } 
                //N0 [label="petallength" ]                J48
                //N58640b4a [label="1: petalwidth"]        RandomTree and REPTree
                else if (!lines[i].contains(" (")) {
                    if (lines[i].contains("\" ]")) {
                        finLabel = "\" ]";
                    } else {
                        finLabel = "\"]";
                    }

                    try {
                        id = lines[i].substring(0, lines[i].indexOf(" [label"));
                    } catch (Exception ex) {
                        System.out.println(lines[i]);
                    }

                    if (lines[i].contains(" : ")) {
                        // RandomTree and REPTree
                        name = lines[i].substring(lines[i].indexOf(" : ") + 3, lines[i].indexOf(finLabel));
                    } else if (lines[i].contains(": ")) {
                        // RandomTree and REPTree
                        name = lines[i].substring(lines[i].indexOf(": ") + 2, lines[i].indexOf(finLabel));
                    } else {
                        // J48
                        name = lines[i].substring(lines[i].indexOf("=\"") + 2, lines[i].indexOf(finLabel));
                    }

                    if (!nToI.containsKey(name)) {
                        name = name.replace("\\", "");
                    }
                    nodes.get(id).setName(name);
                }
            }
        }
        return nodes;
    }

    /**
     * Converts a HashMap<String,Node> to a representation of mAnDE.
     */
    private void graphToSPnDE(HashMap<String, Node> nodes) {
        if (getN() == 1) {
            nodes.values().forEach((nodo) -> {
                nodo.getChildren().values().forEach((child) -> {
                    if (!nodo.getName().equals("") && !child.getName().equals("")) {
                        toSP1DE(nodo.getName(), child.getName());
                    }
                });
            });
        } else if (getN() == 2) {
            nodes.values().forEach((nodo) -> {
                nodo.getChildren().values().forEach((child) -> {
                    if (!nodo.getName().equals("") && !child.getName().equals("")) {
                        toSP2DE(nodo.getName(), child.getName(), nodo.getParent().getName(),
                                nodo.getChildrenArray(child.getName()), child.getChildrenArray());
                    }
                });
            });
        }

    }

    /**
     * Create an mSP1DE with the variable 'parent' (if it doesn't already exist), and add the variable 'child' as a dependency, and vice versa.
     *
     * @param parent Name of the parent in the mSP1DE.
     * @param child Name of the child in the mSP1DE.
     */
    private void toSP1DE(String parent, String child) {
        if (!parent.equals(child)) {
            if (!mSPnDEs.containsKey(parent.hashCode())) {
                mSPnDEs.put(parent.hashCode(), new mSP1DE(parent));
            }
            if (!child.equals("")) {
                if (!mSPnDEs.containsKey(child.hashCode())) {
                    mSPnDEs.put(child.hashCode(), new mSP1DE(child));
                }
                try {
                    ((mSP1DE) mSPnDEs.get(parent.hashCode())).moreChildren(child);
                    ((mSP1DE) mSPnDEs.get(child.hashCode())).moreChildren(parent);
                } catch (NullPointerException ex) {
                }
            }
        }
    }

    /**
     * Create an mSP2DE with the variable 'parent' (if it doesn't already exist), and add the variable 'child' as a dependency, and vice versa.
     *
     * @param parent Name of the parent in the mSP2DE.
     * @param child Name of the child in the mSP2DE.
     * @param grandparent Name of the parent of the parent in the mSP2DE.
     * @param brothers Name of the other children of the father in the mSP2DE.
     * @param grandchildren Name of the children of the child in the mSP2DE.
     */
    private void toSP2DE(String parent, String child, String grandparent,
            ArrayList<String> brothers, ArrayList<String> grandchildren) {
        if (!parent.equals(child)) {
            if (!mSPnDEs.containsKey(parent.hashCode() + child.hashCode())) {
                mSPnDEs.put(parent.hashCode() + child.hashCode(), new mSP2DE(parent, child));
            }
            try {
                mSP2DE elem = ((mSP2DE) mSPnDEs.get(parent.hashCode() + child.hashCode()));
                if (!grandparent.equals(parent)) {
                    elem.moreChildren(grandparent);
                }
                elem.moreChildren(brothers);
                elem.moreChildren(grandchildren);
            } catch (NullPointerException ex) {
            }
        }
    }

    /**
     * Executes in parallel the 'creaTabla()' functions of each mSPnDE, 
     * and terminates when all have executed it.
     */
    private void calculateTables_mSPnDEs() {
        List<Object> list = new ArrayList<>(mSPnDEs.values());

         //Create the thread pool, one for each mSPnDEs
        ExecutorService executor = Executors.newFixedThreadPool(mSPnDEs.size());

        //Calls the mSP1DE function that creates the table for each mSP1DE
        list.forEach((spode) -> {
            executor.execute(() -> {
                if (getN() == 1) {
                    ((mSP1DE) spode).buildTables();
                } else if (getN() == 2) {
                    ((mSP2DE) spode).buildTables();
                }
            });
        });

        //Stop supporting calls 
        try {
            //Wait until they are all finished (at T days stop)
            executor.awaitTermination(Long.MAX_VALUE, TimeUnit.DAYS);
        } catch (InterruptedException ex) {
        }
    }

    /**
     * Method of execution using Chow-Liu trees instead of decision trees.
     *
     * @throws Exception
     */
    private void useChowLiu() throws Exception {
        // We create the TAN classifier, which creates an extended Chow-Liu tree.
        BayesNet net = new BayesNet();
        weka.classifiers.bayes.net.search.local.TAN tan = new weka.classifiers.bayes.net.search.local.TAN();
        net.setSearchAlgorithm(tan);
        net.buildClassifier(data);

        // Parse the generated tree
        String[] lines = new String[0];
        try {
            lines = net.toString().split("\r\n|\r|\n");
        } catch (Exception ex) {
        }

        HashMap<String, Node> nodes = new HashMap();

        for (int i = 4; i < lines.length; i++) {
            //att_1(1): class att_15   TAN
            String[] actual = lines[i].split(" ");
            if (!"LogScore".equals(actual[0])) {
                // If 3, a node has another node as a parent and class
                if (actual.length == 3) {
                    // We remove the excess part of the first name.
                    actual[0] = actual[0].substring(0, lines[i].indexOf("("));

                    if (nodes.get(actual[2]) == null) {
                        nodes.put(actual[2], new Node(actual[2], actual[2]));
                    }
                    if (nodes.get(actual[0]) == null) {
                        nodes.put(actual[0], new Node(actual[0], actual[0]));
                    }
                    nodes.get(actual[0]).setParent(nodes.get(actual[2]));
                    nodes.get(actual[2]).addChild(nodes.get(actual[0]));
                }
            }
        }

        graphToSPnDE(nodes);
    }

    /**
     * Method that changes the execution, executing a tree, and for the 
     * variables that appear in it, another tree in which these behave 
     * like the class.
     */
    private void variousTrees() {
        HashSet<String> variables = new HashSet();
        J48 j48 = new J48();
        try {
            j48.buildClassifier(data);
        } catch (Exception ex) {
        }

        // We get the name of the variables that appear in the tree
        treeParser(j48).values().forEach((var) -> {
            if (!"".equals(var.getName())) {
                variables.add(var.getName());
            }
        });

        int classIndex = data.classIndex();

        // For each one, let's run the tree
        variables.forEach((var) -> {
            J48 tree = new J48();
            data.setClassIndex(nToI.get(var));
            try {
                tree.buildClassifier(data);
            } catch (Exception ex) {
            }

            HashSet<String> newVariables = new HashSet();
            // We get the name of the variables
            treeParser(tree).values().forEach((var2) -> {
                if (!"".equals(var2.getName())) {
                    newVariables.add(var2.getName());
                }
            });

            // If the class is inside the created tree
            if (newVariables.contains(iToN.get(classIndex))) {
                if (!mSPnDEs.containsKey(var.hashCode())) {
                    mSPnDEs.put(var.hashCode(), new mSP1DE(var));
                }
                newVariables.forEach((temp) -> {
                    if (!temp.equals(classIndex)) {
                        ((mSP1DE) mSPnDEs.get(var.hashCode())).moreChildren(temp);
                    }
                });
            }
        });

        data.setClassIndex(classIndex);
    }

    /**
     * Add the mSP1DE's that have not been created after parsing the trees. Each of them will have no children, so their probability will be P(Xi|y).
     */
    private void completeAll_mSP1DEs() {
        for (int i = 0; i < data.numAttributes(); i++) {
            if (!mSPnDEs.containsKey(iToN.get(i).hashCode())) {
                mSPnDEs.put(iToN.get(i).hashCode(), new mSP1DE(iToN.get(i)));
            }
        }
    }

    /**
     * Initialise HashMaps to convert indexes to names.
     */
    private void initializeNameToIndex() {
        nToI = new HashMap<>();
        iToN = new HashMap<>();
        for (int i = 0; i < data.numAttributes(); i++) {
            nToI.put(data.attribute(i).name(), i);
            iToN.put(i, data.attribute(i).name());
        }
    }

    /**
     * Returns the number of nSPnDEs and Variables per nSPnDE.
     *
     * @return The number of nSPnDEs and Variables per nSPnDE.
     * @throws java.io.IOException
     */
    public double[] nSPnDEs_variables() throws IOException {
        double[] res = new double[2];
        res[0] = mSPnDEs.size();
        res[1] = 0;
        mSPnDEs.values().forEach((spode) -> {
            if (getN() == 1) {
                res[1] += (((mSP1DE) spode).getNChildren() / res[0]);
            } else if (getN() == 2) {
                res[1] += (((mSP2DE) spode).getNChildren() / res[0]);
            }
        });

        File f = new File("temp.txt");
        FileWriter file = new FileWriter(f, true);
        PrintWriter pw = new PrintWriter(file, true);
        pw.println(res[0] + "," + res[1]);
        file.close();

        return res;
    }

    /**
     * @param chowLiu The chowLiu hyperparameter to be set.
     */
    public void setChowLiu(boolean chowLiu) {
        this.chowLiu = chowLiu;
    }

    /**
     * @param pruning The pruning hyperparameter to set.
     */
    public void setPruning(boolean pruning) {
        this.pruning = pruning;
    }

    /**
     * @param complete The complete hyperparameter to be set.
     */
    public void setComplete(boolean complete) {
        this.complete = complete;
    }

    /**
     * @param bagSize The bagSize hyperparameter to be set
     */
    public void setBagSize(double bagSize) {
        this.bagSize = bagSize;
    }

    /**
     * @param discretizeBefore The discretizeBefore hyperparameter to be set
     */
    public void setDiscretizeBefore(boolean discretizeBefore) {
        this.discretizeBefore = discretizeBefore;
    }

    /**
     * @param discretize4 The discretize4 hyperparameter to be set
     */
    public void setDiscretize4(boolean discretize4) {
        this.discretize4 = discretize4;
    }

    /**
     * @param n The n hyperparameter to be set
     */
    public void setN(int n) {
        if (n > 0 && n < 3) {
            this.n = n;
        }
    }

    /**
     * @param majority The majority hyperparameter to be set
     */
    public void setMajority(boolean majority) {
        this.majority = majority;
    }

    /**
     * @param repTree The repTree hyperparameter to be set
     */
    public void setRepTree(boolean repTree) {
        this.repTree = repTree;
    }

    /**
     * @param variousTrees The variousTrees hyperparameter to be set
     */
    public void setVariousTrees(boolean variousTrees) {
        this.variousTrees = variousTrees;
    }

    /**
     * @param ensemble The ensemble hyperparameter to be set
     */
    public void setEnsemble(boolean ensemble) {
        this.ensemble = ensemble;
    }

    /**
     * @param randomForest The randomForest to be set
     */
    public void setRandomForest(boolean randomForest) {
        this.randomForest = randomForest;
    }

    /**
     * @param discretized The discretized to be set
     */
    public void setDiscretized(boolean discretized) {
        this.discretized = discretized;
    }

    /**
     * @return The discretized
     */
    public boolean isDiscretized() {
        return discretized;
    }

    /**
     * @return The chowLiu
     */
    public boolean isChowLiu() {
        return chowLiu;
    }

    /**
     * @return The repTree
     */
    public boolean isRepTree() {
        return repTree;
    }

    /**
     * @return The pruning
     */
    public boolean isPruning() {
        return pruning;
    }

    /**
     * @return The complete
     */
    public boolean isComplete() {
        return complete;
    }

    /**
     * @return The variousTrees
     */
    public boolean isVariousTrees() {
        return variousTrees;
    }

    /**
     * @return The ensemble
     */
    public boolean isEnsemble() {
        return ensemble;
    }

    /**
     * @return The randomForest
     */
    public boolean isRandomForest() {
        return randomForest;
    }

    /**
     * @return The bagSize
     */
    public double getBagSize() {
        return bagSize;
    }

    /**
     * @return The n
     */
    public int getN() {
        return n;
    }

    /**
     * @return The discretizeBefore
     */
    public boolean isDiscretizeBefore() {
        return discretizeBefore;
    }

    /**
     * @return The discretize4
     */
    public boolean isDiscretize4() {
        return discretize4;
    }

    /**
     * @return The majority
     */
    public boolean isMajority() {
        return majority;
    }

    /**
     * Returns default capabilities of the classifier.
     *
     * @return The capabilities of this classifier.
     */
    @Override
    public Capabilities getCapabilities() {
        Capabilities result = super.getCapabilities();
        result.disableAll();

        // attributes
        result.enable(Capabilities.Capability.NOMINAL_ATTRIBUTES);
        result.enable(Capabilities.Capability.NUMERIC_ATTRIBUTES);
        result.enable(Capabilities.Capability.MISSING_VALUES);

        // class
        result.enable(Capabilities.Capability.NOMINAL_CLASS);
        result.enable(Capabilities.Capability.MISSING_CLASS_VALUES);

        return result;
    }

    /**
     * Returns an enumeration describing the available options
     *
     * @return an enumeration of all the available options
     */
    @Override
    public Enumeration listOptions() {
        Vector newVector = new Vector(12);

        newVector.addElement(new Option("\tn of the mAnDE (1 or 2, default 1)\n", "N", 1, "-N <int>")); 
        newVector.addElement(new Option("\tRealise Chow-Liu trees instead of decision trees", "CH", 0, "-CH"));
        newVector.addElement(new Option("\tUse REPTree trees instead of J48 trees", "REP", 0, "-REP"));
        newVector.addElement(new Option("\tNOT performs pruning of decision trees", "P", 0, "-P"));
        newVector.addElement(new Option("\tCompletes the mSP1DEs that would not have been created", "C", 0, "-C"));
        newVector.addElement(new Option("\tCompletes a first tree, and with the variables that appear in it, one for each one (severalTrees)", "V", 0, "-V"));
        newVector.addElement(new Option("\tRealise an ensemble of decision trees", "E", 0, "-E"));
        newVector.addElement(new Option("\tRealise the ensemble of decision trees using Random Forest", "RF", 0, "-RF"));
        newVector.addElement(new Option("\tSet the number of instances used to create each tree when using ensembles (0, 100]\n", "B", 100,"-B <double>"));
        newVector.addElement(new Option("\tDiscretize after making the decision tree", "D", 0, "-D"));
        newVector.addElement(new Option("\tDiscretize in 4 bins the variables that remain undiscretized", "D4", 0, "-D4"));
        newVector.addElement(new Option("\tUse a majority vote instead of sum of probabilities", "M", 0, "-M"));

        return newVector.elements();
    }

    /**
     * Convert a list of options to the hyperparameters of the algorithm.
     *
     * @param options Options to convert
     * @throws java.lang.Exception
     */
    @Override
    public void setOptions(String[] options) throws Exception {
        String N = Utils.getOption('N', options);
        if (N.length() != 0) {
            n = Integer.parseInt(N);
            if (n < 1 || n > 2) {
                n = 1;
            }
        } else {
            n = 1;
        }

        chowLiu = Utils.getFlag("CH", options);

        repTree = Utils.getFlag("REP", options);

        pruning = !Utils.getFlag("P", options);

        complete = Utils.getFlag("C", options);

        variousTrees = Utils.getFlag("V", options);

        ensemble = Utils.getFlag("E", options);

        randomForest = Utils.getFlag("RF", options);

        String Bag = Utils.getOption('B', options);
        if (Bag.length() != 0) {
            bagSize = Integer.parseInt(Bag);
        } else {
            bagSize = 100;
        }

        discretizeBefore = !Utils.getFlag("D", options);

        discretize4 = Utils.getFlag("D4", options);

        majority = Utils.getFlag("M", options);

        Utils.checkForRemainingOptions(options);

    }

    /**
     * Devuelve las opciones actuales del clasificador.
     *
     * @return Un array de strings que se pueda añadir a setOptions
     */
    @Override
    public String[] getOptions() {
        Vector result = new Vector();

        result.add("-N");
        result.add("" + n);

        if (chowLiu) {
            result.add("-CH");
        }

        if (repTree) {
            result.add("-REP");
        }

        if (!pruning) {
            result.add("-P");
        }

        if (complete) {
            result.add("-C");
        }

        if (variousTrees) {
            result.add("-V");
        }

        if (ensemble) {
            result.add("-E");
        }

        if (randomForest) {
            result.add("-RF");
        }

        result.add("-B");
        result.add("" + bagSize);

        if (!discretizeBefore) {
            result.add("-D");
        }

        if (discretize4) {
            result.add("-D4");
        }

        if (majority) {
            result.add("-M");
        }

        return (String[]) result.toArray(new String[result.size()]);
    }

    /**
     * Main method to test this class.
     *
     * @param args options
     */
    public static void main(String[] args) {
        runClassifier(new mAnDE(), args);
    }

}
