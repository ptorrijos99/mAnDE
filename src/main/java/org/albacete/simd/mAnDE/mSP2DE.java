package org.albacete.simd.mAnDE;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Objects;
import weka.core.Instance;
import weka.core.Utils;

/**
 *
 * @author Pablo Torrijos Arenas
 */
public class mSP2DE implements mSPnDE {

    /**
     * Name of the first Super-Parent of the mSP2DE.
     */
    private final String xi1_s;

    /**
     * ID of the first Super-Parent of the mSP2DE.
     */
    private final int xi1_i;

    /**
     * Name of the second Super-Parent of the mSP2DE.
     */
    private final String xi2_s;

    /**
     * ID of the second Super-Parent of the mSP2DE.
     */
    private final int xi2_i;

    /**
     * Link the name of the children of the mSP2DE with their probability table.
     */
    private final HashMap<String, double[][][][]> children;

    /**
     * List of children of the mSP2DE.
     */
    private final HashSet<String> listChildren;

    /**
     * Overall probability table of the mSP2DE.
     */
    private double[][][] globalProbs;

    /**
     * Constructor. Creates an mSP2DE passing it as an argument the name of the two variables xi1 and xi2 that are going to be Super-Parents of the rest of the variables together with the class 'y'.
     *
     * @param xi1 Parent variable 1
     * @param xi2 Parent variable 2
     */
    public mSP2DE(String xi1, String xi2) {
        this.xi1_s = xi1;
        this.xi2_s = xi2;
        this.xi1_i = mAnDE.nToI.get(xi1);
        this.xi2_i = mAnDE.nToI.get(xi2);
        this.listChildren = new HashSet<>();
        this.children = new HashMap<>();
    }

    /**
     * Create the probability tables for the mSP2DE, both the global P(y,Xi) and the conditional for each variable P(Xj|y,Xi).
     */
    @Override
    public void buildTables() {
        this.globalProbs = new double[mAnDE.classNumValues] //y
                [mAnDE.varNumValues[xi1_i]] //Xi1
                [mAnDE.varNumValues[xi2_i]];    //Xi2

        listChildren.forEach((child) -> {
            this.children.put(child, new double[mAnDE.classNumValues] //y
                    [mAnDE.varNumValues[xi1_i]] //Xi1
                    [mAnDE.varNumValues[xi2_i]] //Xi2
                    [mAnDE.varNumValues[mAnDE.nToI.get(child)]]); //Xj
        });

        // Creation of contingency tables
        for (int i = 0; i < mAnDE.numInstances; i++) {
            Instance inst = mAnDE.data.get(i);

            // Creation of the probability table P(y,Xi1,Xi2)
            globalProbs[(int) inst.value(mAnDE.y)][(int) inst.value(xi1_i)][(int) inst.value(xi2_i)] += 1;

            // Creation of the probability table P(Xj|y,Xi1,Xi2)
            children.forEach((String xj, double[][][][] tableXj) -> {
                int xj_i = mAnDE.nToI.get(xj);
                tableXj[(int) inst.value(mAnDE.y)][(int) inst.value(xi1_i)][(int) inst.value(xi2_i)][(int) inst.value(xj_i)] += 1;
            });
        }

        // Conversion to Joint Probability Distribution
        for (double[][] globalProbs_y : globalProbs) {
            for (double[] globalProbs_y_x1 : globalProbs_y) {
                for (int j = 0; j < globalProbs_y_x1.length; j++) {
                    globalProbs_y_x1[j] /= mAnDE.numInstances;
                }
            }
        }

        // Conversion to Conditional Probability Distribution
        children.forEach((String xj, double[][][][] tableXj) -> {
            double sum;
            for (double[][][] tableXj_y : tableXj) {
                for (double[][] tableXj_y_xi1 : tableXj_y) {
                    for (double[] tableXj_y_xi1_xi2 : tableXj_y_xi1) {
                        sum = Utils.sum(tableXj_y_xi1_xi2);
                        if (sum != 0) {
                            for (int k = 0; k < tableXj_y_xi1_xi2.length; k++) {
                                tableXj_y_xi1_xi2[k] /= sum;
                            }
                        }
                    }
                }
            }
        });
    }

    /**
     * Calculates the probabilities for each value of the class given an instance. To do this, the formula is applied: P(y,Xi1,Xi2) * (\prod_{i=1}^{Children} P(Xj|y,Xi1,Xi2)), with Xi1 and Xi2 being the parent variables in the mSP2DE, and Xj each of the child variables.
     *
     * @param inst Instance on which to calculate the class.
     * @return Probabilities for each value of the class for the given instance.
     */
    @Override
    public double[] probsForInstance(Instance inst) {
        double[] res = new double[mAnDE.classNumValues];
        double xi1 = inst.value(xi1_i);
        double xi2 = inst.value(xi2_i);

        // We initialise the probability of each class value to P(y,xi).
        for (int i = 0; i < res.length; i++) {
            res[i] = globalProbs[i][(int) xi1][(int) xi2];
        }

        /* For each child Xj, we multiply P(Xj|y,Xi1,Xi2) by the result 
         * accumulated for each of the values of the class
         */
        children.forEach((String xj_s, double[][][][] tableXj) -> {
            for (int i = 0; i < res.length; i++) {
                res[i] *= tableXj[i][(int) xi1][(int) xi2][(int) inst.value(mAnDE.nToI.get(xj_s))];
            }
        });

        // We normalise the values by dividing them by the sum of all the values.
        double sum = Utils.sum(res);
        if (sum != 0) {
            for (int i = 0; i < res.length; i++) {
                res[i] /= sum;
            }
        }

        return res;
    }

    /**
     * Add a variable as a child in the mSP2DE.
     *
     * @param child Name of the variable to add as a child.
     */
    @Override
    public void moreChildren(String child) {
        if (!child.equals("")) {
            listChildren.add(child);
        }
    }

    /**
     * Add several variables as children in the mSP2DE.
     *
     * @param children Name of the variables to be added as children.
     */
    @Override
    public void moreChildren(ArrayList<String> children) {
        children.forEach((child) -> {
            if (!child.equals("")) {
                listChildren.add(child);
            }
        });
    }

    /**
     * Returns the number of children of the mSP2DE.
     * @return The number of children of the mSP2DE.
     */
    @Override
    public int getNChildren() {
        return children.size();
    }

    /**
     *
     * @param o Objeto a comparar.
     * @return True si los objetos son iguales y False si no lo son.
     */
    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (o == null) {
            return false;
        }
        if (!(o instanceof mSP2DE)) {
            return false;
        }

        mSP2DE that = (mSP2DE) o;
        return super.equals(that)
                && ((Objects.equals(this.xi1_i, that.xi1_i)
                && Objects.equals(this.xi1_s, that.xi1_s)
                && Objects.equals(this.xi2_i, that.xi2_i)
                && Objects.equals(this.xi2_s, that.xi2_s))
                || (Objects.equals(this.xi1_i, that.xi2_i)
                && Objects.equals(this.xi1_s, that.xi2_s)
                && Objects.equals(this.xi2_i, that.xi1_i)
                && Objects.equals(this.xi2_s, that.xi1_s)));
    }

    @Override
    public int hashCode() {
        int hash = 7;
        hash = 19 * hash + Objects.hashCode(this.xi1_s) + Objects.hashCode(this.xi2_s);
        hash = 19 * hash + this.xi1_i + this.xi2_i;
        return hash;
    }

}
