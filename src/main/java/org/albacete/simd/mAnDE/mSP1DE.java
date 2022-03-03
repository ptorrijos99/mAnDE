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
 *    mSP1DE.java
 *    Copyright (C) 2022 Universidad de Castilla-La Mancha, España
 *    @author Pablo Torrijos Arenas
 *
 */

package org.albacete.simd.mAnDE;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Objects;
import weka.core.Instance;
import weka.core.Utils;

public class mSP1DE implements mSPnDE {

    /**
     * Nombre del Super-Parent del mSP1DE.
     */
    private final String xi_s;

    /**
     * ID of the Super-Parent of the mSP1DE.
     */
    private final int xi_i;

    /**
     * Link the name of the children of the mSP1DE with their probability table.
     */
    private final HashMap<String, double[][][]> children;

    /**
     * List of children of the mSP1DE.
     */
    private final HashSet<String> listChildren;

    /**
     * Global probability table of the mSP1DE.
     */
    private double[][] globalProb;

    /**
     * Constructor. Build to mSP1DE passing it as argument the name of the variable xi that is going to be Super-Parent of the rest of the variables next to the class 'y'.
     * 
     * @param xi Father variable
     */
    public mSP1DE(String xi) {
        this.xi_s = xi;
        this.xi_i = mAnDE.nToI.get(xi);
        this.listChildren = new HashSet<>();
        this.children = new HashMap<>();
    }

    /**
     * Create the probability tables for the mSP1DE, both the global P(y,Xi) and
     * the conditional for each variable P(Xj|y,Xi).
     */
    @Override
    public void buildTables() {
        this.globalProb = new double[mAnDE.classNumValues] //y
                [mAnDE.varNumValues[xi_i]]; //Xi

        listChildren.forEach((child) -> {
            this.children.put(child, new double[mAnDE.classNumValues] //y
                    [mAnDE.varNumValues[xi_i]] //Xi
                    [mAnDE.varNumValues[mAnDE.nToI.get(child)]]); //Xj
        });

        // Creation of the contigency tables
        for (int i = 0; i < mAnDE.numInstances; i++) {
            Instance inst = mAnDE.data.get(i);

            // Creation of the probability table P(y,Xi)
            globalProb[(int) inst.value(mAnDE.y)][(int) inst.value(xi_i)] += 1;

            // Creation of the probability table P(y,Xi)
            children.forEach((String xj, double[][][] tablaXj) -> {
                int xj_i = mAnDE.nToI.get(xj);
                tablaXj[(int) inst.value(mAnDE.y)][(int) inst.value(xi_i)][(int) inst.value(xj_i)] += 1;
            });
        }

        // Conversion to Joint Probability Distribution
        for (double[] globalProb_y : globalProb) {
            for (int j = 0; j < globalProb_y.length; j++) {
                globalProb_y[j] /= mAnDE.numInstances;
            }
        }

        // Conversion to Conditional Probability Distribution
        children.forEach((String xj, double[][][] tableXj) -> {
            double sum;
            for (double[][] tableXj_y : tableXj) {
                for (double[] tableXj_y_xi : tableXj_y) {
                    sum = Utils.sum(tableXj_y_xi);
                    if (sum != 0) {
                        for (int k = 0; k < tableXj_y_xi.length; k++) {
                            tableXj_y_xi[k] /= sum;
                        }
                    }
                }
            }
        });
    }

    /**
     * Calculates the probabilities for each value of the class given an instance. To do this, the formula is applied: P(y,Xi) * (\prod_{i=1}^{Children} P(Xj|y,Xi)), with Xi being the parent variable in the mSP1DE, and Xj each of the child variables.
     *
     * @param inst Instance on which to compute the class.
     * @return Probabilities for each value of the class for the given instance.
     */
    @Override
    public double[] probsForInstance(Instance inst) {
        double[] res = new double[mAnDE.classNumValues];
        double xi = inst.value(xi_i);

        // We initialise the probability of each class value to P(y,xi).
        for (int i = 0; i < res.length; i++) {
            res[i] = globalProb[i][(int) xi];
        }

        /* For each child Xj, we multiply P(Xj|y,Xi) by the result 
         * accumulated for each of the values of the class
        */
        children.forEach((String xj_s, double[][][] tablaXj) -> {
            for (int i = 0; i < res.length; i++) {
                res[i] *= tablaXj[i][(int) xi][(int) inst.value(mAnDE.nToI.get(xj_s))];
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
     * Add a variable as a child in the mSP1DE.
     *
     * @param child Name of the variable to be added as a child.
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
     * Returns the number of children of mSP1DE.
     * 
     * @return The number of children
     */
    @Override
    public int getNChildren() {
        return children.size();
    }

    /**
     *
     * @param o Object to compare.
     * @return True if the objects are equal and False if they are not.
     */
    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (o == null) {
            return false;
        }
        if (!(o instanceof mSP1DE)) {
            return false;
        }

        mSP1DE that = (mSP1DE) o;
        return super.equals(that)
                && Objects.equals(this.xi_i, that.xi_i)
                && Objects.equals(this.xi_s, that.xi_s);
    }

    @Override
    public int hashCode() {
        int hash = 3;
        hash = 47 * hash + Objects.hashCode(this.xi_s);
        hash = 47 * hash + this.xi_i;
        return hash;
    }
}
