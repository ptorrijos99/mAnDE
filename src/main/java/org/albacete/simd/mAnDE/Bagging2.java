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
 *    Bagging2.java
 *    Copyright (C) 2022 Universidad de Castilla-La Mancha, España
 *    @author Pablo Torrijos Arenas
 *
 */

package org.albacete.simd.mAnDE;

import weka.classifiers.Classifier;
import weka.classifiers.meta.Bagging;
import weka.core.Debug.Random;
import weka.core.Instances;

public class Bagging2 extends Bagging {

    /**
     * The size of each bag sample, as a percentage of the training size
     *
     * Changued from int to double
     */
    protected double m_BagSizePercentDouble = 100;

    /**
     * Returns a training set for a particular iteration.
     *
     * Changued to use the variable m_BagSizePercentDouble instead of
     * m_BagSizePercent
     *
     * @param iteration the number of the iteration for the requested training
     * set.
     * @return the training set for the supplied iteration number
     * @throws Exception if something goes wrong when generating a training set.
     */
    @Override
    protected synchronized Instances getTrainingSet(int iteration) throws Exception {

        Random r = new Random(m_Seed + iteration);

        // create the in-bag indicator array if necessary
        if (m_CalcOutOfBag) {
            m_inBag[iteration] = new boolean[m_data.numInstances()];
            return m_data.resampleWithWeights(r, m_inBag[iteration], getRepresentCopiesUsingWeights(), m_BagSizePercentDouble);
        } else {
            return m_data.resampleWithWeights(r, null, getRepresentCopiesUsingWeights(), m_BagSizePercentDouble);
        }
    }

    /**
     * Sets the size of each bag, as a percentage of the training set size.
     *
     * @param newBagSizePercentDouble the bag size, as a percentage.
     */
    public void setBagSizePercentDouble(double newBagSizePercentDouble) {

        m_BagSizePercentDouble = newBagSizePercentDouble;
    }

    /**
     * Gets the size of each bag, as a percentage of the training set size.
     *
     * @return the bag size, as a percentage.
     */
    public double getBagSizePercentDouble() {

        return m_BagSizePercentDouble;
    }

    /**
     * Gets all of the classifiers of the ensemble.
     *
     * @return an array with the classifiers of the ensemble.
     */
    public Classifier[] getClassifiers() {
        return m_Classifiers;
    }

}
