/*
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

 /*
 *    Bagging2.java
 *    Copyright (C) 1999-2012 University of Waikato, Hamilton, New Zealand
 *
 */
/**
 *
 * @author Pablo Torrijos Arenas
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
