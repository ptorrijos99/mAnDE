package weka.attributeSelection;

import java.util.ArrayList;
import java.util.Enumeration;
import java.util.Vector;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.OptionHandler;
import weka.core.RevisionUtils;
import weka.core.Statistics;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformationHandler;
import weka.core.Utils;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;
import weka.attributeSelection.SymmetricalUncertAttributeEval;

/**
 * <!-- globalinfo-start --> Class which implements the IWSS search algorithm.
 *  This is a wrapper algorithm, but the Naive Bayes classifier is 
 *  embedded in it, so this presents the advantages of a wrapper 
 *  search, with the time complexity of a filter algorithm.
 * <br/>
 * <br/>
 * For more information on this, see<br/>
 * <br/>
 * Speeding up incremental wrapper feature subset selection with Naive Bayes
 * classifier. Bermejo, Pablo; Gamez, Jose A.; Puerta, Jose M. KNOWLEDGE-BASED
 * SYSTEMS, vol. 55, p. 140--147. (2014).
 * <p/>
 * <!-- globalinfo-end -->
 * 
 * <!-- technical-bibtex-start --> BibTeX:
 * 
 * <pre>
 * &#64;inproceedings{Bermejo.et.al2014,
 *      author = {Pablo Bermejo and Jos\'e A. G\'amez mez and Jos\'e M. Puerta},
 *      title = {Speeding up incremental wrapper feature subset selection with Naive Bayes classifier},
 *      journal = {Knowledge-Based Systems},
 *      volume = {55},
 *      pages = {140 - 147},
 *      year = {2014},
 *      issn = {0950-7051},
 *      doi = {http://dx.doi.org/10.1016/j.knosys.2013.10.016},
 * }
 * </pre>
 * <p/>
 * <!-- technical-bibtex-end -->
 * 
 * <!-- options-start --> Valid options are:
 * <p/>
 * 
 * <pre>
 * -minFolds <num>
 *  number [1-5] of folds in the inner cross-validation which must
 *  be improved when testing a new attribute in the IWSS process.
 * </pre>
 * 
 * <pre>
 * -replace
 *  Add replacement option for the embedded IWSS algorithm.
 * </pre>
 * 
 * <pre>
 * -theta <num>
 *  (0-1] (1 means no Early stopping). Theta value of the early stopping criterion for the embedded IWSS algorithm.
 * </pre>
 * 
 * 
 * 
 * 
 * <!-- options-end -->
 * 
 * @author Pablo Bermejo (Pablo.Bermejo@uclm.es) * @version $Revision: 1.0 $
 */

public class IWSSembeddedNB extends ASSearch implements OptionHandler,
		StartSetHandler, TechnicalInformationHandler {

	/** for serialization */
	static final long serialVersionUID = 5174597384392985135L;

	

	/**
	 * Minimum number of folds to be improved when IWSS selects a new attribute.
	 */
	protected int m_minFolds = 2;

	/**
	 * value for early stopping: (0-1] (1 means no early stop) percentage of remaining
	 * attributes to visit in the IWSS algorithm
	 */
	protected double m_theta = 1.0;

	/**
	 * whether to use the attributes replacement enhancement of the IWSS
	 * algorithm
	 */
	protected boolean m_replace = false;

	
	/** The number of classes (or 1 for numeric class) */
	protected int m_numClasses;

	/** index of the class attribute */
	protected int m_classIndex;
	
	/** evaluator - never used in search */
	ASEvaluation m_evaluator;

	/**
	 * Number of folds for the inner cross validation of the IWSS process
	 */
	public static final int m_IWSSfolds = 5;
	
	/** marker for the inner cv in the embedded iwss process */
	int[] firstOfFold = new int[m_IWSSfolds];

	/** marker for the innver cv in the embedded iwss process */
	int[] lastOfFold = new int[m_IWSSfolds];

	/**
	 * Number of former attributes in ranking to be straight-forward selected
	 * before running the incremental search.
	 */
	protected int m_formerSelected = 0;

	/** It is set to true if IWSS.setStartSet is called */
	protected boolean m_skipRanking = false;

	/** the data structure to store the probs for each instance and attribute */
	double[][] jointProb = null;
	ArrayList<double[][]> lpAttributes = new ArrayList<double[][]>();

	/** selected attributes in the embedded process */
	ArrayList<Integer> m_selected;

	/**
	 * The discretization filter.
	 */
	protected weka.filters.supervised.attribute.Discretize m_Disc = null;

	/**
	 * Returns a string describing this classifier
	 * 
	 * @return a description of the classifier suitable for displaying in the
	 *         explorer/experimenter gui
	 */
	public String globalInfo() {
		return "Class which implements the IWSS search algorithm. This is a wrapper algorithm, but the "
		+ "Naive Bayes classifier is embedded in it, so this presents the advantages of "		+ ""
		+ "a wrapper search, with the time complexity of a filter algorithm.\n "
		+ "For more information on this version of NB with embedded feature selection, see\n\n"
				+ getTechnicalInformation().toString();
	}

	/**
	 * Returns an instance of a TechnicalInformation object, containing detailed
	 * information about the technical background of this class, e.g., paper
	 * reference or book this class is based on.
	 * 
	 * @return the technical information about this class
	 */
	public TechnicalInformation getTechnicalInformation() {
		TechnicalInformation result;

		result = new TechnicalInformation(Type.ARTICLE);
		result.setValue(Field.AUTHOR,
				"Pablo Bermejo and Jose A. Gamez and Jose M. Puerta");
		result.setValue(
				Field.TITLE,
				"Speeding up incremental wrapper feature subset selection with Naive Bayes classifier");
		result.setValue(Field.JOURNAL, "Knowledge-Based Systems");
		result.setValue(Field.VOLUME, "55");
		result.setValue(Field.YEAR, "2014");
		result.setValue(Field.PAGES, "140-147");
		result.setValue(Field.PUBLISHER, "Elsevier");

		return result;
	}

	/**
	 * performs the search
	 * 
	 * @param instances
	 *            set of instances serving as training data
	 * @exception Exception
	 *                if the classifier has not been generated successfully
	 */
	@Override
	public int[] search(ASEvaluation ASEvaluator, Instances instances)
			throws Exception {

		m_evaluator=ASEvaluator;
		
		if (!m_replace)
			return iwss(instances);
		else
			return iwssR(instances);
	}

	public int[] iwss(Instances instances) throws Exception {
		int[] ranking = getRanking(instances);

		m_numClasses = instances.numClasses();
		m_classIndex = instances.classIndex();
		m_selected = new ArrayList<Integer>();
		instances.stratify(m_IWSSfolds);

		// preparing the folds for NUMFOLDS-cv and data structures
		prepareFolds(instances);
		double[] currentAcc = new double[m_IWSSfolds];
		double[] bestAcc = new double[m_IWSSfolds];
		double currentAvgAcc = -1.0;
		double bestAvgAcc = -1.0;
		for (int i = 0; i < m_IWSSfolds; i++) {
			bestAcc[i] = -1.0;
		}
		ArrayList<ArrayList<Potential>> classifiers = new ArrayList<ArrayList<Potential>>(
				m_IWSSfolds);
		for (int f = 0; f < m_IWSSfolds; f++) {
			ArrayList<Potential> l = new ArrayList<Potential>();
			classifiers.add(l);
		}

		// running the main process
		initFoldsClassifiers(instances);
		int improvement;

		int toVisit = (int) (ranking.length * m_theta);
		for (int i = m_formerSelected; i < toVisit; i++) {
			m_selected.add(ranking[i]);
			improvement = 0;
			currentAvgAcc = 0.0;
			ArrayList<Potential> newDist = new ArrayList<Potential>(m_IWSSfolds);
			double[][] newTotal = new double[instances.numInstances()][m_numClasses];
			double[][] newLP = new double[instances.numInstances()][m_numClasses];
			for (int j = 0; j < m_IWSSfolds; j++) {
				Potential pot = this.getConditionalClassDistribution(
						ranking[i], j, instances);
				newDist.add(pot);
				updateLogProbAtt(pot, j, newLP, instances);
				currentAcc[j] = updateTotalByAddition(newTotal, newLP, j,
						instances);
				currentAvgAcc += currentAcc[j];
				if (currentAcc[j] > bestAcc[j]) {
					improvement++;
				}
			}

			currentAvgAcc /= ((double) m_IWSSfolds);
			if ((currentAvgAcc > bestAvgAcc)
					&& (improvement >= m_minFolds)) {
				for (int j = 0; j < m_IWSSfolds; j++) {
					bestAcc[j] = currentAcc[j];
				}
				bestAvgAcc = currentAvgAcc;

				toVisit = i + (int) (m_theta * (ranking.length - i));

				// updating the classifier
				jointProb = newTotal;
				for (int f = 0; f < m_IWSSfolds; f++) {
					classifiers.get(f).add(newDist.get(f));
				}
				// lpAttributes.add(newLP);

			} else {
				m_selected.remove(m_selected.size() - 1);
			}

		}

		return getSelected();
	}

	public int[] iwssR(Instances dataset) throws Exception {
		// getting the ranking
		int[] ranking = this.getRanking(dataset);

		m_numClasses = dataset.numClasses();
		m_classIndex = dataset.classIndex();
		m_selected = new ArrayList<Integer>();
		dataset.stratify(m_IWSSfolds);

		// preparing the folds for NUMFOLDS-cv and data structures
		prepareFolds(dataset);
		double[] currentAcc = new double[m_IWSSfolds];
		double[] bestAcc = new double[m_IWSSfolds];
		double[] bestItAcc = new double[m_IWSSfolds];
		double currentAvgAcc = -1.0;
		double bestAvgAcc = -1.0;
		double bestItAvgAcc = -1.0;
		for (int i = 0; i < m_IWSSfolds; i++) {
			bestAcc[i] = -1.0;
		}

		ArrayList<ArrayList<Potential>> classifiers = new ArrayList<ArrayList<Potential>>(
				m_IWSSfolds);
		for (int f = 0; f < m_IWSSfolds; f++) {
			ArrayList<Potential> l = new ArrayList<Potential>();
			classifiers.add(l);
		}

		// running the main process
		initFoldsClassifiers(dataset);
		int improvement;
		int bestMove = -1; // -1 means no improvement, 0, 1, ... means
							// replacement, selected.size() means addition

		int toVisit = (int) (ranking.length * m_theta);
		for (int i = 0; i < toVisit; i++) {
			for (int f = 0; f < m_IWSSfolds; f++) {
				bestItAcc[f] = -1.0;
			}
			bestItAvgAcc = -1.0;
			bestMove = -1;

			// first we test addition
			m_selected.add(ranking[i]);

			improvement = 0;
			currentAvgAcc = 0.0;
			ArrayList<Potential> newDist = new ArrayList<Potential>(m_IWSSfolds);
			double[][] newTotal = new double[dataset.numInstances()][m_numClasses];
			double[][] newLP = new double[dataset.numInstances()][m_numClasses];
			double[][] bestItTotal = null;
			for (int j = 0; j < m_IWSSfolds; j++) {
				Potential pot = this.getConditionalClassDistribution(
						ranking[i], j, dataset);
				newDist.add(pot);

				updateLogProbAtt(pot, j, newLP, dataset);
				currentAcc[j] = updateTotalByAddition(newTotal, newLP, j,
						dataset);
				currentAvgAcc += currentAcc[j];
				if (currentAcc[j] > bestAcc[j]) {
					improvement++;
				}
			}

			currentAvgAcc /= (double) m_IWSSfolds;
			if ((currentAvgAcc > bestAvgAcc)
					&& (improvement >= m_minFolds)) {
				for (int j = 0; j < m_IWSSfolds; j++) {
					bestItAcc[j] = currentAcc[j];
				}
				bestItAvgAcc = currentAvgAcc;
				bestMove = m_selected.size();

				bestItTotal = newTotal;
			}

			// testing swapping
			for (int p = 0; p < m_selected.size() - 1; p++) {

				improvement = 0;
				currentAvgAcc = 0.0;
				newTotal = new double[dataset.numInstances()][m_numClasses];

				for (int j = 0; j < m_IWSSfolds; j++) {
					currentAcc[j] = updateTotalByReplacement(newTotal, newLP,
							j, p, dataset);
					currentAvgAcc += currentAcc[j];
					if (currentAcc[j] > bestAcc[j]) {
						improvement++;
					}
				}

				currentAvgAcc /= (double) m_IWSSfolds;
				if ((currentAvgAcc > bestAvgAcc)
						&& (improvement >= m_minFolds)) {// is a candidate
					if (currentAvgAcc > bestItAvgAcc) {// in fact is the best
														// candidate up to now!!
						for (int j = 0; j < m_IWSSfolds; j++) {
							bestItAcc[j] = currentAcc[j];
						}
						bestItAvgAcc = currentAvgAcc;
						bestMove = p;

						bestItTotal = newTotal;
					}
				}

			}// end of "swapping" for

			if (bestMove != -1) {
				for (int j = 0; j < m_IWSSfolds; j++) {
					bestAcc[j] = bestItAcc[j];
				}
				bestAvgAcc = bestItAvgAcc;

				// updating the classifier
				jointProb = bestItTotal;
				for (int f = 0; f < m_IWSSfolds; f++) {
					classifiers.get(f).add(newDist.get(f));
				}
				lpAttributes.add(newLP);

				if (bestMove == m_selected.size()) {// addition

				} else { // replacement
					m_selected.remove(bestMove);
					lpAttributes.remove(bestMove);
				}

				toVisit = i + (int) (m_theta * (ranking.length - i));

			} else {// discarding the studied attribute
				m_selected.remove(m_selected.size() - 1);
			}

		}

		return getSelected();

	}

	/**
	 * Update the sum of logs (joint probability) for each class, for all
	 * instance, for the corresponding fold, having that and attribute is
	 * inserted and another is removed (replacedPos)
	 * 
	 * @param newTotal
	 *            joint probabilities vector to update
	 * @param LP
	 *            logarithm of the probability of the attribute (whose insertion
	 *            is being tested) over instances, for each class
	 * @param fold
	 *            corresponding fold
	 * @param replacedPos
	 *            attribute index logs to remove from total sum
	 * @return accuracy for this fold when testing the swapping in current
	 *         selected subet of a new attribute by one already selected
	 */
	private double updateTotalByReplacement(double[][] newTotal, double[][] LP,
			int fold, int replacedPos, Instances dataset) {
		double acc = 0;
		double max = -1.0;
		int maxPos = -1;
		Instance row = null;
		double[][] LPtoQuit = lpAttributes.get(replacedPos);

		for (int i = firstOfFold[fold]; i <= lastOfFold[fold]; i++) {
			row = dataset.instance(i);
			max = -1000000000000.0;
			for (int c = 0; c < m_numClasses; c++) {
				newTotal[i][c] = jointProb[i][c] + LP[i][c] - LPtoQuit[i][c];
				if (newTotal[i][c] > max) {
					max = newTotal[i][c];
					maxPos = c;
				}
			}
			if (((int) row.value(m_classIndex)) == maxPos) {
				acc += 1.0;
			}
		}

		acc /= (double) (lastOfFold[fold] - firstOfFold[fold] + 1);

		return acc;
	}

	/**
	 * 
	 * @return the indexes of selected attribute after the IWSS(NB) search
	 */
	public int[] getSelected() {

		int[] s = new int[m_selected.size()];
		for (int i = 0; i < m_selected.size(); i++) {
			s[i] = m_selected.get(i);
		}

		return s;
	}

	/**
	 * Compute LogP for a given attribute and the set of instances corresponding
	 * to the required fold
	 * 
	 * @param pot
	 *            Potential of the attribute whose LogP is to be computed
	 * @param fold
	 *            of inner cross validation respect from which to get the
	 *            distribution
	 * @param matrix
	 *            to update
	 * 
	 */
	private void updateLogProbAtt(Potential pot, int fold, double[][] column,
			Instances dataset) {

		Instance row;
		double[][] logDist = null;

		if (pot.isDiscrete) {// precomputing logs
			logDist = new double[pot.numStates][m_numClasses];
			for (int v = 0; v < pot.numStates; v++) {
				for (int c = 0; c < m_numClasses; c++) {
					if (Utils.eq(pot.distribution[v][c], 0.0)) {
						logDist[v][c] = 0.0; // never should happen because of
												// laplace smoothing
					} else {
						logDist[v][c] = Math.log(pot.distribution[v][c]);
					}
				}
			}
		}

		double probValue;
		for (int i = firstOfFold[fold]; i <= lastOfFold[fold]; i++) {
			row = dataset.instance(i);
			for (int c = 0; c < m_numClasses; c++) {
				if (row.isMissing(pot.attIndex)) {
					column[i][c] = 0.0; // for missing I use 0.0 because is the
										// neutral
				} // element for sum, so this is like discarding them
				else {
					if (pot.isDiscrete) {
						column[i][c] = logDist[(int) row.value(pot.attIndex)][c];
					} else {
						probValue = getProbabilityFromNormalDistribution(
								row.value(pot.attIndex),
								pot.distribution[0][c], pot.distribution[1][c],
								pot.precision);
						if (probValue < 1e-300) {
							column[i][c] = -1000;
						} else {
							column[i][c] = Math.log(probValue);
						}

					}
				}
			}

		}
	}

	/**
	 * Update the sum of logs (joint probability) for each class, for all
	 * instance, for the corresponding fold
	 * 
	 * @param newTotal
	 *            joint probabilities vector to update
	 * @param LP
	 *            logarithm of the probability of the attribute (whose insertion
	 *            is being tested) over instances, for each class
	 * @param fold
	 *            corresponding fold
	 * @return accuracy for this fold when testing the insertion in current
	 *         selected subet of a new attribute
	 */
	private double updateTotalByAddition(double[][] newTotal, double[][] LP,
			int fold, Instances dataset) {
		double acc = 0;
		double max = -1.0;
		int maxPos = -1;
		Instance row = null;

		for (int i = firstOfFold[fold]; i <= lastOfFold[fold]; i++) {
			row = dataset.instance(i);
			max = -1000000000000.0;
			for (int c = 0; c < m_numClasses; c++) {
				newTotal[i][c] = jointProb[i][c] + LP[i][c];
				if (Utils.gr(newTotal[i][c], max)) {
					max = newTotal[i][c];
					maxPos = c;
				}

			}
			if (((int) row.value(m_classIndex)) == maxPos) {
				acc += 1.0;
			}
		}

		acc /= (double) (lastOfFold[fold] - firstOfFold[fold] + 1);

		return acc;
	}

	/**
	 * @param v
	 *            value to round
	 * @param precision
	 *            precision when roudning v
	 * @return the result of rounding the passed value Round a data value using
	 *         the defined precision
	 */
	private double round(double v, double precision) {
		return Math.rint(v / precision) * precision;
	}

	/**
	 * Get a probability estimate for a value given a normal distribution
	 * 
	 * @param value
	 *            whose probability is being estimated
	 * @param mean
	 *            of normal distribution
	 * @param stdDev
	 *            standard deviation of normal distribution
	 * @param precision
	 *            precision for probability
	 * @return the probability
	 */
	public double getProbabilityFromNormalDistribution(double value,
			double mean, double stdDev, double precision) {

		double v = round(value, precision);
		double zLower = (v - mean - (precision / 2)) / stdDev;
		double zUpper = (v - mean + (precision / 2)) / stdDev;

		double pLower = Statistics.normalProbability(zLower);
		double pUpper = Statistics.normalProbability(zUpper);
		return pUpper - pLower;
	}

	/**
	 * Gets precision for a numerical variable. Copyed from Weka(NaiveBayes)
	 * 
	 * @param attIndex
	 * @param fold
	 *            corresponding fold of inner cross validation
	 * @return the precision of passed attribute
	 */
	private double getPrecision(int attIndex, int fold, Instances dataset) {
		double precision = 0.01;

		int n = dataset.numInstances()
				- (lastOfFold[fold] - firstOfFold[fold] + 1);
		double[] copyOfAtt = new double[n];
		int k = 0;
		for (int f = 0; f < m_IWSSfolds; f++) {
			if (f != fold) {
				for (int i = firstOfFold[f]; i <= lastOfFold[f]; i++, k++) {
					copyOfAtt[k] = dataset.instance(i).value(attIndex);
				}
			}
		}

		int[] sortedPos = Utils.sort(copyOfAtt);
		if ((n == 0) || (Utils.isMissingValue(copyOfAtt[sortedPos[0]]))) {
			return precision;
		}

		double lastVal = copyOfAtt[sortedPos[0]];
		double currentVal, deltaSum = 0;
		int distinct = 0;
		for (int i = 1; i < n; i++) {
			if (Utils.isMissingValue(copyOfAtt[sortedPos[i]])) {
				break;
			}

			currentVal = copyOfAtt[sortedPos[i]];
			if (currentVal != lastVal) {
				deltaSum += currentVal - lastVal;
				lastVal = currentVal;
				distinct++;
			}
		}

		if (distinct > 0) {
			precision = deltaSum / distinct;
		}

		return precision;
	}

	/**
	 * Gets the conditional distribution of an attribute for the given fold
	 * 
	 * @param attIndex
	 *            attribute whose conditional distribution is desired
	 * @param fold
	 *            of inner cross validation respect from which to get the
	 *            distribution
	 * @return conditional class distribution of attIndex given class for
	 *         indicated fold
	 */
	private Potential getConditionalClassDistribution(int attIndex, int fold,
			Instances dataset) {
		Potential pot = new Potential();
		Attribute att = dataset.attribute(attIndex);
		pot.attIndex = attIndex;

		if (att.isNominal()) {
			pot.isDiscrete = true;
			pot.numStates = att.numValues();
		} else if (att.isNumeric()) {
			pot.isDiscrete = false;
			pot.numStates = 2;
			pot.precision = getPrecision(attIndex, fold, dataset);
		}
		double[][] dist = new double[pot.numStates][m_numClasses];
		for (int i = 0; i < pot.numStates; i++) {
			for (int j = 0; j < m_numClasses; j++) {
				dist[i][j] = 0.0; // in the numeric case [0][c] = mean and
									// [1][c] = stdDev
			}
		}
		Instance row = null;
		int n = 0;
		double[] sumOfValues = new double[m_numClasses]; // only for numerical
		double[] sumOfValuesSq = new double[m_numClasses]; // only for numerical
		int[] count = new int[m_numClasses]; // only for numerical

		if (!pot.isDiscrete) {
			for (int c = 0; c < m_numClasses; c++) {
				dist[1][c] = pot.precision / (2 * 3);
				count[c] = 0;
				sumOfValues[c] = 0.0;
				sumOfValuesSq[c] = 0.0;
			}
		}

		for (int f = 0; f < m_IWSSfolds; f++) {
			if (f == fold) {
				continue;
			}
			for (int i = firstOfFold[f]; i <= lastOfFold[f]; i++) {
				row = dataset.instance(i);
				if (!row.isMissing(attIndex)) {
					n++;
					if (pot.isDiscrete) {
						dist[(int) row.value(attIndex)][(int) row
								.value(m_classIndex)]++;
					} else {// numeric
						int classValue = (int) row.value(m_classIndex);
						count[classValue]++;
						double value = round(row.value(attIndex), pot.precision);
						sumOfValues[classValue] += value;
						sumOfValuesSq[classValue] += value * value;

						dist[0][classValue] = sumOfValues[classValue]
								/ count[classValue];
						double sd = Math.sqrt(Math
								.abs(sumOfValuesSq[classValue]
										- dist[0][classValue]
										* sumOfValues[classValue])
								/ ((double) count[classValue]));
						if (sd > 1e-10) {
							dist[1][classValue] = Math.max(pot.precision
									/ (2 * 3), sd);
						}

					}
				}
			}
		}

		// from counts to probs.
		if (pot.isDiscrete) {
			double[] totalClass = new double[m_numClasses];
			for (int c = 0; c < m_numClasses; c++) {
				totalClass[c] = 0.0;
				for (int v = 0; v < pot.numStates; v++) {
					totalClass[c] += dist[v][c];
				}
			}
			for (int c = 0; c < m_numClasses; c++) {
				for (int v = 0; v < pot.numStates; v++) {
					dist[v][c] = (dist[v][c] + 1.0)
							/ (totalClass[c] + pot.numStates);// Laplace
																// smoothing
				}
			}
		} else {// numeric
			// in this case mean and variance have been direcly updated while
			// counting
		}

		pot.distribution = dist;

		return pot;
	}

	/**
	 * Prepares train and test data for evaluation and counting how many folds
	 * get higher accuracy than folds of evaluation with the up to now best
	 * features subset. No projections are performed but pointer to folds are
	 * used.
	 * 
	 */
	private void prepareFolds(Instances dataset) {
		int numInstances = dataset.numInstances();
		int numInstForFold = numInstances / m_IWSSfolds;
		int last = -1;
		for (int numFold = 0; numFold < m_IWSSfolds; numFold++) {
			firstOfFold[numFold] = last + 1;
			if (numFold < numInstances % m_IWSSfolds) {
				last += numInstForFold + 1;
			} else {
				last += numInstForFold;
			}
			lastOfFold[numFold] = last;
		}

	}

	/**
	 * Ranks attributes based on Simmetrical Uncertainty conditioned on class
	 * attribute.
	 * 
	 * @param data
	 *            Instances with attributes to rank
	 * @return integer vector with attributes ordered from max to min
	 *         Simmetrical Uncertainty
	 * @exception if
	 *                something goes wrong during ranking
	 */
	public int[] getRanking(Instances data) throws Exception {
		int[] ranking = new int[data.numAttributes() - 1];

		if (!m_skipRanking) {
			SymmetricalUncertAttributeEval evaluator = new SymmetricalUncertAttributeEval();
			evaluator.buildEvaluator(data);

			double[] values = new double[data.numAttributes() - 1];
			for (int i = 0; i < data.numAttributes() - 1; i++) {
				values[i] = evaluator.evaluateAttribute(i);
			}

			int[] aux = Utils.stableSort(values);
			for (int i = 0; i < data.numAttributes() - 1; i++) {
				ranking[i] = aux[aux.length - i - 1];
			}
		} else {

			for (int i = 0; i < ranking.length; i++)
				ranking[i] = i;
		}

		return ranking;
	}

	/**
	 * Calculates the class membership probabilities for the given test
	 * instance.
	 * 
	 * @param instance
	 *            the instance to be classified
	 * @return predicted class probability distribution
	 * @exception Exception
	 *                if there is a problem generating the prediction
	 */

	/**
	 * get a String[] describing the value set for all options
	 * 
	 * @return String[] describing the options
	 */
	public String[] getOptions() {
		int length = 4;

		if (m_replace)
			length++;

		int current = 0;

		String[] options = new String[length];
		options[current++] = "-minFolds";
		options[current++] = "" + m_minFolds;
		options[current++] = "-theta";
		options[current++] = "" + m_theta;

		if (m_replace)
			options[current++] = "-replace";

		return options;
	}

	/**
	 * Returns a description of the search
	 * 
	 * @return a description of the classifier as a string.
	 */
	public String toString() {
		
		String s = "";
		s += "Incremental Wrapper Subset Selection (IWSS) with embedded Naive Bayes:\n";
		s += "Please note the attribute evaluator chosen (" + m_evaluator.getClass() +") is ignored. " +
				"Embedded Naive Bayes is used instead.";
		s += "\nMinimum folds better:"+m_minFolds+"\n";
		s += "Selected Attributes: " + m_selected.toString() + "\n";
		if (m_theta < 1)
			s += "Early Stopping was used: theta=" + m_theta + "\n";
		if (m_replace)
			s += "Replace selection was used: IWSSr mode\n";
		
		
		return s;

	}

	/**
	 * Returns the revision string.
	 * 
	 * @return the revision
	 */
	public String getRevision() {
		return RevisionUtils.extract("$Revision: 1 $");
	}

	/**
	 * Initializes the classifier for each fold just by computing the class
	 * distribution and intilializes the value of the joint prob for each
	 * instance as the log of the class probability
	 */
	private void initFoldsClassifiers(Instances dataset) {
		double[][] dist = new double[m_IWSSfolds][m_numClasses];

		int n;
		for (int f = 0; f < m_IWSSfolds; f++) {
			n = 0;
			for (int i = 0; i < m_numClasses; i++) {
				dist[f][i] = 0.0;
			}
			for (int i = 0; i < m_IWSSfolds; i++) {
				if (i != f) {
					for (int j = firstOfFold[i]; j <= lastOfFold[i]; j++) {
						dist[f][(int) dataset.instance(j).value(m_classIndex)]++;
					}
					n += lastOfFold[i] - firstOfFold[i] + 1;
				}
			}
			for (int i = 0; i < m_numClasses; i++) {
				dist[f][i] = (dist[f][i] + 1) / (double) (n + m_numClasses);// Laplace
																			// smoothing
			}
		}

		// now initializing the joint probability of each instance
		int numInstances = dataset.numInstances();
		jointProb = new double[numInstances][m_numClasses];
		for (int f = 0; f < m_IWSSfolds; f++) {
			for (int i = firstOfFold[f]; i <= lastOfFold[f]; i++) {
				for (int c = 0; c < m_numClasses; c++) {
					jointProb[i][c] = Math.log(dist[f][c]);
				}
			}
		}

	}

	/**
	 * 
	 * @param mf
	 *            minimum number of folds whose wrapper goodness must be
	 *            improved in the inner cross-validation when selecting a new
	 *            attribute.
	 */
	public void setMinFolds(int mf) {
		m_minFolds = mf;
	}

	
	/**
	 * 
	 * @param t
	 *            the value to tune the early stopping procedure. Range is
	 *            (0-1]. When t=1, early stopping is turned off.
	 */
	public void setTheta(double t) {
		if (t > 1)
			t = 1;
		if (t <= 0)
			t = 0.01;
		m_theta = t;
	}

	/**
	 * Returns the tip text for this property
	 * 
	 * @return tip text for this property suitable for displaying in the
	 *         explorer/experimenter gui
	 */
	public String thetaTipText() {
		return "(0-1] (1 means no Early Stopping). Percentage of remaining attributes to test without acceptance ";
	}

	/**
	 * Activate the option of testing, at each step of the incremental search,
	 * the swapping of a selected feature with another not selected yet. This
	 * increases the worst-case theoretical complexity from linear to quadratic.
	 * 
	 * @param r
	 */
	public void setReplace(boolean r) {
		m_replace = r;
	}

	public String replaceTipText() {
		return "Test, at each step of the incremental search, the swapping of a "
				+ "selected feature with another not selected yet. This increases the worst-case "
				+ "theoretical complexity of IWSS from linear to quadratic.";

	}

	/**
	 * Setting a start set instancies m_skipRanking=true. Since this is an
	 * incremental search on a ranking, a startset can only be a set of
	 * consecutive numbers from 0 to N. When using this method, it is supposed
	 * that attributes in dataset are ranked by decreasing order; that is,
	 * attribute 0 is better thank attribute 1 in m_dataset.
	 * 
	 * @param set
	 *            a string set of consecutive values starting by 1 (for GUI
	 *            Explorer compatibility), separated by comma.
	 * @throws Exception
	 *             if set does not start by 1 or numbers are not consecutive
	 * */

	@Override
	public void setStartSet(String set) throws Exception {
		String[] numbers = set.split(",");

		if (!set.equals("")) {
			if (Integer.parseInt(numbers[0]) != 1)
				throw new Exception(
						"Start set must start by 1 because. Start set refers to the former and consecutive attributes in ranking.");
			for (int i = 0; i < numbers.length; i++) {
				if (Integer.parseInt(numbers[i]) != (i + 1))
					throw new Exception(
							"Numbers in start set must be consecutive. Start set refers to the former and consecutive attributes in ranking.: "
									+ set);
			}
		}
		m_formerSelected = set.equals("") ? 1 : numbers.length;
		m_skipRanking = true;

	}

	/**
	 * 
	 * @return the start set
	 */
	@Override
	public String getStartSet() {
		String s = "";
		for (int i = 1; i <= m_formerSelected; i++)
			s = i + ",";
		return s.substring(0, s.length() - 1); // skip last comma
	}

	/**
	 * Parses a given list of options.
	 * <p/>
	 * 
	 * <!-- options-start --> Valid options are:
	 * <p/>
	 * 
	 * 
	 * <pre>
	 *  -minFolds <num>
	 *  minimum number of folds whose wrapper goodness must be improved in the inner cross-validation when selecting a new attribute.
	 *  (default 2)
	 * </pre>
	 * 
	 * <pre>
	 *  -theta <num>
	 *   the value to tune the early stopping procedure. Range is (0-1]. When theta=1, early stopping is turned off.
	 *  (default 1)
	 * </pre>
	 * 
	 * 
	 * 
	 * <pre>
	 *   -replace
	 *   Flag to activate the option of testing, at each step of the incremental search, the swapping of
	 *   a selected feature with another not selected yet. This increases the worst-case theoretical complexity
	 * from linear to quadratic.
	 * 
	 * <pre>
	 * 
	 * <!-- options-end -->
	 * 
	 * @param options
	 *            the list of options as an array of strings
	 * 
	 * @throws Exception
	 * if an option is not supported
	 */
	@Override
	public void setOptions(String[] options) throws Exception {
		resetOptions();

		String selectionString = Utils.getOption("minFolds", options);
		if (selectionString.length() != 0) {
			setMinFolds(Integer.parseInt(selectionString));
		}

		selectionString = Utils.getOption("theta", options);
		if (selectionString.length() != 0) {
			setTheta(Double.parseDouble(selectionString));
		}

		
		setReplace(Utils.getFlag("replace", options));

	}
	
	/**
	 * 
	 * @return The minimum number of folds in inner CV whose wrapper goodness
	 *         must be improved when selecting a new attribute.
	 */
	public int getMinFolds() {
		return m_minFolds;
	}

	/**
	 * 
	 * @return The value of the parameter (0-1] which controls the early
	 *         stopping
	 */
	public double getTheta() {
		return m_theta;
	}
	
	/**
	 * 
	 * @return true if attributes replace selection is set for the IWSS process
	 */
	public boolean getReplace() {
		return m_replace;
	}

	


	/**
	 * Returns the tip text for this property
	 * 
	 * @return tip text for this property suitable for displaying in the
	 *         explorer/experimenter gui
	 */
	public String minFoldsTipText() {
		return "Min number of folds in inner CV to be improved for new attribute acceptance.";
	}
	
	
	
	/**
	 * Reset all options to their default values
	 */
	public void resetOptions() {

		m_minFolds = 2;
		m_theta = 1.0;
		m_replace = false;

	}

	/**
	 * Returns an enumeration describing the available options.
	 * 
	 * @return an enumeration of all the available options.
	 **/
	@Override
	public Enumeration<Option> listOptions() {

		Vector<Option> newVector = new Vector<Option>(3);

		newVector
				.addElement(new Option(
						"\tMin number of folds to be better in order to accept the new variable\n",
						"minFolds", 1, "-minFolds <num>"));

		newVector
				.addElement(new Option(
						"\t(0-1] (1 means no Early Stopping). Percentage of remaining attributes to test without acceptance\n",
						"theta", 1, "-theta <num>"));

		newVector
				.addElement(new Option(
						"\t Whether to apply the attribute replacement enhachement of the IWSS search algorithm",
						"replace", 0, "-replace"));

		return newVector.elements();

	}

}


class Potential {

	/** if false the var is of numerical type */
	public boolean isDiscrete;
	/** default is for numerical vars (mean and variance) */
	public int numStates = 2;
	public double[][] distribution = null;
	public int attIndex = -1;
	public double precision = 0.01;// only useful for real values

	public Potential() {
	}

	/**
	 * Prints out the distribution
	 * 
	 * @return the string representing the distribution
	 */
	@Override
	public String toString() {
		String s = "";
		int nC = distribution[0].length;
		for (int c = 0; c < nC; c++) {
			s += "Class " + c + " : ";
			for (int v = 0; v < numStates; v++) {
				s += distribution[v][c] + "\t";
			}
			s += "\n";
		}
		s += "Precision: " + precision + "\n";

		return s;
	}
}
// end of class Potential

