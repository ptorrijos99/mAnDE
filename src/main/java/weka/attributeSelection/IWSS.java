package weka.attributeSelection;

import java.beans.BeanDescriptor;
import java.beans.EventSetDescriptor;
import java.beans.PropertyDescriptor;
import java.util.ArrayList;
import java.util.Enumeration;
import java.util.Vector;
import java.util.Random;

import weka.core.Instances;
import weka.core.Option;
import weka.core.OptionHandler;
import weka.core.SelectedTag;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformationHandler;
import weka.core.Utils;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;
import weka.attributeSelection.SymmetricalUncertAttributeEval;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;
import weka.gui.beans.ChartListener;
import weka.gui.beans.ConfigurationListener;
import weka.gui.beans.DataSource;
import weka.gui.beans.DataSourceListener;
import weka.gui.beans.FilterCustomizer;
import weka.gui.beans.IncrementalClassifierEvaluator;
import weka.gui.beans.InstanceListener;
import weka.gui.beans.TestSetListener;
import weka.gui.beans.TestSetProducer;
import weka.gui.beans.TextListener;
import weka.gui.beans.TrainingSetListener;
import weka.gui.beans.TrainingSetProducer;
import weka.attributeSelection.AttributeEvaluator;

/**
 * <!-- globalinfo-start --> IWSS:<br/>
 * <br/>
 * This attribute selector is specially designed to handle high-dimensional datasets.
 * It first creates a ranking of attributes based on the selected metric, and then it
 * runs an Incremental Wrapper Subset Selection over the ranking (linear complexity)
 * by selecting attributes (using the WrapperSubsetEval class) which improve the 
 * performance for a given minimum number of folds out of the folds of the the wrapper
 * cross-validation. It contains the theta option which permits to tune an early stopping
 * (sublinear complexity).It contains the replaceSelection option, which tests at each 
 * step of the incremental search swapping a selected attribute by the current candidate,
 *  this reduces the mean number of selected attributes without decreasing performance but
 *  it increases the linear complexity to quadratic.
 <br/>
 * <br/>
 * For more information see:<br/>
 * <br/>
 * Pablo Bermejo, Jos{\'e} A. G{\'a}mez and Jos{\'e},(2011). Improving
 * Incremental Wrapper-Based Subset Selection via Replacement and Early
 * Stopping. International Journal of Pattern Recognition and Artificial
 * Intelligence. 25(5):605-625.
 * <p/>
 * <!-- globalinfo-end -->
 * 
 * <!-- technical-bibtex-start --> BibTeX:
 * 
 * <pre>
 * @article{Bermejo-IncrementalSearch,
 *   author    = {Pablo Bermejo and
 *                Jos{\'e} A. G{\'a}mez and
 *                Jos{\'e} Miguel Puerta},
 *   title     = {Improving Incremental Wrapper-Based Subset Selection via
 *                Replacement and Early Stopping},
 *   journal   = {IJPRAI},
 *   volume    = {25},
 *   number    = {5},
 *   year      = {2011},
 *   pages     = {605-625},
 * 
 * }
 * </pre>
 * <p/>
 * <!-- technical-bibtex-end -->
 * 
 * <!-- options-start --> Valid options are:
 * <p/>
 * 
 * <pre>
 *  -minFolds <num>
 *  minimum number of folds whose wrapper goodness must be improved in the inner cross-validation when selecting a new attribute.
 *  (default 2)
 * </pre>
 * 
 * <pre>
 *  -theta <num>
 *   the value to tune the early stopping procedure. Range is (0-1]. When t=1, early stopping is turned off.
 *  (default 1)
 * </pre>
 * 
 * <pre>
 *  -rankingMetric <num>
 *   AttributeEvaluator to use for the creation of the ranking over which IWSS will be run.
 *  (default weka.attributeSelection.SymmetricalUncertAttributeEval)
 * </pre>
 * 
 * *
 * 
 * <pre>
 *   -replaceSelection
 *   Flag to activate the option of testing, at each step of the incremental search, the swapping of
 *   a selected feature with another not selected yet. This increases the worst-case theoretical complexity
 * from linear to quadratic.
 * 
 * <pre>
 * 
 * 	<!-- options-end -->
 * 
 * @author Pablo Bermejo (Pablo.Bermejo@uclm.es)
 * @version $Revision: 1.0.0 $
 */

public class IWSS extends ASSearch implements OptionHandler, StartSetHandler,
		TechnicalInformationHandler {

	private static final long serialVersionUID = -1119139262589317876L;

	/****************** OPTIONS **********************/

	/**
	 * Minimum number of folds in inner cross-validation whose wrapper goodness
	 * must be improved when selecting a new attribute.
	 */
	protected int m_minFolds = 2;

	/** Single Attribute evaluator used to create the ranking */
	protected ASEvaluation m_rankingMetric = new SymmetricalUncertAttributeEval();

	/**
	 * Value for early stopping: (0-1] percentage of remaining attributes to
	 * visit. Updated each time an attribute is selected.
	 */
	protected double m_theta = 1.0;

	/**
	 * Flag to activate the option of testing, at each step of the incremental
	 * search, the swapping of a selected feature with another not selected yet.
	 * This increases the worst-case theoretical complexity from linear to
	 * quadratic.
	 */
	protected boolean m_replaceSelection = false;

	/*********** OPTIONS end here **********************/

	/** The training set for inner cross validation */
	protected ArrayList<Instances> m_trainingData;

	/** The test set for inner cross validation */
	protected ArrayList<Instances> m_testData;

	/**
	 * Array which keeps the best cross-validation so far in the incremental
	 * search
	 */
	protected double[] m_bestGoodness;

	/** The dataset */
	protected Instances m_dataset = null;

	/** Selected attributes */
	protected ArrayList<Integer> m_selected;

	/** Number of selected attributes (first in ranking is selected by default) */
	protected int m_numSelected = 1;

	/**
	 * Number of former attributes in ranking to be straight-forward selected
	 * before running the incremental search.
	 */
	protected int m_formerSelected = 1;

	/** It is set to true if IWSS.setStartSet is called */
	protected boolean m_skipRanking = false;

	/** Number of folds in the inner cross-validation */
	protected static int m_numFolds;

	/**
	 * The constructor
	 * 
	 * @param data
	 *            Instances object to initiate global object dataset
	 */
	public IWSS() throws Exception {

		resetOptions();

	}

	/**
	 * Reset all options to their default values
	 */
	public void resetOptions() {

		m_minFolds = 2;
		m_theta = 1.0;
		m_rankingMetric = new SymmetricalUncertAttributeEval();
		m_replaceSelection = false;

	}

	/**
	 * Ranks attributes based on a bivariated (X;Class) metric set in
	 * m_rankingMetric
	 * 
	 * @param data
	 *            Instances with attributes to rank
	 * @return integer vector with attributes ordered from max to min value of
	 *         the filter metric
	 * @exception if
	 *                something goes wrong during ranking
	 */
	public int[] getRanking(Instances data) throws Exception {
		int[] ranking = new int[data.numAttributes() - 1];

		m_rankingMetric.buildEvaluator(data);

		double[] values = new double[data.numAttributes() - 1];
		for (int i = 0; i < data.numAttributes() - 1; i++) {
			values[i] = ((AttributeEvaluator) m_rankingMetric)
					.evaluateAttribute(i);
		}

		int[] aux = Utils.stableSort(values);
		for (int i = 0; i < data.numAttributes() - 1; i++) {
			ranking[i] = aux[aux.length - i - 1];
		}

		return ranking;
	}

	public boolean getReplaceSelection() {
		return m_replaceSelection;
	}

	/**
	 * Prepares train and test data for evaluation and counting how many folds
	 * get higher wrapper goodness than folds of evaluation with the up to now
	 * best features subset.
	 * 
	 */
	protected void prepareFolds(int seed) {
		m_trainingData = new ArrayList<Instances>(m_numFolds);
		m_testData = new ArrayList<Instances>(m_numFolds);
		Instances d = null;
		Random random = new Random();
		random.setSeed(seed);
		for (int i = 0; i < m_numFolds; i++) {
			d = m_dataset.trainCV(m_numFolds, i, random);
			m_trainingData.add(d);
			d = m_dataset.testCV(m_numFolds, i);
			m_testData.add(d);
		}
	}

	/**
	 * Get the goodness of a given classifier over the passed data set
	 * 
	 * 
	 * @return goodness of the wrapper metric set in options
	 */
	private double getGoodness(Evaluation eval, SelectedTag eval_tag)
			throws Exception {
		double goodness = eval.pctCorrect();
		// "1-eval.[error]" is used because comparisons of goodness are
		// performed with the > operator
		switch (eval_tag.getSelectedTag().getID()) {
		case WrapperSubsetEval.EVAL_DEFAULT:
			goodness = 1 - eval.errorRate();
			break;
		case WrapperSubsetEval.EVAL_ACCURACY:
			goodness = 1 - eval.errorRate();
			break;
		case WrapperSubsetEval.EVAL_RMSE:
			goodness = 1 - eval.rootMeanSquaredError();
			break;
		case WrapperSubsetEval.EVAL_MAE:
			goodness = 1 - eval.meanAbsoluteError();
			break;
		case WrapperSubsetEval.EVAL_FMEASURE:
			goodness = eval.weightedFMeasure();
			break;
		case WrapperSubsetEval.EVAL_AUC:
			goodness = eval.weightedAreaUnderROC();
			break;
		case WrapperSubsetEval.EVAL_AUPRC:
			goodness = eval.weightedAreaUnderPRC();
			break;
		}

		return goodness;
	}

	/**
	 * Projects the given dataset over the passed attributes and class
	 * 
	 * @param data
	 *            Instances object to project
	 * @param subset
	 *            List of attributes to keep in projection. Class is
	 *            automatically added
	 * @return new Instances object projected.
	 */
	private Instances getProjection(Instances data, ArrayList<Integer> subset)
			throws Exception {
		Instances projection = null;

		int atts[] = new int[subset.size() + 1];
		for (int j = 0; j < subset.size(); j++) {
			atts[j] = subset.get(j);
		}
		atts[subset.size()] = data.classIndex();

		Remove filter = new Remove();
		filter.setAttributeIndicesArray(atts);
		filter.setInvertSelection(true);
		filter.setInputFormat(data);
		projection = new Instances(filter.getOutputFormat());
		projection = Filter.useFilter(data, filter);

		projection.setClassIndex(subset.size());

		return projection;
	}

	/**
	 * 
	 * @return the indexes of selected attribute after the IWSS search
	 */
	public int[] getSelected() {

		int[] s = new int[m_selected.size()];
		for (int i = 0; i < m_selected.size(); i++) {
			s[i] = m_selected.get(i);
		}

		return s;
	}

	/**
	 * Returns an enumeration describing the available options.
	 * 
	 * @return an enumeration of all the available options.
	 **/
	@Override
	public Enumeration listOptions() {

		Vector<Option> newVector = new Vector<Option>(3);

		newVector
				.addElement(new Option(
						"\tMin number of folds to be better in order to accept the new variable\n",
						"minFolds", 1, "-minFolds <num>"));

		newVector
				.addElement(new Option(
						"\t>0 theta < 1 for Eary Stopping. Percentage of remaining attributes to test without acceptance\n",
						"theta", 1, "-theta <num>"));

		newVector
				.addElement(new Option(
						"\tAttributeEvaluator to use when creating the ranking of attributes on which the IWSS search will be run\n",
						"rankingMetric", 1,
						"-rankingMetric <weka.attributeSelection.AttributeEvaluator>"));

		if ((m_rankingMetric != null)
				&& (m_rankingMetric instanceof OptionHandler)) {
			newVector.addElement(new Option("", "", 0,
					"\nOptions specific to scheme "
							+ m_rankingMetric.getClass().getName() + ":"));
			Enumeration<Option> enu = ((OptionHandler) m_rankingMetric)
					.listOptions();

			while (enu.hasMoreElements()) {
				newVector.addElement(enu.nextElement());
			}
		}

		return newVector.elements();

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
	 * Returns the tip text for this property
	 * 
	 * @return tip text for this property suitable for displaying in the
	 *         explorer/experimenter gui
	 */
	public String minFoldsTipText() {
		return "Min number of folds in inner CV to be improved for new attribute acceptance.";
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
		return "> 0 and < 1 for use of Early Stopping.Percentage of remaining attributes to test without acceptance ";
	}

	/**
	 * Activate the option of testing, at each step of the incremental search, the swapping of
	 *   a selected feature with another not selected yet. This increases the worst-case theoretical complexity
	 *   from linear to quadratic.
	 * @param r
	 */
	public void setReplaceSelection(boolean r) {
		m_replaceSelection = r;
	}
	
	public String replaceSelectionTipText(){
		return "Test, at each step of the incremental search, the swapping of a "
		+"selected feature with another not selected yet. This increases the worst-case "
		+"theoretical complexity of IWSS from linear to quadratic.";
		
	}

	/**
	 * 
	 * @param AE
	 *            the attribute evaluator (X;Class) used for building the
	 *            ranking
	 */
	public void setRankingMetric(ASEvaluation AE) throws Exception {
		if (!(AE instanceof AttributeEvaluator))
			throw new Exception(
					"The metric used for building the ranking must be a single Attribute Evaluator (e.g. InfoGain), not a subset evaluator (e.g. Cfs)");
		m_rankingMetric = AE;
	}

	/**
	 * Returns the tip text for this property
	 * 
	 * @return tip text for this property suitable for displaying in the
	 *         explorer/experimenter gui
	 */
	public String rankingMetricTipText() {
		return "The weka.attributeSelection.AttributeEvaluator to use when building the univariate ranking on which IWSS will be run. Do not choose subset evaluators, just single attribute evaluators.";
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
	 *   the value to tune the early stopping procedure. Range is (0-1]. When t=1, early stopping is turned off.
	 *  (default 1)
	 * </pre>
	 * 
	 * <pre>
	 *  -rankingMetric <num>
	 *   AttributeEvaluator to use for the creation of the ranking over which IWSS will be run.
	 *  (default weka.attributeSelection.SymmetricalUncertAttributeEval)
	 * </pre>
	 * 
	 * <pre>
	 *   -replaceSelection
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
	 * @param AttributeEvaluator 
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

		selectionString = Utils.getOption("rankingMetric", options);
		if (selectionString.length() != 0){
			String[] evalSpec=Utils.splitOptions(selectionString);
			String evalname=evalSpec[0];
			evalSpec[0]="";
			setRankingMetric(ASEvaluation.forName(evalname, evalSpec));
		}
		

		m_replaceSelection = Utils.getFlag("replaceSelection", options);

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
	 * @return The attribute evaluator used for building the ranking of
	 *         attributes over which the IWSS will be run
	 */

	public ASEvaluation getRankingMetric() {
		return m_rankingMetric;
	}

	/**
	 * get a String[] describing the value set for all options
	 * 
	 * @return String[] describing the options
	 */
	public String[] getOptions() {
		int length = 6;

	    if (m_replaceSelection)
			length++;

		int current = 0;

		String[] options = new String[length];
		options[current++] = "-minFolds";
		options[current++] = "" + m_minFolds;
		options[current++] = "-theta";
		options[current++] = "" + m_theta;

		if (m_replaceSelection)
			options[current++] = "-replaceSelection";

		options[current++]= "-rankingMetric";
		options[current++]=getRankingMetricSpec();
			

		return options;
	}

	/**
	 * Performs an incremental wrapper subset selection, as described in the
	 * referenced paper, over a filter-ranking. Attributes are selected if they
	 * improve the average wrapper goodness not only in the mean of
	 * cross-validation, but in m_minFolds out of m_numFolds of
	 * cross-validation.
	 * 
	 * @return indexes of attributes in data selected after IWSS search
	 * @throws exception
	 *             is something goes wrong
	 */
	@Override
	public int[] search(ASEvaluation ASEvaluator, Instances data)
			throws Exception {
		if (!(ASEvaluator instanceof WrapperSubsetEval)) {
			throw new Exception(ASEvaluator.getClass().getName() + " is not a "
					+ "WrapperSubsetEval!");
		}

		// init variables
		m_numFolds = ((WrapperSubsetEval) ASEvaluator).getFolds();
		m_bestGoodness = new double[m_numFolds];
		m_selected = new ArrayList<Integer>();
		m_dataset = new Instances(data, 0, data.numInstances() - 1);
		m_dataset.stratify(m_numFolds);
		for (int i = 0; i < m_numFolds; i++) {
			m_bestGoodness[i] = -1.0;
		}

		// prepare data
		int[] ranking = doRanking();
		prepareFolds(((WrapperSubsetEval) ASEvaluator).getSeed());

		// set attributes to be selected beforehand
		for (int i = 0; i < m_formerSelected; i++)
			m_selected.add(ranking[i]);
		m_numSelected = m_formerSelected;

		// compute first best wrapper goodness
		computeWrapperGoodness(m_bestGoodness,
				((WrapperSubsetEval) ASEvaluator).getClassifier(),
				((WrapperSubsetEval) ASEvaluator).getEvaluationMeasure());
		// bestAvgGoodness = weka.core.Utils.mean(m_bestGoodness);

		// compute early stopping
		int toVisit = (int) (ranking.length * m_theta);
		
		if(!m_replaceSelection) runIWSS(toVisit, ranking, ASEvaluator);
		else runIWSSreplace(toVisit, ranking, ASEvaluator);

		return getSelected();
	}

	protected void runIWSS(int toVisit, int[] ranking, ASEvaluation ASEvaluator)
			throws Exception {

		double[] currentGoodness = new double[m_numFolds];
		// incremental search
		for (int i = m_formerSelected; i < toVisit; i++) {

			m_selected.add(ranking[i]);
			int foldsImproved = computeWrapperGoodness(currentGoodness,
					((WrapperSubsetEval) ASEvaluator).getClassifier(),
					((WrapperSubsetEval) ASEvaluator).getEvaluationMeasure());

			double currentAvgGoodness = Utils.mean(currentGoodness);
			double bestAvgGoodness = Utils.mean(m_bestGoodness);
			if ((currentAvgGoodness > bestAvgGoodness)
					&& (foldsImproved >= m_minFolds)) {
				for (int j = 0; j < m_numFolds; j++) {
					m_bestGoodness[j] = currentGoodness[j];
				}
				bestAvgGoodness = currentAvgGoodness;
				m_numSelected++;
				toVisit = i + (int) (m_theta * (ranking.length - i));
			} else {
				m_selected.remove(m_selected.size() - 1);
			}

		}

	}
	
	protected void runIWSSreplace(int toVisit, int[] ranking, ASEvaluation ASEvaluator)
	throws Exception {
		
		double[] currentGoodness = new double[m_numFolds];
		double[] bestItAcc=new double[m_numFolds];
		int bestMove = -1; // -1 means no improvement, 0, 1, ... means replacement
		
		
		for (int i = m_formerSelected; i < toVisit; i++) {
			for (int f = 0; f < m_numFolds; f++) {
				bestItAcc[f] = -1.0;
			}
			double bestItAvgAcc = -1.0;
			bestMove = -1;

			// first we test addition
			m_selected.add(ranking[i]);

			int foldsImproved = computeWrapperGoodness(currentGoodness,
					((WrapperSubsetEval) ASEvaluator).getClassifier(),
					((WrapperSubsetEval) ASEvaluator).getEvaluationMeasure());
			
			double currentAvgGoodness = Utils.mean(currentGoodness);
			double bestAvgGoodness = Utils.mean(m_bestGoodness);
			
			if ((currentAvgGoodness > bestAvgGoodness) && (foldsImproved >= m_minFolds)) {
				for (int j = 0; j < m_numFolds; j++) {
					bestItAcc[j] = currentGoodness[j];
				}
				bestItAvgAcc = currentAvgGoodness;
				bestMove = m_selected.size();
			}

			// testing swapping
			int replacedVar;
			for (int p = 0; p < m_selected.size() - 1; p++) {
				replacedVar = m_selected.remove(p);
				foldsImproved = computeWrapperGoodness(currentGoodness,
						((WrapperSubsetEval) ASEvaluator).getClassifier(),
						((WrapperSubsetEval) ASEvaluator).getEvaluationMeasure());
				currentAvgGoodness =weka.core.Utils.mean(currentGoodness);
				
				if ((currentAvgGoodness > bestAvgGoodness)
						&& (foldsImproved >= m_minFolds)) {// is a candidate
					if (currentAvgGoodness > bestItAvgAcc) {// in fact is the best
														// candidate up to now!!
						for (int j = 0; j < m_numFolds; j++) {
							bestItAcc[j] = currentGoodness[j];
						}
						bestItAvgAcc = currentAvgGoodness;
						bestMove = p;
					}
				}

				m_selected.add(p, replacedVar);

			}// end of "swapping" for

			if (bestMove != -1) {
				for (int j = 0; j < m_numFolds; j++) {
					m_bestGoodness[j] = bestItAcc[j];
				}
				bestAvgGoodness = bestItAvgAcc;
				if (bestMove == m_selected.size()) {// addition
					m_numSelected++;
				} else { // replacement
					m_selected.remove(bestMove);
				}
				toVisit = i + (int) (m_theta * (ranking.length - i));
			} else {// discarding the studied attribute
				m_selected.remove(m_selected.size() - 1);
			}

		}
		
		
	}

	/**
	 * 
	 * @return ranking over which to perform incremental search
	 */
	protected int[] doRanking() {
		int[] r = null;

		if (!m_skipRanking)
			try {
				r = getRanking(m_dataset);
			} catch (Exception e) {
				e.printStackTrace();
			}
		else {
			r = new int[m_dataset.numAttributes() - 1];
			for (int i = 0; i < r.length; i++)
				r[i] = i;
		}

		return r;
	}

	/**
	 * weka.classifiers.Evaluation cannot be used because IWSS needs to keep
	 * track of the wrapper goodness for each fold of the inner cross
	 * validation.
	 * 
	 * @param storeGoodness
	 *            array in which to store the goodness computed for each fold
	 * @param tempClassifier
	 *            model to use for classification
	 * @return the number of folds in cross validation which obtained better
	 *         goodness than before selecting the last attribute
	 */
	protected int computeWrapperGoodness(double[] storeGoodness,
			Classifier tempClassifier, SelectedTag eval_tag) throws Exception {
		Instances projectedTest, projectedTrain;
		int improvement = 0;
		for (int j = 0; j < m_numFolds; j++) {

			projectedTrain = getProjection(m_trainingData.get(j), m_selected);
			tempClassifier.buildClassifier(projectedTrain);
			projectedTest = getProjection(m_testData.get(j), m_selected);

			Evaluation eval = new Evaluation(projectedTrain);
			eval.evaluateModel(tempClassifier, projectedTest);
			storeGoodness[j] = getGoodness(eval, eval_tag);
			if (storeGoodness[j] > m_bestGoodness[j]) {
				improvement++;
			}
		}
		return improvement;
	}

	/**
	 * Setting a start set instancies m_skipRanking=true. Since this is an
	 * incremental search on a ranking, a startset can only be a set of
	 * consecutive numbers from 0 to N. When using this method, it is supposed
	 * that attributes in dataset are ranked by decreasing order; that is,
	 * attribute 0 is better thank attribute 1 in m_dataset.
	 * 
	 * 
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

	@Override
	public TechnicalInformation getTechnicalInformation() {
		TechnicalInformation result;

		result = new TechnicalInformation(Type.ARTICLE);
		result.setValue(Field.AUTHOR,
				"Pablo Bermejo and Jose A. Gamez and Jose M. Puerta");
		result.setValue(Field.YEAR, "2011");
		result.setValue(
				Field.TITLE,
				"Improving Incremental Wrapper-Based Subset Selection via Replacement and Early Stopping");
		result.setValue(Field.JOURNAL,
				"International Journal of Pattern Recognition and Artificial Intelligence");
		result.setValue(Field.VOLUME, "25");
		result.setValue(Field.NUMBER, "5");
		result.setValue(Field.PAGES, "605-625");
		result.setValue(Field.ISSN, "0218-0014");

		return result;
	}

	/**
	 * Returns a string describing this search algorithm
	 * 
	 * @return a description of the search algorithm suitable for displaying in
	 *         the explorer/experimenter gui
	 */
	public String globalInfo() {
		return "IWSS:\n\n"
				+ "This attribute selector is specially designed to handle high-dimensional datasets.\n"
				 + "It first creates a ranking of attributes based on the selected metric, and then it "
				 + "runs an Incremental Wrapper Subset Selection over the ranking (linear complexity) "
				 + "by selecting attributes (using the WrapperSubsetEval class) which improve the " 
				 + "performance for a given minimum number of folds out of the folds of the the wrapper "
				 + "cross-validation. It contains the theta option which permits to tune an early stopping "
				 + "(sublinear complexity).It contains the replaceSelection option, which tests at each " 
				 + "step of the incremental search swapping a selected attribute by the current candidate, "
				 + "this reduces the mean number of selected attributes without decreasing performance but "
				 + "it increases the linear complexity to quadratic.\n\n" 
				 + getTechnicalInformation().toString();
	}

	@Override
	public String toString() {
		String s = "";
		s += "Incremental Wrapper Subset Selection (IWSS):\n";
		s += "Selected Attributes: " + m_numSelected + "\n";
		s += "Merit of best subset found: " + Utils.mean(m_bestGoodness) + "\n";
		s += "Metric used for creating the ranking: "
				+ m_rankingMetric.getClass().getName() + "\n";
		if (m_theta < 1)
			s += "Early Stopping was used: theta=" + m_theta + "\n";
		if (m_replaceSelection)
			s += "Replace selection was used\n";

		return s;
	}
	
	 /**
	   * Gets the evaluator specification string, which contains the class name of
	   * the attribute evaluator and any options to it
	   *
	   * @return the evaluator string.
	   */
	  protected String getRankingMetricSpec() {
	    
	    ASEvaluation e = m_rankingMetric;
	    if (e instanceof OptionHandler) {
	      return e.getClass().getName() + " "
		+ Utils.joinOptions(((OptionHandler)e).getOptions());
	    }
	    return e.getClass().getName();
	  }

} // end of class


	  
	  


