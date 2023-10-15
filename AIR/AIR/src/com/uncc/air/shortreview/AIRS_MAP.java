package com.uncc.air.shortreview;

import com.lhy.tool.ToolException;
import com.lhy.tool.util.Utils;
import com.uncc.topicmodel.data.Dataset;
import com.uncc.topicmodel.data.Dictionary;
import com.uncc.topicmodel.data.Document;
import com.uncc.topicmodel.TopicModelException;
import com.uncc.air.AIRConstants;
import com.uncc.air.AIRException;
import com.uncc.air.ErrorMessage;
import com.uncc.air.Estimator;
import com.uncc.air.Model;
import com.uncc.air.ParamManager;
import com.uncc.air.data.AIRDataset;
import com.uncc.air.general.eval.Evaluation;
import com.uncc.air.optimizer.SimpleConstrainGradientOptimizer;
import com.uncc.air.util.DirichletParamsEstimator;
import com.uncc.air.util.Measure;

import java.io.BufferedWriter;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Properties;

import org.apache.commons.math3.special.Gamma;
import org.apache.commons.math3.util.FastMath;

/**
 * @author Huayu Li
 */
public class AIRS_MAP extends Model implements
				Dictionary.DictionaryListener, Estimator {

	private static final boolean DEBUG_DOC         = false;
	private static final boolean SHIFT             = false;
	private static final boolean LOGLIKELIHOOD_ABS = false;

	private static final String FILE_NAME_SHORT_REV = "AIRS_MAP";

	private static final double THRESHOLD_EM          = 1.0e-4;
	private static final double THRESHOLD_THETA       = 1.0e-5;
	private static final double THRESHOLD_THETA_OMEGA = 1.0e-5;
	private static final double THRESHOLD_ALPHA_EST   = 1.0e-5;
	private static final double MAX_EM_ITER_NUM       = 200;

	private static final String STRING_EM          =
			"[EM] Iter = %s, logLikelihood = %s, perplexity = %s, logLikelihoodError = %s.";
	private static final String STRING_THETA_OMEGA =
			"\t\t[E-Step Optimize Theta-Omega] Iter = %s, omegaError = %s, thetaError = %s, omega = %s, theta = %s";
	private static final String STRING_ALPHA_EST   =
			"[Alpha Estimation] AlphaEstIter = %s, logLikelihoodError = %s, Alpha = %s";

	private int maxAlphaEstIterNum = 5;

	private double      LAMBDA;
	private double[][]  LAMBDA_HAT;
	private double[]    GAMMAS;
	private double[]    alphas;
	private double      betaInit;

	private double[][]   pt;
	private double[][]   theta;
	private double[][][] omega;
	private double[][][] phi;
	private ArrayList<double[][][]> posterior;
	private ArrayList<double[][]> posteriorSum;
	private File   outputPath;
	private File   groundtruthScoreFile;
	private File   groundtruthDisFile;
	private LeftToRightEMEvaluation validationEval;
	private LeftToRightEMEvaluation testEval;

	public AIRS_MAP(ParamManager paramManager)
				throws AIRException, TopicModelException {
	
		initNewModel(paramManager);
	}

	@Override
	public void dictionaryChanged() {
		Utils.println("DICTIONARY CHANGED.");

		try {
			initBeta(betaInit, keywords);

			double[][][] phiNew = new double[SENTIMENT_NUM][TOPIC_NUM][dictionary.getSize()];
			for (int sentiment = 0; sentiment < SENTIMENT_NUM; sentiment ++) {
				for (int topic = 0; topic < TOPIC_NUM; topic ++) {
					for (int word = 0; word < phi[sentiment][topic].length; word ++) {
						phiNew[sentiment][topic][word] = phi[sentiment][topic][word];
					}
				}
			}
			phi = phiNew;
		} catch (AIRException e) {
			e.printStackTrace();
			System.exit(-1);
		}
	}

	@Override
	public String getPrefix() {
		return "AIRS_MAP_%s";
	}

	@Override
	public Properties getParameters() {
		Properties props = new Properties();

		props.setProperty(AIRConstants.STRING_DOCUMENT_NUM,        String.valueOf(data.getDocumentSize()));
		props.setProperty(AIRConstants.STRING_TOPIC_NUM,           String.valueOf(TOPIC_NUM));
		props.setProperty(AIRConstants.STRING_DIC_NUM,             String.valueOf(dictionary.getSize()));
		props.setProperty(AIRConstants.STRING_DIC_NUM,             String.valueOf(dictionary.getSize()));
		props.setProperty(AIRConstants.STRING_BETA_INIT,           String.valueOf(betaInit));
		props.setProperty(AIRConstants.STRING_GAMMAS,              String.valueOf(Utils.convertToString(GAMMAS, ",")));
		props.setProperty(AIRConstants.STRING_LAMBDA,              String.valueOf(LAMBDA));
		props.setProperty(AIRConstants.STRING_MAX_EM_ITER_NUM,     String.valueOf(MAX_EM_ITER_NUM));
		props.setProperty(AIRConstants.STRING_EM_THRESHOLD,        String.valueOf(THRESHOLD_EM));
		props.setProperty(AIRConstants.STRING_MAX_ALPHA_EST_I_NUM, String.valueOf(maxAlphaEstIterNum));
		props.setProperty("ThetaThreshold",                        String.valueOf(THRESHOLD_THETA));
		props.setProperty("OmegaThetaThreshold",                   String.valueOf(THRESHOLD_THETA));
		props.setProperty(AIRConstants.STRING_TOKEN_NUM,           String.valueOf(data.getTermNum()));
		props.setProperty(AIRConstants.STRING_MODEL_OUTPUT_PATH,   AIRConstants.PATH_OUTPUT.getAbsolutePath());
		props.setProperty(AIRConstants.STRING_KEYWORD_FILE,        keywords == null ? "" : "YES");

		props.putAll(data.getParams());

		return props;
	}

	@Override
	public void estimate() throws ToolException, AIRException {
		getParameters().list(System.out);

		BufferedWriter perpWriter = null;
		double logLikelihoodError = Double.MAX_VALUE;
		double logLikelihoodOld   = Double.MAX_VALUE;
		int    alphaEstIter       = 0;

		try {
			perpWriter = Utils.createBufferedWriter(new File(
					outputPath, AIRConstants.FILE_NAME_PERPLEXITY));
			Utils.writeAndPrint(perpWriter, String.format("[" + FORMAT_MODEL_NAME + "]",
					LAMBDA, GAMMAS[0], GAMMAS[1], TOPIC_NUM), true);
		} catch (IOException e) {
			e.printStackTrace();
		}

		do {
			EM(perpWriter);

			if (validationEval != null) {
				double logLikelihood = validationEval.modelLogLikelihood();
				if (logLikelihoodOld != Double.MAX_VALUE) {
					if (! LOGLIKELIHOOD_ABS) {
						logLikelihoodError = (logLikelihoodOld - logLikelihood) / logLikelihoodOld;
					} else {
						logLikelihoodError = Math.abs((logLikelihoodOld - logLikelihood) / logLikelihoodOld);
					}
				}
				logLikelihoodOld = logLikelihood;
				try {
					Utils.writeAndPrint(perpWriter,
						String.format("[Validation][%s] logLikelihood = %s, perplexity = %s",
								validationEval.getTitle(),
								logLikelihood,
								validationEval.perplexity(logLikelihood)),
						true);
				} catch (IOException e) {
					e.printStackTrace();
				}
			}

			if ((alphaEstIter + 1) < maxAlphaEstIterNum) {
				alphas = DirichletParamsEstimator.estimate(
						theta, TOPIC_NUM, 1.0e-5, outputPath, 0);
			}

			try {
				Utils.writeAndPrint(perpWriter, String.format(
						STRING_ALPHA_EST,
						alphaEstIter, logLikelihoodError,
						Utils.convertToString(alphas, ",")),
							true);
			} catch (IOException e) {
				e.printStackTrace();
			}
			alphaEstIter ++;
		} while (alphaEstIter < maxAlphaEstIterNum &&
					logLikelihoodError > THRESHOLD_ALPHA_EST);

		if (testEval != null) {
			double logLikelihood = testEval.modelLogLikelihood();
			try {
				Utils.writeAndPrint(perpWriter,
					String.format("[Test][%s] logLikelihood = %s, perplexity = %s",
							testEval.getTitle(),
							logLikelihood,
							testEval.perplexity(logLikelihood)),
					true);
			} catch (IOException e) {
				e.printStackTrace();
			}
		}

		Utils.cleanup(perpWriter);
	}

	public double EM(BufferedWriter perpWriter) throws ToolException {
		double logLikelihoodError = Double.MAX_VALUE;
		double logLikelihoodOld   = Double.MAX_VALUE;
		int iter                  = 0;
		do {
			updatePosteriors();
			Mstep();
	
			double logLikelihood = calculateLogLikelihood();
			if (iter != 0) {
				if (! LOGLIKELIHOOD_ABS) {
					logLikelihoodError = (logLikelihoodOld - logLikelihood) / logLikelihoodOld;
				} else {
					logLikelihoodError = Math.abs((logLikelihoodOld - logLikelihood) / logLikelihoodOld);
				}
			}
			logLikelihoodOld = logLikelihood;

			if (perpWriter != null) {
				try {
					Utils.writeAndPrint(perpWriter,
							String.format(STRING_EM,
								iter, logLikelihood,
								calculatePerplexity(logLikelihood),
								logLikelihoodError), true);
				} catch (IOException e) {
					e.printStackTrace();
				}
			}
		} while (logLikelihoodError > THRESHOLD_EM &&
					++ iter < MAX_EM_ITER_NUM);

		storeResult();

		if (groundtruthScoreFile != null || groundtruthDisFile != null) {
			try {
				String seval = Measure.evaluate(
					getScoreFile(outputPath),
					groundtruthScoreFile,
					sumWordProb(phi),
					groundtruthDisFile,
					AIRConstants.CONFIG_MANAGER.getRatingScaler(),
					true, AIRConstants.DEFAULT_MISS_ASPECT_RATING);
				Utils.writeAndPrint(perpWriter, seval, true);
			} catch (ToolException e) {
				e.printStackTrace();
			} catch (IOException e) {
				e.printStackTrace();
			}
		}

		return logLikelihoodOld;
	}

	private void updatePosteriors() {
		posterior    = new ArrayList<double[][][]>();
		posteriorSum = new ArrayList<double[][]>();  
		for (int doc = 0; doc < data.getDocumentSize(); doc ++) {
			Document document  = data.getDocument(doc);
			double[][][] probs = new double[document.wordNum][TOPIC_NUM][SENTIMENT_NUM];
			double[][] probSum = new double[TOPIC_NUM][SENTIMENT_NUM];
			for (int position = 0; position < document.getWordNum(); position ++) {
				probs[position] = getPosteriors(document.words[position],
						phi, theta[doc], pt[doc], omega[doc]);
				for (int topic = 0; topic < TOPIC_NUM; topic ++) {
					for (int sentiment = 0; sentiment < SENTIMENT_NUM; sentiment ++) {
						probSum[topic][sentiment] += probs[position][topic][sentiment];
					}
				}
			}
			posterior.add(probs);
			posteriorSum.add(probSum);
		}
	}

	private void Mstep() throws ToolException {
		int debug = 0;

		for (int doc = 0; doc < data.getDocumentSize(); doc ++) {
	
			double[] posteriorSentimentSum = new double[TOPIC_NUM];
			for (int topic = 0; topic < TOPIC_NUM; topic ++) {
				for (int sentiment = 0; sentiment < SENTIMENT_NUM; sentiment ++) {
					posteriorSentimentSum[topic] += posteriorSum.get(doc)[topic][sentiment];
				}
			}

			inference(data.getDocument(doc).getWordNum(), pt[doc],
					theta[doc], omega[doc], posteriorSum.get(doc),
					posteriorSentimentSum,
					LAMBDA_HAT[doc],
					LAMBDA_HAT[doc][0] + LAMBDA_HAT[doc][1],
					1.0e-2,debug);
			//if (doc % 1000 == 0) Utils.println("doc = " + doc);
		}

		updatePhi(posterior);
	}

	private void inference(int wordNum, double[] pt, double[] theta,
				double[][] omega, double[][] posteriorSum,
				double[] posteriorSentimentSum,
				double[] lambdaHat, double lambdaHatSum,
				 double argTol, int debug) throws ToolException {
		

		optimizeThetaAndOmega(theta, omega, posteriorSum,
					posteriorSentimentSum, lambdaHat,
					lambdaHatSum, argTol, debug);
		updatePt(pt, wordNum, posteriorSum);
	}

	private void optimizeThetaAndOmega(double[] theta, double[][] omega,
			double[][] posteriorSum, double[] posteriorSentimentSum,
				double[] lambdaHat, double lambdaHatSum,
				 double argTol, int debug) throws ToolException {
		double thetaError   = Double.MAX_VALUE;
		double omegaError   = Double.MAX_VALUE;
		double[] thetaOld   = null;
		double[][] omegaOld = null;
		int iter            = 0;
		do {
			thetaOld = Arrays.copyOf(theta, theta.length);
			omegaOld = Utils.copyOf(omega);
	
			updateOmega(omega, posteriorSum, theta, lambdaHat,
					lambdaHatSum);
			optimizeTheta(theta, posteriorSentimentSum, omega,
					lambdaHat, lambdaHatSum, argTol, debug);

			thetaError = Utils.sumOfDiffAbs(theta, thetaOld);
			omegaError = Utils.sumOfDiffAbs(omega, omegaOld);

			iter ++;
		} while (thetaError > THRESHOLD_THETA_OMEGA ||
					omegaError > THRESHOLD_THETA_OMEGA);

		if (DEBUG_DOC) Utils.println(String.format(STRING_THETA_OMEGA,
						iter, omegaError, thetaError,
						Utils.convertToString(omega[0], ","),
						Utils.convertToString(theta, ",")));
	}

	private void optimizeTheta(double[] theta, double[] posteriorSentimentSum,
					double[][] omega, double[] lambdaHat,
						double lambdaHatSum,
						double argTol, int debug) {
		try {
			new ThetaOptimizer(theta , posteriorSentimentSum,
				  omega, lambdaHat, lambdaHatSum, argTol).optimize(debug);
		} catch (AIRException e) {
			e.printStackTrace();
		}
	}

	private void updateOmega(double[][] omega, double[][] posteriorSum,
					double[] theta, double[] lambdaHat,
						double lambdaHatSum) {
		for (int topic = 0; topic < TOPIC_NUM; topic ++) {
			if (SHIFT) {
				omega[topic][0] = (posteriorSum[topic][1] + theta[topic] * lambdaHat[0] - 1.0) /
						(posteriorSum[topic][1] + posteriorSum[topic][2] +
								theta[topic] * lambdaHatSum - 2.0 );
			} else {
				omega[topic][0] = (posteriorSum[topic][1] + theta[topic] * lambdaHat[0]) /
					(posteriorSum[topic][1] + posteriorSum[topic][2] +
							theta[topic] * lambdaHatSum);
			}
			omega[topic][1] = 1 - omega[topic][0];
		}
	}

	private void updatePt(double[] pt, int wordNum, double[][] posteriorSum) {
		pt[0] = 0.0;
		for (int topic = 0; topic < TOPIC_NUM; topic ++) {
			pt[0] += posteriorSum[topic][0];
		}
		if (SHIFT) {
			pt[0] = (pt[0] + GAMMAS[0] - 1) / (wordNum + GAMMAS[0] + GAMMAS[1] - 2.0);
		} else {
			pt[0] = (pt[0] + GAMMAS[0]) / (wordNum + GAMMAS[0] + GAMMAS[1]);
		}
		pt[1] = 1 - pt[0];
	}

	private void updatePhi(ArrayList<double[][][]> posterior) {
		for (int sentiment = 0; sentiment < SENTIMENT_NUM; sentiment ++) {
			Utils.fills(phi[sentiment], 0.0);
			for (int topic = 0; topic < TOPIC_NUM; topic ++) {
				double sum = 0.0;
				for (int doc = 0; doc < data.getDocumentSize(); doc ++) {
					for (int pos = 0; pos < data.getDocument(doc).getWordNum(); pos ++) {
						int word = data.getDocument(doc).words[pos];
						phi[sentiment][topic][word] += posterior.get(doc)[pos][topic][sentiment];
						sum += posterior.get(doc)[pos][topic][sentiment];
					}
				}
				for (int word = 0; word < dictionary.getSize(); word ++) {
					if (SHIFT) {
						phi[sentiment][topic][word] = (phi[sentiment][topic][word] + betas[topic][word] - 1.0) /
							(sum + betaSum[topic] - dictionary.getSize());
					} else {
						phi[sentiment][topic][word] = (phi[sentiment][topic][word] + betas[topic][word]) /
							(sum + betaSum[topic]);
					}
				}
			}
		}
	}

	private double[][] getPosteriors(int word, double[][][] phi,
				double[] theta, double[] pt, double[][] omega) {
		double[][] posterior = new double[TOPIC_NUM][SENTIMENT_NUM];
		double sum           = 0.0;
		for (int topic = 0; topic < TOPIC_NUM; topic ++) {
			for (int sentiment = 0; sentiment < SENTIMENT_NUM; sentiment ++) {
				if (sentiment == 0) {
					posterior[topic][sentiment] =
							phi[sentiment][topic][word] *
							theta[topic] *
							pt[0];
				} else {
					posterior[topic][sentiment] =
							phi[sentiment][topic][word] *
							theta[topic] *
							pt[1] *
							omega[topic][sentiment - 1];
				}
				sum += posterior[topic][sentiment];
			}
		}
		for (int topic = 0; topic < TOPIC_NUM; topic ++) {
			for (int sentiment = 0; sentiment < SENTIMENT_NUM; sentiment ++) {
				posterior[topic][sentiment] /= sum;
			}
		}
		return posterior;
	}

	private double calculatePerplexity(double logLikelihood) {
		return FastMath.exp(-1.0 * logLikelihood / data.getTermNum());
	}

	private double calculateLogLikelihood() {
		double logLikelihood = 0.0;
		for (int doc = 0; doc < data.getDocumentSize(); doc ++) {
			logLikelihood += calculateLogLikelihood(data.getDocument(doc),
					phi, theta[doc], pt[doc], omega[doc]);
		}
		return logLikelihood;
	}

	private double calculateLogLikelihood(Document doc, double[][][] phi,
				double[] theta, double[] pt, double[][] omega) {
		double logLikelihood = 0.0;
		for (int position = 0; position < doc.getWordNum(); position ++) {
			int word       = doc.words[position];
			logLikelihood += FastMath.log(calculateLikelihood(word, phi, theta, pt, omega));
		}
		return logLikelihood;
	}

	private double calculateLikelihood(int word, double[][][] phi,
				double[] theta, double[] pt, double[][] omega) {
		double likelihood = 0.0;
		for (int topic = 0; topic < TOPIC_NUM; topic ++) {
			for (int sentiment = 0; sentiment < SENTIMENT_NUM; sentiment ++) {
				if (sentiment == 0) {
					likelihood += theta[topic] *
							phi[sentiment][topic][word] *
							pt[0];
				} else {
					likelihood +=  theta[topic] *
							phi[sentiment][topic][word] *
							pt[1] *
							omega[topic][sentiment - 1];
				}
			}
		}
		return likelihood;
	}

	private void initNewModel(ParamManager paramManager) throws AIRException,
							TopicModelException {
		parseParamArguments(paramManager);
		dictionary.addDictionaryListener(this);

		initBeta(betaInit, keywords);
		initAlpha();
		initTheta();
		initPhi();
		initPt();
		initOmega();

		if (paramManager.getDataValidationFile() != null) {
			validationEval = new LeftToRightEMEvaluation(
				new AIRDataset(dictionary, paramManager.getDataValidationFile(),
				new File(paramManager.getModelOutputPath(),
						AIRConstants.FILE_NAME_DIC_MAP),
								false));
		}
		if (paramManager.getDataTestFile() != null) {
			testEval = new LeftToRightEMEvaluation(
					new AIRDataset(dictionary, paramManager.getDataTestFile(),
					new File(paramManager.getModelOutputPath(),
						AIRConstants.FILE_NAME_DIC_MAP),
								false));
		}
	}

	private void parseParamArguments(ParamManager paramManager)
					throws AIRException, TopicModelException {
		if (paramManager == null) {
			throw new AIRException(ErrorMessage.ERROR_NO_ARGS);
		}

		// init dictionary
		dictionary = new Dictionary();

		// load data set
		data = new AIRDataset(dictionary, paramManager.getDataTrainFile(),
				new File(paramManager.getModelOutputPath(),
						AIRConstants.FILE_NAME_DIC_MAP),
								false);
		data.getParams().list(System.out);

		if (paramManager.getTopicNum() != null) {
			TOPIC_NUM = paramManager.getTopicNum();
		} else {
			throw new AIRException(ErrorMessage.ERROR_NO_TOPIC_NUM);
		}

		if (paramManager.getTopWordNum() != null) {
			TOP_WORD_NUM = paramManager.getTopWordNum();
		} else {
			throw new AIRException(ErrorMessage.ERROR_NO_TOP_WORD_NUM);
		}

		if (paramManager.getBetaInit() != null) {
			betaInit = shift(paramManager.getBetaInit());
		} else {
			throw new AIRException(ErrorMessage.ERROR_NO_BETA);
		}

		if (paramManager.getGammas() != null) {
			GAMMAS = paramManager.getGammas();

			shift(GAMMAS);
		} else {
			throw new AIRException(ErrorMessage.ERROR_NO_GAMMA);
		}

		if (paramManager.getLambda() != null) {
			LAMBDA     = shift(paramManager.getLambda());
			LAMBDA_HAT = new double[data.getDocumentSize()][2];
			for (int doc = 0; doc < data.getDocumentSize(); doc ++) {
				LAMBDA_HAT[doc][0] = data.getDocument(doc).Ri * LAMBDA;
				LAMBDA_HAT[doc][1] = (1 - data.getDocument(doc).Ri) * LAMBDA;
			}
		} else {
			throw new AIRException(ErrorMessage.ERROR_NO_LAMBDA);
		}

		if (paramManager.getMaxAlphaEstNum() != null) {
			maxAlphaEstIterNum = paramManager.getMaxAlphaEstNum();
		}

		if (paramManager.getModelOutputPath() != null) {
			File base  = new File(paramManager.getModelOutputPath(),
					FILE_NAME_SHORT_REV);
			outputPath = new File(base, String.format(FORMAT_MODEL_NAME,
					LAMBDA, GAMMAS[0], GAMMAS[1], TOPIC_NUM));
		} else {
			throw new AIRException(ErrorMessage.ERROR_NO_OUTPUT_PATH);
		}

		if (paramManager.getKeywordFile() != null) {
			keywords = loadKeywords(paramManager.getKeywordFile(),
								dictionary);
		}

		if (paramManager.getGroundtruthScoreFile() != null) {
			groundtruthScoreFile = paramManager.getGroundtruthScoreFile();
		}

		if (paramManager.getGroundtruthTopicWordDistributionFile() != null) {
			groundtruthDisFile = paramManager.getGroundtruthTopicWordDistributionFile();
		}
	}

	private void initAlpha() {
		alphas = new double[TOPIC_NUM];
		for (int topic = 0; topic < TOPIC_NUM; topic ++) {
			alphas[topic] = 2.0 / TOPIC_NUM;
		}
	}

	private void initTheta() {
		theta = new double[data.getDocumentSize()][TOPIC_NUM];
		for (int doc = 0; doc < data.getDocumentSize(); doc ++) {
			initTheta(theta[doc]);
		}
	}

	private void initTheta(double[] theta) {
		try {
			double alphaSum = Utils.sum(alphas);
			for (int topic = 0; topic < TOPIC_NUM; topic ++) {
				theta[topic] = alphas[topic] / alphaSum;
			}
		} catch (ToolException e) {
			Utils.err("Alpha hasn't been initiated.");
			System.exit(-1);
		}
	}

	private void initPhi() {
		phi = new double[SENTIMENT_NUM][TOPIC_NUM][dictionary.getSize()];
		for (int sentiment = 0; sentiment < phi.length; sentiment ++) {
			for (int topic = 0; topic < phi[sentiment].length; topic ++) {
				for (int word = 0; word < phi[sentiment][topic].length; word ++) {
					if (keywords != null) phi[sentiment][topic][word] = betas[topic][word] / betaSum[topic];
					else phi[sentiment][topic][word] = Math.random() + 1.0 / dictionary.getSize();
				}
				Utils.normalize(phi[sentiment][topic]);
			}
		}
	}

	private void initPt() {
		double mean = GAMMAS[0] / (GAMMAS[0] + GAMMAS[1]);
		pt          = new double[data.getDocumentSize()][2];
		for (int doc = 0; doc < data.getDocumentSize(); doc ++) {
			pt[doc][0] = mean;
			pt[doc][1] = 1 - mean;
		}
	}

	private void initOmega() {
		omega = new double[data.getDocumentSize()][TOPIC_NUM][2];
		for (int doc = 0; doc < data.getDocumentSize(); doc ++) {
			for (int topic = 0; topic < TOPIC_NUM; topic ++) {
				initOmega(data.getDocument(doc), omega[doc]);
			}
		}
	}

	private void initOmega(Document doc, double[][] omega) {
		double mean = GAMMAS[0] / (GAMMAS[0] + GAMMAS[1]);
		for (int topic = 0; topic < TOPIC_NUM; topic ++) {
			omega[topic][0] = (1.0 - mean) * doc.Ri;
			omega[topic][1] = (1.0 - mean) * (1 - doc.Ri);
		}
	}

	private boolean storeResult() {
		boolean res = true;

		File topwordOuputFile = new File(outputPath, format(AIRConstants.FILE_NAME_TOP_WORD));
		File ptOutputFile     = new File(outputPath, format(AIRConstants.FILE_NAME_PARAM_T));
		File thetaOutputFile  = new File(outputPath, format(AIRConstants.FILE_NAME_PARAM_THETA));
		File paramOutputFile  = new File(outputPath, format(AIRConstants.FILE_NAME_PARAM));

		if (! Utils.save(getParameters(), paramOutputFile, true)) {
			Utils.err(ErrorMessage.ERROR_STORE_PARAMS);
			if (res) res = false;
		}
		if (! storeTopWords(topwordOuputFile, false, phi, null)) {
			Utils.err(ErrorMessage.ERROR_STORE_TOP_WORD);
			if (res) res = false;
		}
		if (! storeScore(omega, getScoreFile(outputPath),
				AIRConstants.CONFIG_MANAGER.getRatingScaler(), true)) {
			Utils.err(ErrorMessage.ERROR_STORE_SCORE);
			if (res) res = false;
		}
		if (! Utils.save(pt, AIRConstants.SPACER_TAB,
						ptOutputFile, true)) {
			Utils.err(ErrorMessage.ERROR_STORE_PARAM_T);
			if (res) res = false;
		}
		if (! Utils.save(theta, AIRConstants.SPACER_TAB,
						thetaOutputFile, true)) {
			Utils.err(ErrorMessage.ERROR_STORE_PARAM_THETA);
			if (res) res = false;
		}
		for (int sentiment = 0; sentiment < SENTIMENT_NUM + 1; sentiment ++) {
			File phiOutputFile = new File(outputPath,
						format(AIRConstants.FILE_NAME_PARAM_PHI + "_" + sentiment));
			if (sentiment < SENTIMENT_NUM) {
				if (! Utils.save(phi[sentiment], 
						AIRConstants.SPACER_TAB,
						phiOutputFile, true)) {
					Utils.err(String.format(ErrorMessage.ERROR_STORE_PHI, sentiment));
					if (res) res = false;
				}
			} else {
				if (! Utils.save(sumWordProb(phi),
						AIRConstants.SPACER_TAB,
						phiOutputFile, true)) {
					Utils.err(String.format(ErrorMessage.ERROR_STORE_PHI, sentiment));
					if (res) res = false;
				}
			}
		}

		return res;
	}

	private double shift(double arg) {
		if (SHIFT) return arg + 1.0;
		else return arg;
	}

	private void shift(double[] args) {
		if (args != null && SHIFT) {
			for (int i = 0; i < args.length; i ++) {
				args[i] = shift(args[i]);
			}
		}
	}

	private class ThetaOptimizer extends SimpleConstrainGradientOptimizer {
		private final int MAX_RETRY_COUNT = 2;

		private double[] posteriorSentimentSum;
		private double[][] omega;
		private double[] lambdaHat;
		private double lambdaHatSum;

		public ThetaOptimizer(double[] args,double[] posteriorSentimentSum,
					double[][] omega, double[] lambdaHat,
					double lambdaHatSum, double argTol)
							throws AIRException {
			super(args, THRESHOLD_THETA , argTol);

			this.posteriorSentimentSum = posteriorSentimentSum;
			this.omega                 = omega;
			this.lambdaHat             = lambdaHat;
			this.lambdaHatSum          = lambdaHatSum;

		}
	
		@Override
		public boolean optimize(int debug) {
			boolean resCode = false;
			int retryCount  = 0;
			do {
				try {
					resCode = super.optimize(debug);
					break;
				} catch (ToolException e) {
					e.printStackTrace();
				} catch (AIRException e) {
					if (retryCount > 0)e.printStackTrace();

					/*
					 *  reset the initial value, lbfgs doesn't
					 *  converge may due to the bad initial value
					 */
					if (retryCount < MAX_RETRY_COUNT - 1) {
						for (int i = 0; i < args.length; i ++) {
							args[i] = Math.random();
						}
					}
				}
				retryCount ++;
			}while (retryCount < MAX_RETRY_COUNT);

			return resCode;
		}

		@Override
		public void calculateObjectGradients(double[] objectGradients,
				double[] objectFunction) {
			Arrays.fill(objectGradients, 0.0);
			objectFunction[0] = 0.0;

			for (int topic = 0; topic < TOPIC_NUM; topic ++) {
				double[] theta_x_lambda = new double[2];
				double theta_lambda_sum = 0.0;
				for (int sentiment = 0; sentiment < 2; sentiment ++) {
					theta_x_lambda[sentiment] = args[topic] * lambdaHat[sentiment];
					theta_lambda_sum         += theta_x_lambda[sentiment];
					double logOmega           =  FastMath.log(omega[topic][sentiment]);
					objectGradients[topic]   += (logOmega - Gamma.digamma(theta_x_lambda[sentiment])) *
									lambdaHat[sentiment];
					if (SHIFT) {
						objectFunction[0]        += (theta_x_lambda[sentiment] - 1.0) * logOmega -
									Gamma.logGamma(theta_x_lambda[sentiment]);
					} else {
						objectFunction[0]        += (theta_x_lambda[sentiment]) * logOmega -
									Gamma.logGamma(theta_x_lambda[sentiment]);
					}
							
				}

				if (SHIFT) { 
					objectGradients[topic] += (
							( posteriorSentimentSum[topic] + alphas[topic] - 1.0 ) / args[topic] +
							Gamma.digamma(theta_lambda_sum) * lambdaHatSum
							);
				} else {
					objectGradients[topic] += (
						( posteriorSentimentSum[topic] + alphas[topic]) / args[topic] +
						Gamma.digamma(theta_lambda_sum) * lambdaHatSum
						);
				}
				objectGradients[topic] *= -1.0;

				if (SHIFT) { 
					objectFunction[0] += (posteriorSentimentSum[topic] + alphas[topic] - 1.0) * FastMath.log(args[topic]) +
							+ Gamma.logGamma(theta_lambda_sum);
				} else {
					objectFunction[0] += (posteriorSentimentSum[topic] + alphas[topic]) * FastMath.log(args[topic]) +
							+ Gamma.logGamma(theta_lambda_sum);
				}
			}
			objectFunction[0] *= -1.0;
		}
	}

	private class LeftToRightEMEvaluation extends Evaluation {
		private final static double THRESHOLD_OMEGA = 1.0e-5;
		private final static double THRESHOLD_PT    = 1.0e-5;
		private final static double THRESHOLD_THETA = 1.0e-5;

		private double[][] posteriorSum        = null;
		private double[][] omega               = null;
		private double[] theta                 = null;
		private double[] pt                    = null;
		private double[] posteriorSentimentSum = null;
		private double[] lambdaHat             = null;
		private double lambdaHatSum;

		public LeftToRightEMEvaluation(Dataset dataset) {
			super(dataset, null);
		}

		@Override
		public  String getTitle() {
			return "AIRS EM";
		};

		@Override
		public double modelLogLikelihood (int doc) throws ToolException {
			if (doc % 1000 == 0) printStep(doc);

			double logLikelihood       = 0.0;
			Document document          = testData.getDocument(doc);
			this.omega                 = new double[TOPIC_NUM][2];
			this.pt                    = new double[2];
			this.theta                 = new double[TOPIC_NUM];
			this.posteriorSentimentSum = new double[TOPIC_NUM];
			this.lambdaHat             = LAMBDA_HAT[doc];
			this.lambdaHatSum          = this.lambdaHat[0] + this.lambdaHat[1];

			initTheta(this.theta);

			for (int position = 0; position < document.getWordNum(); position ++) {
				EM(document, position);

				logLikelihood += FastMath.log(calculateLikelihood(
							document.words[position],
							phi, this.theta,
							this.pt,  this.omega));
			}

			return logLikelihood;
		}

		private void EM(Document doc, int position) throws ToolException {
			double ptError      = Double.MAX_VALUE;
			double thetaError   = Double.MAX_VALUE;
			double omegaError   = Double.MAX_VALUE;
			double[] ptOld      = null;
			double[] thetaOld   = null;
			double[][] omegaOld = null;
			int  iter           = 0;
			int debug           = 0;

			do {
				ptOld    = Arrays.copyOf(this.pt, this.pt.length);
				thetaOld = Arrays.copyOf(this.theta, this.theta.length);
				omegaOld = Utils.copyOf(this.omega);

				evaluatePosterior(doc, position);
				inference(position, this.pt, this.theta,
						this.omega,  this.posteriorSum,
						this.posteriorSentimentSum,
						this.lambdaHat, this.lambdaHatSum,
						1.0e-2, debug);

				ptError    = Utils.sumOfDiffAbs(this.pt, ptOld);
				thetaError = Utils.sumOfDiffAbs(this.theta, thetaOld);
				omegaError = Utils.sumOfDiffAbs(this.omega, omegaOld);

				iter ++;
			} while (omegaError > THRESHOLD_OMEGA || ptError > THRESHOLD_PT ||
						thetaError > THRESHOLD_THETA);
		}
	
		private void evaluatePosterior(Document doc, int position) {
			double[][][] posterior      = new double[position][TOPIC_NUM][SENTIMENT_NUM];
			this.posteriorSum           = new double[TOPIC_NUM][SENTIMENT_NUM];
			this.posteriorSentimentSum  = new double[TOPIC_NUM];
			for (int pos = 0; pos < position; pos ++) {
				posterior[pos] = getPosteriors(doc.words[pos],
						phi, this.theta, this.pt, this.omega);
				for (int topic = 0; topic < TOPIC_NUM; topic ++) {
					for (int sentiment = 0; sentiment < SENTIMENT_NUM; sentiment ++) {
						this.posteriorSum[topic][sentiment] += posterior[pos][topic][sentiment];
						this.posteriorSentimentSum[topic]   += posterior[pos][topic][sentiment];
					}
				}
			}
		}
	}
}
