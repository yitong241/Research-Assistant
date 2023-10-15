package com.uncc.air.general;

import com.lhy.tool.ToolException;
import com.lhy.tool.util.Utils;
import com.uncc.air.AIRConstants;
import com.uncc.air.AIRException;
import com.uncc.air.ErrorMessage;
import com.uncc.air.Estimator;
import com.uncc.air.Model;
import com.uncc.air.ParamManager;
import com.uncc.air.data.AIRDataset;
import com.uncc.air.general.eval.EvaluationModel;
import com.uncc.air.general.eval.EvaluationPool;
import com.uncc.air.optimizer.ExpConstraintGradientOptimizer;
import com.uncc.air.util.DirichletParamsEstimator;
import com.uncc.air.util.Measure;
import com.uncc.lda.est.em.LDANewtonRaphson;
import com.uncc.topicmodel.TopicModelException;
import com.uncc.topicmodel.data.Dataset;
import com.uncc.topicmodel.data.Dictionary;
import com.uncc.topicmodel.data.Document;

import java.io.BufferedWriter;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.Properties;

import org.apache.commons.math3.special.Gamma;
import org.apache.commons.math3.util.FastMath;

/**
 * @author Huayu Li
 */
public class AIR_VB extends Model implements
			Dictionary.DictionaryListener, EvaluationModel, Estimator {
	private static final boolean DEBUG = false;

	private static final String FILE_NAME_SUBFOLDER  = "AIR_VB";
	private static final String FILE_NAME_ALPHA_INIT = "alpha_init";

	private static final double EM_THRESHOLD     = 1.0e-4;
	private static final double E_THRESHOLD      = 1.0e-4;
	private static final double ALPHA_THRESHOLD  = 1.0e-4;
	private static final double GAMMA_THRESHOLD  = 1.0e-4;
	private static final double LAMBDA_THRESHOLD = 1.0e-4;
	private static final int  MAX_EM_ITER_NUM   = 300;

	private double[][] lambdaHat;
	private double[] lambdaHatSum;
	private double[] gammas;
	private double gammaSum;
	private double estimatedLambda;
	private double lambdaInit;

	private ArrayList<double[][]> phi    = null;
	private ArrayList<double[][]> varPi  = null;
	private double[][][] varMu           = null;
	private double[][] varTau            = null;
	private double[][] varEta            = null;
	private double[][][] wordProb        = null;
	private double[][] wordProbSum       = null;
	private double[] alpha               = null;
	private File outputPath              = null;
	private File groundtruthScoreFile    = null;
	private File groundtruthDisFile      = null;
	private EvaluationPool testEval      = null;

	public AIR_VB(ParamManager paramManager, File alphaInitFile)
					throws TopicModelException,
						AIRException, ToolException {
		initNewModel(paramManager, alphaInitFile);
	}

	@Override
	public void dictionaryChanged() {
		double[][][] wordProbNew = new double[SENTIMENT_NUM][TOPIC_NUM][dictionary.getSize()];
		for (int sentiment = 0; sentiment < SENTIMENT_NUM; sentiment ++) {
			for (int topic = 0; topic < TOPIC_NUM; topic ++) {
				for (int word = 0; word < wordProb[sentiment][topic].length; word ++) {
					wordProbNew[sentiment][topic][word] = wordProb[sentiment][topic][word];
				}
			}
		}
		wordProb = wordProbNew;
	}

	@Override
	public String getPrefix() {
		return "vb_%s";
	}

	@Override
	public int getTopicNum() {
		return TOPIC_NUM;
	}

	@Override
	public int getSentimentNum() {
		return SENTIMENT_NUM;
	}

	@Override
	public double[] getGammas() {
		return gammas;
	}

	@Override
	public double[] getLambdaHat(double rating, int topic) {
		return convertToLambdaHat(estimatedLambda, rating);
	}

	@Override
	public double[][][] getWordProb() {
		return wordProb;
	}

	@Override
	public double[] getAlphas() {
		return alpha;
	}

	@Override
	public Dictionary getDictionary() {
		return dictionary;
	}

	@Override
	public Dataset getDataset(File dataFile, Dictionary dictionary, File dicFile,
				boolean isRestore) throws TopicModelException {
		return new AIRDataset(dictionary, dataFile, dicFile, isRestore);
	}

	@Override
	public void estimate () throws ToolException{
		getParameters().list(System.out);

		double logLikelihood      = 0;
		double logLikelihoodOld   = Double.MAX_VALUE;
		double error              = Double.MAX_VALUE;
		int    iter               = 0;
		long startTime            = System.currentTimeMillis();
		BufferedWriter perpWriter = null;
		File perOutputFile        = new File(outputPath, format(
						AIRConstants.FILE_NAME_PERPLEXITY));
		double[] alphaOld         = null;
		double[] gammaOld         = null;
		double[][][] wordProbOld  = new double[SENTIMENT_NUM][TOPIC_NUM][dictionary.getSize()];
		double[][] wordProbSumOld = new double[SENTIMENT_NUM][TOPIC_NUM];

		try {
			perpWriter = Utils.createBufferedWriter(perOutputFile, true);
			Utils.writeAndPrint(perpWriter,
				String.format(FORMAT_MODEL_NAME,
				lambdaInit,
				gammas[0], gammas[1],
				TOPIC_NUM), true);
		} catch (IOException e) {
			e.printStackTrace();
		}
		do {
			alphaOld = Arrays.copyOf(alpha, alpha.length);
			gammaOld = Arrays.copyOf(gammas, gammas.length);
			for (int sentiment = 0; sentiment < SENTIMENT_NUM; sentiment ++) {
				wordProbOld[sentiment]    = Utils.copyOf(wordProb[sentiment]);
				wordProbSumOld[sentiment] = Arrays.copyOf(wordProbSum[sentiment],
								wordProbSum[sentiment].length);
			}

			try {
				logLikelihood = EStep();
						MStep();
			} catch (AIRException e) {
				logLikelihood = Double.NaN;
				e.printStackTrace();
			}
			if (Double.isNaN(logLikelihood) || Double.isInfinite(logLikelihood)) {
				Utils.err("logLikelihood = " + logLikelihood);
				alpha       = alphaOld;
				gammas      = gammaOld;
				wordProb    = wordProbOld;
				wordProbSum = wordProbSumOld;
				break;
			}
			if (logLikelihoodOld != Double.MAX_VALUE) {
				error = Math.abs((logLikelihood - logLikelihoodOld) / logLikelihoodOld);
			}
			if (iter % 1 == 0) {
				try {
					Utils.writeAndPrint(perpWriter, 
						"\t[EM] iter = " + iter +
						"; LogLikelihood = " + logLikelihood +
						"; Perplexity = " + calculatePerplexity(logLikelihood) +
						"; error = " + error +
						"; OldLogLikelihood = " + logLikelihoodOld +
						"; Duration = " + (System.currentTimeMillis() - startTime) / 1000.0,
						true);
					Utils.writeAndPrint(perpWriter, String.format(
							"\t\tGammas = %s, Alpha = %s",
							Utils.convertToString(gammas, ","),
							Utils.convertToString(alpha, ",")),
								true);
				} catch (IOException e) {
					e.printStackTrace();
				}
				startTime = System.currentTimeMillis();
			}
			logLikelihoodOld = logLikelihood;
		} while (error > EM_THRESHOLD && ++ iter < MAX_EM_ITER_NUM);

		if (error <= EM_THRESHOLD) Utils.println("Converged.");
		else Utils.println("Not Converged.");

		for (int sentiment = 0; sentiment < SENTIMENT_NUM; sentiment ++) {
			for (int topic = 0; topic < TOPIC_NUM; topic ++) {
				for (int word = 0; word < dictionary.getSize(); word ++) {
					wordProb[sentiment][topic][word] /= wordProbSum[sentiment][topic];
				}
			}
		}

		storeResult(outputPath);

		if (groundtruthScoreFile != null || groundtruthDisFile != null) {
			try {
				String seval = Measure.evaluate(
						getScoreFile(outputPath),
						groundtruthScoreFile,
						Model.sumWordProb(wordProb),
						groundtruthDisFile,
						AIRConstants.CONFIG_MANAGER.getRatingScaler(),
						true, AIRConstants.DEFAULT_MISS_ASPECT_RATING);
				Utils.write(perpWriter, seval, true);
			} catch (ToolException e) {
				e.printStackTrace();
			} catch (IOException e) {
				e.printStackTrace();
			}
		}

		estimateLambda();
		if (testEval != null) {
			testEval.evaluate();
			try {
				testEval.outputEvaluation(perpWriter, "[Test] ");
			} catch (IOException e) {
				e.printStackTrace();
			}
		}

		Utils.cleanup(perpWriter);
	}

	private double EStep() throws ToolException, AIRException {
		double logLikelihood = 0.0;
		for (int doc = 0; doc < data.getDocumentSize(); doc ++) {
			logLikelihood += EStep(data.getDocument(doc),
						phi.get(doc), varPi.get(doc),
						varEta[doc], varMu[doc],
						varTau[doc], lambdaHat[doc],
						lambdaHatSum[doc]);
		}
		//System.out.println("\tlogLikelihood = " + logLikelihood +"; Perplexity = " + calculatePerplexity(logLikelihood));
		return logLikelihood;
	}

	private double EStep (Document doc, double[][] phi, double[][] varPi,
				double[] varEta, double[][] varMu,
				double[] varTau, double[] lambdaHat,
				double lambdaHatSum) throws ToolException,
								AIRException {
		//long startTime       = System.currentTimeMillis();
		double logLikelihoodError = Double.MAX_VALUE;
		double logLikelihoodOld   = Double.MAX_VALUE;
		int iter                  = 0;
		double[] piInit           = {gammas[0] / gammaSum, lambdaHat[0] / lambdaHatSum};
		double[] piHatInit        = getPiHat(piInit);//{gammas[0] / gammaSum, lambdaHat[0] / lambdaHatSum};
		for (int topic = 0; topic < TOPIC_NUM; topic ++) {
			varEta[topic] = alpha[topic] + doc.wordNum * 1.0 / TOPIC_NUM;
			for (int sentiment = 0; sentiment < 2; sentiment ++) {
				varMu[topic][sentiment] = lambdaHat[sentiment] +
							doc.wordNum * piHatInit[sentiment + 1] / TOPIC_NUM;
			}
		}
		for (int sentiment = 0; sentiment < 2; sentiment ++) {
			varTau[sentiment] = gammas[sentiment] + doc.wordNum *
					( sentiment == 0 ? piHatInit[0] : (1 - piHatInit[0]) );
		}
		for (int position = 0; position < doc.getUniqueWordNum(); position ++) {
			for (int topic = 0; topic < TOPIC_NUM; topic ++) {
				phi[position][topic] = 0.0;//-1.0 * Math.log(TOPIC_NUM);
			}
			for (int sentiment = 0; sentiment < 2; sentiment ++) {
				varPi[position][sentiment] = piInit[sentiment];
			}
		}

		do {
			double[] phiSum            = new double[TOPIC_NUM];
			double varPi0Sum           = 0.0;
			double[][] varPi1Sum       = new double[TOPIC_NUM][2];
			double[][] digammaVarMuSum = new double[TOPIC_NUM][2];
			for (int topic = 0; topic < TOPIC_NUM; topic ++) {
				double varMuSenSum = varMu[topic][0] + varMu[topic][1];
				for (int sentiment = 0; sentiment < 2; sentiment ++) {
					digammaVarMuSum[topic][sentiment] =
							Gamma.digamma(varMu[topic][sentiment]) -
							Gamma.digamma(varMuSenSum);
				}
			}
			for (int position = 0; position < doc.getUniqueWordNum(); position ++) {
				int word              = doc.uniqueWords[position];
				int count             = doc.uniqueWordCounts[position];
				double[][] logEpsilon = new double[TOPIC_NUM][SENTIMENT_NUM];
				double[] piHat        = getPiHat(varPi[position]);
				double[] logPi1       = new double[2];
				double pi0            = 0.0;
				double sum            = 0.0;
				Utils.fills(phi[position], 0.0);
				for (int topic = 0; topic < TOPIC_NUM; topic ++) {
					for (int sentiment = 0; sentiment < SENTIMENT_NUM; sentiment ++) {
						logEpsilon[topic][sentiment] =
								FastMath.log(wordProb[sentiment][topic][word]) -
								FastMath.log(wordProbSum[sentiment][topic]);
						phi[position][topic]        +=
								piHat[sentiment] *
								logEpsilon[topic][sentiment];
					}
					phi[position][topic] += Gamma.digamma(varEta[topic]);
					for (int sentiment = 0; sentiment < 2; sentiment ++) {
						phi[position][topic] += piHat[sentiment + 1] *
									digammaVarMuSum[topic][sentiment];
					}
					
					if (topic == 0) {
						sum = phi[position][topic];
					} else {
						sum = Utils.logSum(sum, phi[position][topic]);
					}
				}
				for (int topic = 0; topic < TOPIC_NUM; topic ++) {
					phi[position][topic] = FastMath.exp(phi[position][topic] - sum);
					if (phi[position][topic] == 0.0) {
						phi[position][topic] = FastMath.exp(-100);
					}
					for (int sentiment = 0; sentiment < 2; sentiment ++) {
						logPi1[sentiment] += phi[position][topic] *
								( logEpsilon[topic][sentiment + 1] +
								  digammaVarMuSum[topic][sentiment] );
					}
					if (phi[position][topic] > 1) {
						System.err.println("phi >= 1 :::" + phi[position][topic] + "; sum=" + sum);
						throw new AIRException("Bad initial values.....");
					}
				}
				// update varPi[position][1]
				varPi[position][1] = FastMath.exp(logPi1[0] -
							Utils.logSum(logPi1[0], logPi1[1]));
				if (varPi[position][1] <= 0 || varPi[position][1] >= 1) {
					varPi[position][1] = getApproximation(varPi[position][1]);
					
				}

				// update varPi[position][1]
				double[] pi = {varPi[position][1], 1 - varPi[position][1]};
				for (int topic = 0; topic < TOPIC_NUM; topic ++) {
					pi0 += phi[position][topic] *
						( logEpsilon[topic][0] -
						  pi[0] * (logEpsilon[topic][1] + digammaVarMuSum[topic][0]) -
						  pi[1] * (logEpsilon[topic][2] + digammaVarMuSum[topic][1]) );
				}
				pi0 += Gamma.digamma(varTau[0]) - Gamma.digamma(varTau[1]) +
						pi[0] * FastMath.log(pi[0]) +
						pi[1] * FastMath.log(pi[1]);
				varPi[position][0] = 1.0 / (FastMath.exp(-1.0 * pi0) + 1.0);

				if (varPi[position][0] <= 0 || varPi[position][0] >= 1 ) {
					varPi[position][0] = getApproximation(varPi[position][0]);
				}

				varPi0Sum += varPi[position][0] * count;
				for (int topic = 0; topic < TOPIC_NUM; topic ++) {
					double value         = phi[position][topic] * count;
					phiSum[topic]       += value;
					varPi1Sum[topic][0] += (1 - varPi[position][0]) * varPi[position][1] * value;
					varPi1Sum[topic][1] += (1 - varPi[position][0]) * (1 - varPi[position][1]) * value;
				}
			} // end for position

			for (int topic = 0; topic < TOPIC_NUM; topic ++) {
				varEta[topic] = alpha[topic] + phiSum[topic];
				for (int sentiment = 0; sentiment < 2; sentiment ++) {
					varMu[topic][sentiment] = lambdaHat[sentiment] + varPi1Sum[topic][sentiment];
				}
			}
			varTau[0] = gammas[0] + varPi0Sum;
			varTau[1] = gammas[1] + doc.wordNum - varPi0Sum;

			double logLikelihood = calculateLogLikelihood(doc, phi,
						varPi, varEta, varTau, varMu,
							lambdaHat, lambdaHatSum);
			if (logLikelihoodOld != Double.MIN_VALUE) {
				logLikelihoodError = (logLikelihoodOld - logLikelihood) / logLikelihoodOld;
			}
			logLikelihoodOld = logLikelihood;
	
			if (DEBUG) {
				Utils.println("\t\t[E-step] doc = " + doc +
					"; iter = " + iter +
					" ; varEtaError = " + logLikelihoodError +
					" ; varEta : "+ Utils.convertToString(varEta, "\t"));
			}
			iter ++;
		} while (logLikelihoodError > E_THRESHOLD);

		if (DEBUG) {
			Utils.println("\t\t[E-step] Loglikelihood = " + logLikelihoodOld +
					", Perplexity = " + calculatePerplexity(logLikelihoodOld));
		}
		/*System.out.println("varIter=" + iter +
				",duration=" + (System.currentTimeMillis() - startTime));*/

		return logLikelihoodOld;
	}

	private void MStep() throws ToolException {
		updateWordProb();
		updateAlpha();

		//updateGamma();
		/*Utils.println("\tGammas = " + Utils.convertToString(gammas, ",") +
				"; gammaSum = " + gammaSum);*/


		//updateLambdaHat();
		/*for (int doc = 0; doc < data.getDocumentSize(); doc ++) {
			log.save(String.format("lambdaHat=%s; lambdaHatSum=%s",
					Utils.convertToString(lambdaHat[doc], ","),
					lambdaHatSum[doc]), false);
		}*/
	}

	private void updateWordProb() {
		for (int sentiment = 0; sentiment < SENTIMENT_NUM; sentiment ++) {
			Utils.fills(wordProb[sentiment], 0.0);
			Utils.fills(wordProbSum[sentiment], 0.0);
			for (int topic = 0; topic < TOPIC_NUM; topic ++) {
				for (int doc = 0; doc < data.getDocumentSize(); doc ++) {
					for (int position = 0; position < data.getDocument(doc).getUniqueWordNum(); position ++) {
						int word        = data.getDocument(doc).uniqueWords[position];
						int count       = data.getDocument(doc).uniqueWordCounts[position];
						double value    = phi.get(doc)[position][topic] * count;
						double[] phiHat = getPiHat(varPi.get(doc)[position]);
						wordProb[sentiment][topic][word] += value * phiHat[sentiment];
						wordProbSum[sentiment][topic]    += value * phiHat[sentiment];
					}
				}
			}
		}
	}

	private void updateAlpha() throws ToolException {
		/*LDAGradientAscent.estimate(alpha, digammaVarEtaDiff(),
					data.getDocumentSize(), ALPHA_THRESHOLD);*/
		new LDANewtonRaphson().estimate(alpha, digammaVarEtaDiff(),
				data.getDocumentSize(), ALPHA_THRESHOLD, 2);
	}

	private void updateGamma() throws ToolException {
		/*new LDANewtonRaphson().estimate(gammas, digammaVarTauDiff(),
				data.getDocumentSize(), GAMMA_THRESHOLD, 2);*/

		/*try {
			BetaOptimizer gammaOptimizer = new BetaOptimizer(
						gammas, digammaVarTauDiff(),
						varTau, GAMMA_THRESHOLD);
			gammaOptimizer.optimize();
		} catch (RTMException e) {
			e.printStackTrace();
		}
		gammaSum = gammas[0] + gammas[1];*/
 
		double[] digammaVarTauDiff = digammaVarTauDiff();
		double[] constants         = new double[digammaVarTauDiff.length];
		for (int topic = 0; topic < constants.length; topic ++) {
			constants[topic] = digammaVarTauDiff[topic] / data.getDocumentSize();
		}
		try {
			DirichletParamsEstimator.optimize(gammas, constants, GAMMA_THRESHOLD, null, 0);
		} catch (AIRException e) {
			e.printStackTrace();
		}

		gammaSum = gammas[0] + gammas[1];
	}

	private void updateLambdaHat() throws ToolException {
		for (int doc = 0; doc < data.getDocumentSize(); doc ++) {
			/*new LDANewtonRaphson().estimate(lambdaHat[doc],
					digammaVarMuDiff(doc),
					TOPIC_NUM, LAMBDA_THRESHOLD, 0);*/
			/*try {
				BetaOptimizer lambdaOptimizer = new BetaOptimizer(
						lambdaHat[doc], digammaVarMuDiff(doc),
						varMu[doc], LAMBDA_THRESHOLD);
				lambdaOptimizer.optimize();
			} catch (RTMException e) {
				e.printStackTrace();
			}*/
			double[] digammaVarMuDiff = digammaVarMuDiff(doc);
			double[] constants        = new double[digammaVarMuDiff.length];
			for (int i = 0; i < digammaVarMuDiff.length; i ++) {
				constants[i] = digammaVarMuDiff[i] / TOPIC_NUM;
			}
			try {
				DirichletParamsEstimator.optimize(lambdaHat[doc],
						constants, LAMBDA_THRESHOLD, null, 0);
			} catch (AIRException e) {
				e.printStackTrace();
			}
			lambdaHatSum[doc] = lambdaHat[doc][0] + lambdaHat[doc][1];
		}
	}

	private double[] digammaVarEtaDiff() {
		double sum[] = new double[TOPIC_NUM];
		for (int doc = 0; doc < data.getDocumentSize(); doc ++) {
			double varEtaSum = 0.0;
			for (int topic = 0; topic < TOPIC_NUM; topic ++) {
				varEtaSum += varEta[doc][topic];
			}
			for (int topic = 0; topic < TOPIC_NUM; topic ++) {
				sum[topic] += Gamma.digamma(varEta[doc][topic]) -
						Gamma.digamma(varEtaSum);
			}
		}
		return sum;
	}

	private double[] digammaVarTauDiff() {
		double sum[] = new double[2];
		for (int doc = 0; doc < data.getDocumentSize(); doc ++) {
			double varTauSum = 0.0;
			for (int sentiment = 0; sentiment < 2; sentiment ++) {
				varTauSum += varTau[doc][sentiment];
			}
			for (int sentiment = 0; sentiment < 2; sentiment ++) {
				sum[sentiment] += Gamma.digamma(varTau[doc][sentiment]) -
							Gamma.digamma(varTauSum);
			}
		}
		return sum;
	}

	private double[] digammaVarMuDiff(int doc) {
		double sum[] = new double[2];
		for (int topic = 0; topic < TOPIC_NUM; topic ++) {
			double varMuSum = varMu[doc][topic][0] + varMu[doc][topic][1];
			for (int sentiment = 0; sentiment < 2; sentiment ++) {
				sum[sentiment] += Gamma.digamma(varMu[doc][topic][sentiment]) -
							Gamma.digamma(varMuSum);
			}
		}
		return sum;
	}

	private double calculatePerplexity(double logLikelihood) {
		return FastMath.exp(-1.0 * logLikelihood / data.getTermNum());
	}

	// bound Loglikelihood
	private double calculateLogLikelihood(Document doc, double[][] phi,
			double[][] varPi, double[] varEta, double[] varTau,
				double[][] varMu, double[] lambdaHat,
					double lambdaHatSum) {
		double logLikelihood        = 0.0;
		double varEtaSum            = 0.0;
		double alphaSum             = 0.0;
		double varTauSum            = 0.0;
		double[] varMuSum           = new double[TOPIC_NUM];
		double[] digammaVarEtaDiff  = new double[TOPIC_NUM];
		double[] digammaVarTauDiff  = new double[2];
		double[][] digammaVarMuDiff = new double[TOPIC_NUM][2];

		for (int sentiment = 0; sentiment < 2; sentiment ++) {
			varTauSum += varTau[sentiment];
		}
		for (int topic = 0; topic < TOPIC_NUM; topic ++) {
			alphaSum  += alpha[topic];
			varEtaSum += varEta[topic];
			for (int sentiment = 0; sentiment < 2; sentiment ++) {
				varMuSum[topic] += varMu[topic][sentiment];
			}
		}
		for (int sentiment = 0; sentiment < 2; sentiment ++) {
			digammaVarTauDiff[sentiment] = Gamma.digamma(varTau[sentiment]) -
							Gamma.digamma(varTauSum);
			logLikelihood               += Gamma.logGamma(varTau[sentiment]) -
							Gamma.logGamma(gammas[sentiment]) +
							(gammas[sentiment] - varTau[sentiment]) *
								digammaVarTauDiff[sentiment];
		}
		if (Double.isNaN(logLikelihood) || Double.isInfinite(logLikelihood)) {
			System.err.println("LogLikelihood = " + logLikelihood +
					"; varTau = " + Utils.convertToString(varTau, ",") +
					"; gammas = " + Utils.convertToString(gammas, ","));
			return logLikelihood;
		}
		logLikelihood += Gamma.logGamma(alphaSum) - Gamma.logGamma(varEtaSum) +
				Gamma.logGamma(gammaSum) - Gamma.logGamma(varTauSum);
		if (Double.isNaN(logLikelihood) || Double.isInfinite(logLikelihood)) {
			System.err.println("LogLikelihood = " + logLikelihood +
					"; alphaSum = " + alphaSum +
					"; varEtaSum = " + varEtaSum +
					"; gammaSum = " + gammaSum +
					"; varTauSum = " + varTauSum);
			return logLikelihood;
		}
		for (int topic = 0; topic < TOPIC_NUM; topic ++) {
			digammaVarEtaDiff[topic] = Gamma.digamma(varEta[topic]) -
							Gamma.digamma(varEtaSum);
			logLikelihood           += Gamma.logGamma(varEta[topic]) -
							Gamma.logGamma(alpha[topic]) +
							(alpha[topic] - varEta[topic]) *
								digammaVarEtaDiff[topic];
			if (Double.isNaN(logLikelihood) || Double.isInfinite(logLikelihood)) {
				System.err.println("LogLikelihood = " + logLikelihood +
						"; varEta[topic] = " + varEta[topic] +
						"; alpha[topic] = " + alpha[topic]);
				return logLikelihood;
			}
			logLikelihood           += Gamma.logGamma(lambdaHatSum) -
							Gamma.logGamma(varMuSum[topic]);
			if (Double.isNaN(logLikelihood) || Double.isInfinite(logLikelihood)) {
				System.err.println("LogLikelihood = " + logLikelihood +
						"; lambdaHatSum = " + lambdaHatSum +
						"; varMuSum[topic] = " + varMuSum[topic]);
				return logLikelihood;
			}
			for (int sentiment = 0; sentiment < 2; sentiment ++) {
				digammaVarMuDiff[topic][sentiment] = Gamma.digamma(varMu[topic][sentiment]) -
									Gamma.digamma(varMuSum[topic]);
				logLikelihood                     += Gamma.logGamma(varMu[topic][sentiment]) -
									Gamma.logGamma(lambdaHat[sentiment]) +
									(lambdaHat[sentiment] - varMu[topic][sentiment]) *
										digammaVarMuDiff[topic][sentiment];
				if (Double.isNaN(logLikelihood) || Double.isInfinite(logLikelihood)) {
					System.err.println("LogLikelihood = " + logLikelihood +
							"; varMu[topic][sentiment] = " + varMu[topic][sentiment] +
							"; lambdaHat[sentiment] = " + lambdaHat[sentiment]);
					return logLikelihood;
				}
			}
			
		}
		for (int position = 0; position < doc.getUniqueWordNum(); position ++) {
			int word       = doc.uniqueWords[position];
			int count      = doc.uniqueWordCounts[position];
			double[] t     = {varPi[position][0], 1 - varPi[position][1]};
			double[] piHat = getPiHat(varPi[position]);
			for (int topic = 0; topic < TOPIC_NUM; topic ++) {
				for (int sentiment = 0; sentiment < SENTIMENT_NUM; sentiment ++) {
					logLikelihood += count * phi[position][topic] * piHat[sentiment] *
							( FastMath.log(wordProb[sentiment][topic][word]) -
									FastMath.log(wordProbSum[sentiment][topic]) );
				}
				logLikelihood += count * phi[position][topic] *(
								digammaVarEtaDiff[topic] -
								FastMath.log(phi[position][topic]) );
				for (int sentiment = 0; sentiment < 2; sentiment ++) {
					logLikelihood += count * phi[position][topic] *
							piHat[sentiment + 1] *
							digammaVarMuDiff[topic][sentiment];
				}
				if (Double.isNaN(logLikelihood) || Double.isInfinite(logLikelihood)) {
					System.err.println("LogLikelihood = " + logLikelihood +
							"; phi.get(doc)[position][topic]=" + phi[position][topic]);
					return logLikelihood;
				}
			}
			
			for (int sentiment = 0; sentiment < 2; sentiment ++) {
				logLikelihood += count * t[sentiment] * digammaVarTauDiff[sentiment];
			}
			for (int sentiment = 0; sentiment < SENTIMENT_NUM; sentiment ++) {
				logLikelihood -= count * piHat[sentiment] * FastMath.log(piHat[sentiment]);
			}
		}
		//if (Double.isNaN(logLikelihood)) Utils.err(Utils.convertToString(alpha, ","));
		return logLikelihood;
	}

	private void estimateLambda() {
		estimatedLambda = 0.0;
		for (int doc = 0; doc < data.getDocumentSize(); doc ++) {
			estimatedLambda +=  lambdaHatSum[doc];
		}
		estimatedLambda /= data.getDocumentSize();

		Utils.println("estimatedLambda = " + estimatedLambda);
	}

	private double[] convertToLambdaHat(double lambda, double rating) {
		double[] result = new double[2];
		result[0] = lambda * rating;
		result[1] = lambda * (1 - rating);
		return result;
	}

	private double[] getPiHat(double p0, double p1) {
		return getPiHat(new double[] {p0, p1});
	}
	private double[] getPiHat(double[] pi) {
		return new double[] {pi[0], (1 - pi[0]) * pi[1], (1 - pi[0]) * (1 - pi[1])};
	}

	private double getApproximation(double value) {
		if (value == 1.0) return FastMath.exp(-50) * (FastMath.exp(50) - 1);
		else if (value == 0.0) return FastMath.exp(-100);

		return value;
	}

	private void initNewModel(ParamManager paramManager,
				File alphaInitFile)
				throws AIRException, TopicModelException,
							ToolException {
		parseParamArguments(paramManager);

		phi   = new ArrayList<double[][]>();
		varPi = new ArrayList<double[][]>();
		for (int doc = 0; doc < data.getDocumentSize(); doc ++) {
			phi.add(new double[data.getDocument(doc).getUniqueWordNum()][TOPIC_NUM]);
			varPi.add(new double[data.getDocument(doc).getUniqueWordNum()][2]);
		}
		varEta = new double[data.getDocumentSize()][TOPIC_NUM];
		varTau = new double[data.getDocumentSize()][2];
		varMu  = new double[data.getDocumentSize()][TOPIC_NUM][2];

		initWordProb();
		initAlpha(alphaInitFile);

		if (! store(alpha, new File(outputPath, format(
				FILE_NAME_ALPHA_INIT)), false)) {
			Utils.err(ErrorMessage.ERROR_STORE_ALPHA_INIT);
		}

		dictionary.addDictionaryListener(this);

		if (paramManager.getDataTestFile() != null) {
			testEval = new EvaluationPool("Test",
					paramManager.getDataTestFile(), outputPath, this,
					paramManager.isRestore());
		}
	}

	private void parseParamArguments(ParamManager paramManager)
					throws AIRException, TopicModelException {
		if (paramManager == null) {
			throw new AIRException(ErrorMessage.ERROR_NO_ARGS);
		}

		if (paramManager.getModelOutputPath() != null) {
			outputPath = new File(paramManager.getModelOutputPath(),
					FILE_NAME_SUBFOLDER);
		} else {
			throw new AIRException(ErrorMessage.ERROR_NO_OUTPUT_PATH);
		}

		if (paramManager.getDataTrainFile() == null) {
			throw new AIRException(ErrorMessage.ERROR_NO_TRAIN_FILE);
		}
		this.dictionary = new Dictionary();
		this.data       = getDataset(paramManager.getDataTrainFile(), dictionary,
					new File(outputPath, format(AIRConstants.FILE_NAME_DIC_MAP)),
						false);

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

		if (paramManager.getGammas() != null) {
			gammas   = paramManager.getGammas();
			gammaSum = gammas[0] + gammas[1];
		} else {
			throw new AIRException(ErrorMessage.ERROR_NO_GAMMA);
		}

		if (paramManager.getLambda() != null) {
			lambdaInit   = paramManager.getLambda();
			lambdaHat    = new double[data.getDocumentSize()][2];
			lambdaHatSum = new double[data.getDocumentSize()];
			for (int doc = 0; doc < data.getDocumentSize(); doc ++) {
				lambdaHat[doc]    = convertToLambdaHat(
							lambdaInit,
							data.getDocument(doc).Ri);
				lambdaHatSum[doc] = lambdaHat[doc][0] + lambdaHat[doc][1];
			}
		} else {
			throw new AIRException(ErrorMessage.ERROR_NO_LAMBDA);
		}

		if (paramManager.getGroundtruthScoreFile() != null) {
			groundtruthScoreFile = paramManager.getGroundtruthScoreFile();
		}

		if (paramManager.getGroundtruthTopicWordDistributionFile() != null) {
			groundtruthDisFile = paramManager.getGroundtruthTopicWordDistributionFile();
		}

		if (paramManager.getKeywordFile() != null) {
			keywords = loadKeywords(paramManager.getKeywordFile(),
								dictionary);
			if (keywords.length != TOPIC_NUM) {
				throw new AIRException(String.format(
						ErrorMessage.ERROR_NOT_MATCH_KEYWORD,
						keywords.length, TOPIC_NUM));
			}
		}
	}

	private void initWordProb() throws ToolException {
		wordProb    = new double[SENTIMENT_NUM][TOPIC_NUM][dictionary.getSize()];
		wordProbSum = new double[SENTIMENT_NUM][TOPIC_NUM];
		if (keywords == null) {
		Utils.println("Randomly initiating wordProb value..");
		for (int sentiment = 0; sentiment < SENTIMENT_NUM; sentiment ++) {
			for (int topic = 0; topic < TOPIC_NUM; topic ++) {
				//wordProb[topic] = randomNormalizedArray(dictionary.getSize());
				for (int word = 0; word < wordProb[sentiment][topic].length; word ++) {
					wordProb[sentiment][topic][word] = Math.random() + 1.0 / dictionary.getSize();
					wordProbSum[sentiment][topic]   += wordProb[sentiment][topic][word];
				}
			}
		}
		} else {
			for (int sentiment = 0; sentiment < SENTIMENT_NUM; sentiment ++) {
				double[][] initValue = new double[TOPIC_NUM][dictionary.getSize()];
				for (int topic = 0; topic < TOPIC_NUM; topic ++) {
					for (int word = 0; word < dictionary.getSize(); word ++) {
						initValue[topic][word] = Math.random() + 1.0 / dictionary.getSize();
					}
				}
				wordProb[sentiment] = initArrayWithKeyword(0.0,
							initValue, keywords);
				for (int topic = 0; topic < TOPIC_NUM; topic ++) {
					for (int word = 0; word < dictionary.getSize(); word ++) {
						wordProbSum[sentiment][topic] += wordProb[sentiment][topic][word];
					}
				}
			}
		}

		/*double[][] a = new double[][]{{0.0122,0.0218,0.4163,0.2165,0.1856,0.1587,0.3355,0.0557,0.1346,0.2099},{0.3917,0.3485,0.2273,0.0111,0.0871,0.3350,0.1199,0.1909,0.2958,0.2002}, {0.2727,0.1790,0.0479,0.2798,0.0940,0.0490,0.0435,0.2749,0.2344,0.0044}, {0.0910,0.2215, 0.2793,0.2177,0.2743,0.2657,0.2071,0.3486,0.0809, 0.2703},{0.2324,0.2291,0.0293,0.2749,0.3591,0.1916,0.2939,0.1299,0.2542,0.3152}};
		for (int topic = 0; topic < TOPIC_NUM; topic ++) {
			for (int w = 0; w < dictionary.getSize(); w ++) {
				wordProb[topic][w] = a[w][topic];  
			}
		}*/
	}

	private void initAlpha(File inputFile) throws ToolException {
		alpha = new double[TOPIC_NUM];
		if (inputFile == null) {
			Utils.println("Initial Alpha is 2 / K");
			Utils.fills(alpha, 2.0 / TOPIC_NUM);
		} else {
			Utils.println("Loading Alpha...");
			alpha = Utils.load1Array(inputFile, AIRConstants.SPACER_TAB);
			if (alpha == null || alpha.length != TOPIC_NUM) {
				throw new ToolException("Failed to load alpha.");
			}
		}

		// randomly initiate
		//alpha = randomNormalizedArray(TOPIC_NUM);
	}

	private ArrayList<Object[]> getTopWordsWithProb(int topic) {
		if (topic < 0 || topic >= TOPIC_NUM) return null;

		ArrayList<Object[]> list = new ArrayList<Object[]>();
		for (int word = 0; word < wordProb[topic].length; word ++) {
			Object[] object = {dictionary.getWord(word), wordProb[topic][word]};
			list.add(object);
		}
		Collections.sort(list, new Comparator<Object[]>() {
			@Override
			public int compare(Object[] obj1, Object[] obj2) {
				return ((Double)obj2[1]).compareTo(
						((Double)obj1[1]));
			}
		});

		ArrayList<Object[]> result = new ArrayList<Object[]>();
		int maxWordNum             = TOP_WORD_NUM > dictionary.getSize() ?
						dictionary.getSize() : TOP_WORD_NUM;
		for (int i = 0; i < maxWordNum; i ++) {
			result.add(list.get(i));
		}
		return result;
	}

	@Override
	public Properties getParameters() {
		Properties props = new Properties();

		props.setProperty(AIRConstants.STRING_DOCUMENT_NUM,		 String.valueOf(data.getDocumentSize()));
		props.setProperty(AIRConstants.STRING_TOPIC_NUM,		 String.valueOf(TOPIC_NUM));
		props.setProperty(AIRConstants.STRING_DIC_NUM,			 String.valueOf(dictionary.getSize()));
		props.setProperty(AIRConstants.STRING_DIC_NUM,			 String.valueOf(dictionary.getSize()));
		props.setProperty(AIRConstants.STRING_MAX_EM_ITER_NUM,	 String.valueOf(MAX_EM_ITER_NUM));
		props.setProperty(AIRConstants.STRING_EM_THRESHOLD,		 String.valueOf(EM_THRESHOLD));
		props.setProperty(AIRConstants.STRING_E_THRESHOLD,		 String.valueOf(E_THRESHOLD));
		props.setProperty(AIRConstants.STRING_ALPHA_THRESHOLD,	 String.valueOf(ALPHA_THRESHOLD));
		props.setProperty(AIRConstants.STRING_TOKEN_NUM,		 String.valueOf(data.getTermNum()));
		props.setProperty(AIRConstants.STRING_ALPHAS,			 String.valueOf(Utils.convertToString(alpha, AIRConstants.SPACER_COMMA)));
		props.setProperty(AIRConstants.STRING_GAMMAS,			 String.valueOf(Utils.convertToString(gammas, AIRConstants.SPACER_COMMA)));
		props.setProperty(AIRConstants.STRING_LAMBDA,			 String.valueOf(lambdaInit));
		props.setProperty(AIRConstants.STRING_MODEL_OUTPUT_PATH, outputPath.getAbsolutePath());
		props.setProperty(AIRConstants.STRING_KEYWORD_FILE,		 keywords == null ? "" : "YES");

		props.putAll(data.getParams());
		
		return props;
	}

	private boolean storeResult(File outputPath) {
		boolean resCode         = true;

		File alphaOutputFile    = new File(outputPath, 
					format(AIRConstants.FILE_NAME_ALPHA_EST));
		File paramOutputFile    = new File(outputPath,
					format(AIRConstants.FILE_NAME_PARAM));
		File topwordOutputFile  = new File(outputPath,
					format(AIRConstants.FILE_NAME_TOP_WORD));
		File piOutputFile       = new File(outputPath, format("pi"));
		File muOutputFile       = new File(outputPath, format("mu"));
		File tauOutputFile      = new File(outputPath, format("tau"));
		File thetaOutputFile    = new File(outputPath, format("eta"));

		double[][][] estOmega = new double[data.getDocumentSize()][TOPIC_NUM][1];
		for (int doc = 0; doc < data.getDocumentSize(); doc ++) {
			for (int topic = 0; topic < TOPIC_NUM; topic ++) {
				estOmega[doc][topic][0] = varMu[doc][topic][0] /
						(varMu[doc][topic][0] + varMu[doc][topic][1]);
			}
		}

		if (! storeTopWords(topwordOutputFile, false, wordProb, null)) {
			Utils.err(ErrorMessage.ERROR_STORE_TOP_WORD);
			if (! resCode) resCode = false;
		}

		if (! Utils.save(getParameters(), paramOutputFile, true)) {
			Utils.err(ErrorMessage.ERROR_STORE_PARAMS);
			if (! resCode) resCode = false;
		}

		if (! storeScore(estOmega, getScoreFile(outputPath),
				AIRConstants.CONFIG_MANAGER.getRatingScaler(), true)) {
			Utils.err(ErrorMessage.ERROR_STORE_SCORE);
			if (! resCode) resCode = false;
		}

		if (! store(alpha, alphaOutputFile, false)) {
			Utils.err(ErrorMessage.ERROR_STORE_ALPHA);
			if (! resCode) resCode = false;
		}

		for (int sentiment = 0; sentiment < SENTIMENT_NUM; sentiment ++) {
			File wordProbOutputFile = new File(outputPath, format(
					AIRConstants.FILE_NAME_WORD_PROB + "_" + sentiment));
			if (! store(wordProb[sentiment], wordProbOutputFile, false)) {
				Utils.err(ErrorMessage.ERROR_STORE_WORD_PROB);
				if (! resCode) resCode = false;
			}
		}

		if (Utils.save(varEta, AIRConstants.SPACER_TAB,
						thetaOutputFile, true)) {
			if (resCode) resCode = false;
		}
		if (! Utils.save(varTau, AIRConstants.SPACER_TAB, tauOutputFile, true)) {
			Utils.err(ErrorMessage.ERROR_STORE_PARAM_T);
			if (resCode) resCode = false;
		}
		if (! storeVarPi(piOutputFile)) {
			Utils.err("Failed to store pi.");
			if (resCode) resCode = false;
		}
		if (! storeVarMu(muOutputFile)) {
			Utils.err("Failed to store mu.");
			if (resCode) resCode = false;
		}

		return resCode;
	}

	private boolean store(double[] array, File outputFile,
							boolean append) {
		return Utils.save(array, AIRConstants.SPACER_TAB, outputFile, append, false);
	}

	private boolean store(double[][] array, File outputFile,
							boolean append) {
		return Utils.save(array, AIRConstants.SPACER_TAB, outputFile, false);
	}

	private boolean storeVarPi(File outputFile) {
		BufferedWriter writer = null;
		try {
			writer = Utils.createBufferedWriter(outputFile);
			for (double[][] pi : varPi) {
				for (int pos = 0; pos < pi.length; pos ++) {
					writer.write(String.valueOf(pi[pos][0]) +
							", " + String.valueOf(pi[pos][1]));
					writer.write(AIRConstants.SPACER_TAB);
				}
				writer.newLine();
			}
			
			return true;
		} catch (IOException e) {
			e.printStackTrace();
		} finally {
			Utils.cleanup(writer);
		}
		return false;
	} 

	private boolean storeVarMu(File outputFile) {
		BufferedWriter writer = null;
		try {
			writer = Utils.createBufferedWriter(outputFile);
			for (double[][] pi : varMu) {
				for (int pos = 0; pos < pi.length; pos ++) {
					writer.write(String.valueOf(pi[pos][0]) +
							", " + String.valueOf(pi[pos][1]));
					writer.write(AIRConstants.SPACER_TAB);
				}
				writer.newLine();
			}
			return true;
		} catch (IOException e) {
			e.printStackTrace();
		} finally {
			Utils.cleanup(writer);
		}
		return false;
	}

	private double[] randomNormalizedArray(int size) {
		double[] array = new double[size];
		double   sum   = 0.0;
		for (int i = 0; i < size; i ++) {
			array[i] = Math.random() + 0.1;
			sum     += array[i];
		}
		for (int i = 0; i < size; i ++) {
			array[i] /= sum;
		}
		return array;
	}

	class BetaOptimizer extends ExpConstraintGradientOptimizer { //GradientOptimizer {
		private double[] digammaVarDiff = null;
		private double[][] variables    = null;

		public BetaOptimizer(double[] args, double[] digammaVarDiff,
				double[][] variables, double errorThreshold)
							throws AIRException {
			super(//GradientOptimizer.MODE_GRADIENT_FUNCTION_INTEGRATE,
					args, errorThreshold, 1.0e-4);

			this.digammaVarDiff = digammaVarDiff;
			this.variables       = variables;
		}
		
		public void calculateObjectGradients(double[] objectGradients,
				double[] objectFunction) {
			//Utils.println("args:" + Utils.convertToString(args, ","));
			double argSum = 0.0;
			for (int i = 0; i < args.length; i ++) {
				argSum += args[i];
			}
			for (int i = 0; i < args.length; i ++) {
				objectGradients[i] = -1.0 * (
						variables.length *
						(Gamma.digamma(argSum) - Gamma.digamma(args[i])) +
							digammaVarDiff[i]
						);
			}
			objectFunction[0] = variables.length * Gamma.logGamma(argSum);
			for (int i = 0; i < args.length; i ++) {
				objectFunction[0] -= variables.length * Gamma.logGamma(args[i]);
			}
			for (int i = 0; i < variables.length; i ++) {
				double varSum = 0.0;
				for (int j = 0; j < args.length; j ++) {
					varSum += variables[i][j];
				}
				for (int j = 0; j < args.length; j ++) {
					objectFunction[0] += (args[j] - 1) *
							(Gamma.digamma(variables[i][j]) -
								Gamma.digamma(varSum));
				}
			}
			objectFunction[0] *= -1.0;
			
		}
		public double getObjectFunction() throws AIRException{
			throw new AIRException("Never calls.");
		}
		public double[] getObjectGradients() throws AIRException{
			throw new AIRException("Never calls.");
		}
	}
}

