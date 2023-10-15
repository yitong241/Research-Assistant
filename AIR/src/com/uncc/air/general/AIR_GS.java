package com.uncc.air.general;

import java.io.BufferedWriter;
import java.io.File;
import java.io.IOException;
import java.util.Arrays;

import com.lhy.tool.util.Utils;
import com.lhy.tool.ToolException;
import com.uncc.topicmodel.TopicModelException;
import com.uncc.air.AIRConstants;
import com.uncc.air.AIRException;
import com.uncc.air.Estimator;
import com.uncc.air.ParamManager;
import com.uncc.air.general.eval.EvaluationPool;
import com.uncc.air.util.Measure;

import org.apache.commons.math3.special.Gamma;

/**
 * @author Huayu Li
 */
public class AIR_GS implements Estimator {
	private static final String FORMAT_ITER_BUILD              =
			"%sBuilding: iter = %s, duration = %s";
	private static final String FORMAT_ESTIMATE_ALPHA          =
			"\t%sEstimating  Alpha: alphaIter = %s, alphaError = %s, alphas = %s";
	private static final String FORMAT_ESTIMATE_LAMBDA         =
			"\t%sEstimating Lambda: lambdaEstIter = %s, lambdaError = %s, lambdas = %s";
	private static final String FORMAT_VALID_LIKELIHOOD_BURNIN =
			"\t%s[Validation] burninIter = %s ";
	private static final String FORMAT_VALID_LIKELIHOOD_ALPHA  =
			"\t%s[Validation] alphaEstIter = %s; logLikelihoodError = %s";
	private static final String FORMAT_VALID_LIKELIHOOD_LAMBDA =
			"\t%s[Validation] lambdaEstIter = %s; logLikelihoodError = %s";
	private static final String FORMAT_TEST                    =
			"\t%s[Test] ";
	private static final String FORMAT_UPDATE_ALPHA            =
			"\t\t%sUpdating Alpha: iter = %s, error = %s, alphas = %s";
	private static final String FORMAT_UPDATE_LAMBDA           =
			"\t\t%sUpdating Lambda: iter = %s, error = %s, lambdas = %s";
	private static final String FORMAT_SECTION_TITLE           =
			"%s[Topic = %s, betaInit=%s, lambda = %s, gamma = %s]";
	private static final String FORMAT_MODEL_NAME              =
			"Lambda=%.2f,Gamma=%.2f,%.2f,Topic=%s";

	private GeneralModel model                = null;
	private File modelOutputPath              = null;
	private EvaluationPool testEvalPool       = null;
	private EvaluationPool validationEvalPool = null;
	private BufferedWriter greedWriter        = null;
	private File groundtruthScoreFile         = null;
	private File groundtruthDisFile           = null;
	private String printPrefix                = null;
	private boolean debugConvergence;
	private boolean isOptimizeLambda;

	public AIR_GS(ParamManager paramManager, String printPrefix)
				throws AIRException, TopicModelException, ToolException {
		this(paramManager, null, printPrefix);
	}

	public AIR_GS(ParamManager paramManager, BufferedWriter greedWriter,
			String printPrefix) throws AIRException, TopicModelException, ToolException {
		if (paramManager == null) {
			throw new AIRException("No parameters specified..");
		}

		// reset model output path
		paramManager.setModelOutputPath(new File(
				new File(paramManager.getModelOutputPath(), "AIR_GS"), 
				String.format(FORMAT_MODEL_NAME,
						paramManager.getLambda(),
						paramManager.getGammas()[0],
						paramManager.getGammas()[1],
						paramManager.getTopicNum())));

		this.modelOutputPath      = paramManager.getModelOutputPath();
		this.debugConvergence     = paramManager.isDebugConvergence();
		this.isOptimizeLambda     = paramManager.isOptimizeLambda();
		this.groundtruthScoreFile = paramManager.getGroundtruthScoreFile();
		this.groundtruthDisFile   = paramManager.getGroundtruthTopicWordDistributionFile();
		this.greedWriter          = greedWriter;
		this.printPrefix          = printPrefix == null ? "" : printPrefix;
		this.model                = new GeneralModel(paramManager.getDataTrainFile(),
						paramManager);

		if (paramManager.getDataValidationFile() != null) {
			validationEvalPool = new EvaluationPool("validation",
					paramManager.getDataValidationFile(),
					modelOutputPath, model, paramManager.isRestore());
		}

		if (paramManager.getDataTestFile() != null) {
			testEvalPool = new EvaluationPool("test",
					paramManager.getDataTestFile(), modelOutputPath, model,
						paramManager.isRestore());
		}
	}

	@Override
	public void estimate() throws ToolException, AIRException {
		model.getParameters().list(System.out);

		File alphaOutputFile        = new File(modelOutputPath,
						AIRConstants.FILE_NAME_ALPHA_EST);
		File lambdaOutputFile       = new File(modelOutputPath,
						AIRConstants.FILE_NAME_LAMBDA_EST);
		File perplexityFile         = new File(modelOutputPath,
						AIRConstants.FILE_NAME_PERPLEXITY);
		BufferedWriter alphaWriter  = null;
		BufferedWriter lambdaWriter = null;
		BufferedWriter perpWriter   = null;
		String printContent         = null;
		try {
			perpWriter   = Utils.createBufferedWriter(perplexityFile);
			alphaWriter  = Utils.createBufferedWriter(alphaOutputFile);
			lambdaWriter = Utils.createBufferedWriter(lambdaOutputFile);
		
			printContent = String.format(FORMAT_SECTION_TITLE,
							printPrefix, model.TOPIC_NUM,
								model.betaInit, model.lambdaInit,
									Utils.convertToString(model.gammas, ","));
			if (greedWriter != null) {
				Utils.write(greedWriter, printContent, true);
			}
			Utils.writeAndPrint(perpWriter, printContent, true);

			if (isOptimizeLambda) {
				estimateWithOptimizingLambda(perpWriter, alphaWriter, lambdaWriter);
			} else {
				estimateWithOptimizingAlpha(perpWriter, alphaWriter);
			}

			if (testEvalPool != null) {
				testEvalPool.evaluate();
				testEvalPool.outputEvaluation(perpWriter,
						String.format(FORMAT_TEST, printPrefix));
				if (greedWriter != null) {
					testEvalPool.outputEvaluation(greedWriter,
						String.format(FORMAT_TEST, printPrefix),
						false);
				}
			}
		} catch (IOException e) {
			e.printStackTrace();
		} finally {
			Utils.cleanup(alphaWriter);
			Utils.cleanup(perpWriter);
			Utils.cleanup(lambdaWriter);
		}
	}

	private void estimateWithOptimizingLambda(BufferedWriter perpWriter,
				BufferedWriter alphaWriter,
					BufferedWriter lambdaWriter)
						throws AIRException, ToolException {
		String printContent     = null;
		double logLikelihoodOld = Double.MAX_VALUE;
		for (;model.lambdaEstIter < model.maxLambdaEstIterNum; model.lambdaEstIter ++) {
			double logLikelihood      = estimateWithOptimizingAlpha(perpWriter, alphaWriter);
			double logLikelihoodError = Double.MAX_VALUE;
			model.alphaEstIter        = 0;
			if (logLikelihoodOld != Double.MAX_VALUE) {
				logLikelihoodError = Math.abs((logLikelihood - logLikelihoodOld)) / Math.abs(logLikelihoodOld);

				try {
					printContent = String.format(FORMAT_VALID_LIKELIHOOD_LAMBDA,
								printPrefix,
								model.lambdaEstIter,
								logLikelihoodError);
					Utils.writeAndPrint(perpWriter, printContent, true);
					if (greedWriter != null) {
						Utils.write(greedWriter, printContent, true);
					}
				} catch (IOException e) {
					e.printStackTrace();
				}
			}
			logLikelihoodOld = logLikelihood;

			if (logLikelihoodError <= GeneralModel.LIKELIHOOD_ERROR_TOLERANCE) break;


			if ((1 + model.lambdaEstIter) < model.maxLambdaEstIterNum) {
				double lambdaError = optimizeLambda(lambdaWriter);
				printContent       = String.format(FORMAT_ESTIMATE_LAMBDA,
							printPrefix,
							model.lambdaEstIter,
							lambdaError,
							Utils.convertToString(model.lambdas, AIRConstants.SPACER_COMMA));
				Utils.println(String.format(FORMAT_ESTIMATE_LAMBDA,
							printPrefix,
							model.lambdaEstIter,
							lambdaError, ""));
				try {
					Utils.write(perpWriter, printContent, true);
					Utils.write(perpWriter, "", true);
					if (greedWriter != null) {
						Utils.write(greedWriter, printContent, true);
						Utils.write(greedWriter, "", true);
					}
				} catch (IOException e) {
					e.printStackTrace();
				}
			}
		}
	}

	private double estimateWithOptimizingAlpha(BufferedWriter perpWriter,
				BufferedWriter alphaWriter) throws AIRException,
								ToolException {
		double alphaError       = Double.MAX_VALUE;
		double logLikelihoodOld = Double.MAX_VALUE;
		String printContent     = null;
		for (;model.alphaEstIter < model.maxAlphaEstIterNum &&
				alphaError > GeneralModel.ALPHA_EST_ERROR_TOLERANCE; model.alphaEstIter ++) {

			// burn-in process
			burnin(perpWriter);
			
			{
				estimateParams();
				model.storeEstimateResult(true);
				model.burninIter = 0;
			}

			if (groundtruthScoreFile != null || groundtruthDisFile != null) {
				try {
					String seval = Measure.evaluate(
							model.getScoreFile(modelOutputPath),
							groundtruthScoreFile,
							GeneralModel.sumWordProb(model.phi),
							groundtruthDisFile,
							AIRConstants.CONFIG_MANAGER.getRatingScaler(),
							true, AIRConstants.DEFAULT_MISS_ASPECT_RATING);
					Utils.writeAndPrint(perpWriter, seval, true);
					if (greedWriter != null) {
						Utils.write(greedWriter, seval, true);
					}
				} catch (ToolException e) {
					e.printStackTrace();
				} catch (IOException e) {
					e.printStackTrace();
				}
			}

			if (validationEvalPool != null) {
				validationEvalPool.evaluate();

				double logLikelihood      = validationEvalPool.getLogLikelihood(EvaluationPool.INDEX_EM);
				double logLikelihoodError = Double.MAX_VALUE;
				if (logLikelihoodOld != Double.MAX_VALUE) {
					logLikelihoodError = Math.abs((logLikelihood - logLikelihoodOld)) / Math.abs(logLikelihoodOld);
				}
				logLikelihoodOld = logLikelihood;

				try {
					printContent = String.format(FORMAT_VALID_LIKELIHOOD_ALPHA,
								printPrefix,
								model.alphaEstIter,
								logLikelihoodError);
					validationEvalPool.outputEvaluation(perpWriter, printContent, true);
					if (greedWriter != null) {
						validationEvalPool.outputEvaluation(greedWriter, printContent, false);
					}
				} catch (IOException e) {
					e.printStackTrace();
				}

				if (logLikelihoodError <= GeneralModel.LIKELIHOOD_ERROR_TOLERANCE) break;
			}


			if ((model.alphaEstIter + 1) < model.maxAlphaEstIterNum) {
				// update alpha
				alphaError   = optimizeAlpha(alphaWriter);
				printContent = String.format(FORMAT_ESTIMATE_ALPHA,
						printPrefix, model.alphaEstIter, alphaError,
						Utils.convertToString(model.alphas, AIRConstants.SPACER_COMMA));
				Utils.println(String.format(FORMAT_ESTIMATE_ALPHA,
						printPrefix, model.alphaEstIter, alphaError, ""));

				try {
					Utils.write(perpWriter, printContent, true);
					if (greedWriter != null) {
						Utils.write(greedWriter, printContent, true);
					}
				} catch (IOException e) {
					e.printStackTrace();
				}
			}
			
		} // end for model.alphaEstIter

		return logLikelihoodOld;
	}

	private void burnin(BufferedWriter perpWriter) throws ToolException {
		long startTime = System.currentTimeMillis();
		int stepLength = 10;
		for (; model.burninIter < model.burninIterNum; model.burninIter ++) {
			sampling();

			if (model.burninIter % stepLength == 0) {
				model.storeModel(false);

				if (debugConvergence) {
					try {
						estimatePhi();
						validationEvalPool.evaluate();
						validationEvalPool.outputEvaluation(perpWriter,
								String.format(FORMAT_VALID_LIKELIHOOD_BURNIN,
									printPrefix, model.burninIter));
					} catch (IOException e) {
						e.printStackTrace();
					}
					if (model.burninIter >= 100) stepLength = 100;
				} else {
					stepLength = 100;
				}

				Utils.println(String.format(FORMAT_ITER_BUILD,
						printPrefix, model.burninIter,
						(System.currentTimeMillis() - startTime) / 1000.0));
				startTime = System.currentTimeMillis();
			}
		} // end for model.burninIter
	}

	private void estimateParams() throws ToolException {
		model.phi   = new double[GeneralModel.SENTIMENT_NUM][model.TOPIC_NUM][model.dictionary.getSize()];
		model.theta = new double[model.data.getDocumentSize()][model.TOPIC_NUM];
		model.omega = new double[model.data.getDocumentSize()][model.TOPIC_NUM][2];
		model.t     = new double[model.data.getDocumentSize()][2];

		double[] beta_sum = model.betaSum;
		double alpha_sum  = Utils.sum(model.alphas);
		double gamma_sum  = Utils.sum(model.gammas);

		for (int iter = 0; iter < model.estimateIterNum; iter ++) {//model.estimateIterNum; iter ++) {
			if (iter % 100 == 0) {
				Utils.println("Estimating Params: iter = " + iter);
			}
			sampling();

			for (int i = 0; i < model.data.getDocumentSize(); i ++) {
				for (int k = 0; k < model.TOPIC_NUM; k ++) {
					double lambda_n[]   = model.getLambdaHat(model.data.getDocuments().get(i).Ri, k);
					double lambda_n_sum = lambda_n[0] + lambda_n[1];

					model.theta[i][k] += ((model.documentTopic[i][k] + model.alphas[k]) /
							(model.documentTopicSum[i] + alpha_sum));
					
					for (int l = 0; l < 2; l ++) {
						model.omega[i][k][l] +=
								((model.documentTopicSentiment[i][k][l] + lambda_n[l]) /
								(model.documentTopicSentimentSum[i][k] + lambda_n_sum));
					}
				}
				for (int j = 0; j < model.t[0].length; j ++) {
					model.t[i][j] += ((model.documentSentiment[i][j] + model.gammas[j]) /
							(model.documentSentimentSum[i] + gamma_sum));
				}
			}
			for (int l = 0; l < GeneralModel.SENTIMENT_NUM; l ++) {
				for (int k = 0; k < model.TOPIC_NUM; k ++) {
					for (int v = 0; v < model.dictionary.getSize(); v ++) {
						model.phi[l][k][v] += ((model.topicSentimentWord[k][l][v] + model.betas[k][v]) /
								(model.topicSentimentWordSum[k][l] + beta_sum[k]));
					}
				}
			}
		}
		for (int i = 0; i < model.data.getDocumentSize(); i ++) {
			for (int k = 0; k < model.TOPIC_NUM; k ++) {
				model.theta[i][k] /= model.estimateIterNum;

				for (int l = 0; l < 2; l ++) {
					model.omega[i][k][l] /= model.estimateIterNum;
				}
			}
			for (int j = 0; j < model.t[0].length; j ++) {
				model.t[i][j] /= model.estimateIterNum;
			}
		}
		for (int l = 0; l < GeneralModel.SENTIMENT_NUM; l ++) {
			for (int k = 0; k < model.TOPIC_NUM; k ++) {
				for (int v = 0; v < model.dictionary.getSize(); v ++) {
					model.phi[l][k][v] /= model.estimateIterNum;
				}
			}
		}
	}

	private void estimatePhi() throws ToolException {
		model.phi         = new double[GeneralModel.SENTIMENT_NUM][model.TOPIC_NUM][model.dictionary.getSize()];
		double[] beta_sum = model.betaSum;

		for (int l = 0; l < GeneralModel.SENTIMENT_NUM; l ++) {
			for (int k = 0; k < model.TOPIC_NUM; k ++) {
				for (int v = 0; v < model.dictionary.getSize(); v ++) {
					model.phi[l][k][v] = ((model.topicSentimentWord[k][l][v] + model.betas[k][v]) /
							(model.topicSentimentWordSum[k][l] + beta_sum[k]));
				}
			}
		}
	}

	private void sampling() throws ToolException {
		for (int m = 0; m < model.data.getDocumentSize(); m ++) {
			for (int n = 0; n < model.data.getDocuments().get(m).wordNum; n ++) {
				int[] res = sampling(m, n, model.data.getDocuments().get(m).Ri);
				model.data.getDocuments().get(m).topics[n]     = res[0];
				model.data.getDocuments().get(m).sentiments[n] = res[1];
			}
		}
	}

	/*
	 * Sampling topic and sentiment index n-th word of m-th document.
	 * Returns 2-dimension array, the first element in the array is
	 * topic and the second element in the array is the sentiment index.
	 */
	private int[] sampling(int m, int n, double Rm) throws ToolException {
		int topic     = model.data.getDocuments().get(m).topics[n];
		int word      = model.data.getDocuments().get(m).words[n];
		int sentiment = model.data.getDocuments().get(m).sentiments[n];

		model.topicSentimentWord[topic][sentiment][word] --;
		model.topicSentimentWordSum[topic][sentiment] --;
		model.documentTopic[m][topic] --;
		model.documentTopicSum[m] --;
		model.documentSentiment[m][(sentiment == 0 ? 0 : 1)] --;
		model.documentSentimentSum[m] --;
		if (sentiment > 0) {
			model.documentTopicSentiment[m][topic][sentiment - 1] --;
			model.documentTopicSentimentSum[m][topic] --;
		}

		double alpha_sum  = Utils.sum(model.alphas);
		double[] beta_sum = model.betaSum;

		double p[][] = new double[model.TOPIC_NUM][GeneralModel.SENTIMENT_NUM];
		double sum   = 0.0;
		for (topic = 0; topic < model.TOPIC_NUM; topic ++) {
			double[] lambda_n = model.getLambdaHat(Rm, topic);
			double lambda_n_sum = Utils.sum(lambda_n);

			for (sentiment = 0; sentiment < GeneralModel.SENTIMENT_NUM; sentiment ++) {
				double commonItem = (((model.topicSentimentWord[topic][sentiment][word] + model.betas[topic][word]) /
							(model.topicSentimentWordSum[topic][sentiment] + beta_sum[topic]))
							*
							((model.documentTopic[m][topic]) + model.alphas[topic]) /
								(model.documentTopicSum[m] + alpha_sum));
				if (sentiment == 0) {
					sum += (
						p[topic][sentiment] = commonItem *
							((model.documentSentiment[m][0] + model.gammas[0]) /
								1.0)//(model.documentSentimentSum[m] + gamma_sum));
						);
				} else {
					sum += (
						p[topic][sentiment] = commonItem *
							((model.documentSentiment[m][1] + model.gammas[1]) /
								1.0)//(model.documentSentimentSum[m] + gamma_sum))
							*
							((model.documentTopicSentiment[m][topic][sentiment - 1] + lambda_n[sentiment - 1])
								/ (model.documentTopicSentimentSum[m][topic] + lambda_n_sum))
						);
				}
			}
		}

		double u        = Math.random();
		double prev_sum = 0.0;
		for (topic = 0; topic < model.TOPIC_NUM; topic ++) {
			for (sentiment = 0; sentiment < GeneralModel.SENTIMENT_NUM; sentiment ++) {
				p[topic][sentiment] /= sum;
				p[topic][sentiment] += prev_sum;
				prev_sum = p[topic][sentiment];

				if (u < p[topic][sentiment]) break;
			}
			if (sentiment < GeneralModel.SENTIMENT_NUM) {
				break;
			}
		}
		if (topic >= model.TOPIC_NUM ||
				sentiment >= GeneralModel.SENTIMENT_NUM) {
			// never reaches{
			Utils.err(topic + ", u=" + u + ", sum=" + prev_sum);
		}

		model.topicSentimentWord[topic][sentiment][word] ++;
		model.topicSentimentWordSum[topic][sentiment] ++;
		model.documentTopic[m][topic] ++;
		model.documentTopicSum[m] ++;
		model.documentSentiment[m][(sentiment == 0 ? 0 : 1)] ++;
		model.documentSentimentSum[m] ++;
		if (sentiment > 0) {
			model.documentTopicSentiment[m][topic][sentiment - 1] ++;
			model.documentTopicSentimentSum[m][topic] ++;
		}

		return new int[] {topic, sentiment};
	}

	private double optimizeAlpha(BufferedWriter writer)
						throws ToolException {
		double[] alphas_old   = Arrays.copyOf(model.alphas, model.alphas.length);
		double error          = 0.0;
		int iter              = 0;
		do {
			error = updateAlpha();

			try {
				if (writer != null) {
					String res = String.format(FORMAT_UPDATE_ALPHA,
							printPrefix,
							iter, error,
							Utils.convertToString(
								model.alphas,
								AIRConstants.SPACER_COMMA));
					Utils.write(writer, res, true);
					//Utils.println(res);
				}
			} catch (IOException e) {
				e.printStackTrace();
			}
			iter ++;
		} while(error > GeneralModel.ALPHA_EST_ERROR_TOLERANCE);

		/*Utils.println("\t Update Alpha: " + Utils.convertToString(
					model.alphas, AIRConstants.SPACER_COMMA));*/

		return Utils.sumOfDiffAbs(model.alphas, alphas_old);
	}

	/*
	 * Update rule :
	 *                         sum_i(diamma(n_i_k + alpha_k) - digamma(alpha_k))
	 * alpha_k_new = alpha_k x -------------------------------------------------
	 *                         sum_i(digamma(n_i + sum_k(alpha_k)) - digamma(sum_k(alpha_k)))
	 */
	private double updateAlpha() throws ToolException {
		double[] alphas_old = Arrays.copyOf(model.alphas, model.alphas.length);
		double alpha_sum    = Utils.sum(alphas_old);
		for (int topic = 0; topic < model.TOPIC_NUM; topic ++) {
			double denominator = 0.0;
			double numerator   = 0.0;
			for (int doc = 0; doc < model.data.getDocumentSize(); doc ++) {
				numerator   += Gamma.digamma(model.documentTopic[doc][topic] + model.alphas[topic]) -
							Gamma.digamma(model.alphas[topic]);
				denominator += Gamma.digamma(model.documentTopicSum[doc] + alpha_sum) -
						Gamma.digamma(alpha_sum);
			}
			alpha_sum           -= model.alphas[topic];
			model.alphas[topic] *= (numerator / denominator);
			alpha_sum           += model.alphas[topic];
		}

		return Utils.sumOfDiffAbs(model.alphas, alphas_old);
	}

	private double optimizeLambda(BufferedWriter lambdaWriter) throws ToolException {
		double[] lambdaOld = Arrays.copyOf(model.lambdas, model.lambdas.length);
		double lambdaError = Double.MAX_VALUE;
		int iter           = 0;
		do {
			lambdaError = updateLambda();

			if (lambdaWriter != null) {
				try {
					Utils.write(lambdaWriter, String.format(
							FORMAT_UPDATE_LAMBDA,
							printPrefix, iter, lambdaError,
							Utils.convertToString(model.lambdas, ",")),
								true);
				} catch (IOException e) {
					
				}
			}
			iter ++;
		} while (lambdaError > GeneralModel.LAMBDA_EST_ERROR_TOLERANCE);

		//model.updateLambdaHat();

		return Utils.sumOfDiffAbs(model.lambdas, lambdaOld);
	}

	/*
	 * Update Rule:
	 * 
	 * a_i_1 = Ri*lambda_k*( digamma(n_i_k_1 + Ri*lambda_k) - digamma(Ri*lambda_k) )
	 * 
	 * a_i_2 = (1-Ri)*lambda_k*( digamma(n_i_k_2 + (1-Ri)*lambda_k) - digamma((1-Ri) * lambda_k) )
	 * 
	 *                               sum_i(a_i_1 + a_i_2)
	 * lambda_k_new = ------------------------------------------------------
	 *                sum_i( digamma(n_i_k + lambda_k) - digamma(lambda_k) )
	 */
	private double updateLambda() throws ToolException {
		double[] lambdasOld = Arrays.copyOf(model.lambdas, model.lambdas.length);
		for (int topic = 0; topic < model.TOPIC_NUM; topic ++) {
			double lambdaOld   = model.lambdas[topic];
			double numerator  = 0.0;
			double denominator = 0.0;
			for (int doc = 0; doc < model.data.getDocumentSize(); doc ++) {
				double etaLambda      = model.data.getDocuments().get(doc).Ri * lambdaOld;
				double inverEtaLambda = (1 - model.data.getDocuments().get(doc).Ri) * lambdaOld;
				numerator += etaLambda * (
							Gamma.digamma(model.documentTopicSentiment[doc][topic][0] + etaLambda) -
							Gamma.digamma(etaLambda)
					     ) + inverEtaLambda * (
							Gamma.digamma(model.documentTopicSentiment[doc][topic][1] + inverEtaLambda) -
							Gamma.digamma(inverEtaLambda)
						);
				denominator += Gamma.digamma(model.documentTopicSentimentSum[doc][topic] + lambdaOld) -
						Gamma.digamma(lambdaOld);
			}
			model.lambdas[topic] = numerator / denominator;
		}
		return Utils.sumOfDiffAbs(model.lambdas, lambdasOld);
	}
}
