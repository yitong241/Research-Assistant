package com.uncc.air.general.eval;

import java.util.Arrays;

import com.lhy.tool.ToolException;
import com.lhy.tool.util.Utils;
import com.uncc.topicmodel.data.Dataset;
import com.uncc.topicmodel.data.Document;

/**
 * @author Huayu Li
 */
public class LeftToRightEMEvaluation extends Evaluation {

	private static final double ALPHA_ERROR_TOLERANCE = 1.0e-5;
	private static final double PT_ERROR_TOLERANCE    = 1.0e-5;
	private static final double OMEGA_ERROR_TOLERANCE = 1.0e-5;

	private double[] thetas;
	private double[][] omegas;
	private double[]   pt;
	private double[][] lambdaHat;
	private double alphaSum;
	private double gammaSum;
	private double[] lambdaHatSum;

	public LeftToRightEMEvaluation(Dataset testData, EvaluationModel evalModel){
		super(testData, evalModel);
	}

	@Override
	public String getTitle() {
		return "EM";
	}

	public void init() {}

	@Override
	public double modelLogLikelihood (int doc) throws ToolException {
		if (doc % 10000 == 0) printStep(doc);

		double logLikelihood = 0.0;
		Document document    = testData.getDocuments().get(doc);
		thetas               = new double[evalModel.getTopicNum()];
		omegas               = new double[evalModel.getTopicNum()][2];
		pt                   = new double[2];
		alphaSum             = Utils.sum(evalModel.getAlphas());
		gammaSum             = Utils.sum(evalModel.getGammas());
		lambdaHat            = new double[evalModel.getTopicNum()][2];
		lambdaHatSum         = new double[evalModel.getTopicNum()];
		for (int topic = 0; topic < evalModel.getTopicNum(); topic ++) {
			lambdaHat[topic]    = evalModel.getLambdaHat(document.Ri, topic);
			lambdaHatSum[topic] = Utils.sum(lambdaHat[topic]);
		}

		for (int position = 0; position < testData.getDocuments().get(doc).wordNum; position ++) {
			updateParams(document, position);

			logLikelihood += Math.log(calculateLikelihood(document, position));
		}

		return logLikelihood;
	}

	private void updateParams(Document document, int position) throws ToolException {
		double[][][] posterior  = null;
		double[] thetasOld      = null;
		double[] ptOld          = null;
		double[][] omegasOld    = null;
		double thetasError      = Double.MAX_VALUE;
		double ptError          = Double.MAX_VALUE;
		double omegaError       = Double.MAX_VALUE;
		do {
			thetasOld = Arrays.copyOf(thetas, thetas.length);
			ptOld     = Arrays.copyOf(pt, pt.length);
			omegasOld = Utils.copyOf(omegas);
			posterior = getPosteriors(document, position);

			Utils.fills(thetas, 0.0);
			Utils.fills(pt, 0.0);
			Utils.fills(omegas, 0.0);
			double[] sum = new double[evalModel.getTopicNum()];
			for (int pos = 0; pos < position; pos ++) {
				for (int topic = 0; topic < evalModel.getTopicNum(); topic ++) {
					pt[0]            += posterior[pos][topic][0];
					omegas[topic][0] += posterior[pos][topic][1];
					sum[topic]       += omegas[topic][0] + posterior[pos][topic][2];

					for (int sentiment = 0; sentiment < evalModel.getSentimentNum(); sentiment ++) {
						thetas[topic] += posterior[pos][topic][sentiment];
					}
				}
			}
			for (int topic = 0; topic < evalModel.getTopicNum(); topic ++) {
				thetas[topic]    = (thetas[topic] + evalModel.getAlphas()[topic]) /
							(position + alphaSum);
				omegas[topic][0] = (omegas[topic][0] + lambdaHat[topic][0]) /
						(sum[topic] + lambdaHatSum[topic]);
				omegas[topic][1] = 1 - omegas[topic][0];
			}
			pt[0]   = (pt[0] + evalModel.getGammas()[0]) /
					(position + gammaSum);
			pt[1]   = 1 - pt[0];

			thetasError = Utils.sumOfDiffAbs(thetas, thetasOld);
			ptError     = Utils.sumOfDiffAbs(pt, ptOld);
			omegaError  = sumOfDiffAbs(omegas, omegasOld);

		} while (thetasError > ALPHA_ERROR_TOLERANCE ||
				ptError > PT_ERROR_TOLERANCE ||
				omegaError > OMEGA_ERROR_TOLERANCE);
	}

	private double calculateLikelihood(Document document, int position) {
		double likelihood = 0.0;
		int word          = document.words[position];

		for (int topic = 0; topic < evalModel.getTopicNum(); topic ++) {
			for (int sentiment = 0; sentiment < evalModel.getSentimentNum(); sentiment ++) {
				if (sentiment == 0) {
					likelihood += evalModel.getWordProb()[sentiment][topic][word] *
							thetas[topic] *
							pt[0];
				} else {
					likelihood += evalModel.getWordProb()[sentiment][topic][word] *
							thetas[topic] *
							pt[1] *
							omegas[topic][sentiment - 1];
				}
			}
		}

		return likelihood;
	}


	private double[][][] getPosteriors(Document document, int position) {
		double[][][] probs = new double[position][evalModel.getTopicNum()][evalModel.getSentimentNum()];
		for (int pos = 0; pos < position; pos ++) {
			probs[pos] = getPosterior(document, pos);
		}
		return probs;
	}
	private double[][] getPosterior(Document document, int position) {
		double[][] prob = new double[evalModel.getTopicNum()][evalModel.getSentimentNum()];
		int word        = document.words[position];

		double sum = 0.0;
		for (int topic = 0; topic < evalModel.getTopicNum(); topic ++) {
			for (int sentiment = 0; sentiment < evalModel.getSentimentNum(); sentiment ++) {
				if (sentiment == 0) {
					prob[topic][sentiment] =
							evalModel.getWordProb()[sentiment][topic][word] *
							thetas[topic] * pt[0];	
				} else {
					prob[topic][sentiment] =
							evalModel.getWordProb()[sentiment][topic][word] *
							thetas[topic] * pt[1] *
							omegas[topic][sentiment - 1];
				}
				sum += prob[topic][sentiment];
			}
		}

		for (int topic = 0; topic < evalModel.getTopicNum(); topic ++) {
			for (int sentiment = 0; sentiment < evalModel.getSentimentNum(); sentiment ++) {
				prob[topic][sentiment] /= sum;
			}
		}

		return prob;
	}

	private double sumOfDiffAbs(double[][] array, double[][] arrayOld) {
		double sum = 0;
		for (int i = 0; i < array.length; i ++) {
			for (int j = 0; j < array[i].length; j ++) {
				sum += Math.abs(array[i][j] - arrayOld[i][j]);
			}
		}
		return sum;
	}
}
