package com.uncc.air.general.eval;


import com.lhy.tool.ToolException;
import com.lhy.tool.util.Utils;
import com.uncc.air.AIRConstants;
import com.uncc.topicmodel.data.Dataset;
import com.uncc.topicmodel.data.Document;

/**
 * @author Huayu Li
 */
public class LeftToRightEvaluation extends Evaluation {
	private static final int PARTICLE_NUM = AIRConstants.CONFIG_MANAGER.getParticleNum();

	private int[] topicCount;
	private int[] topicCountSum;
	private int[] sentimentCount;
	private int[] sentimentCountSum;
	private int[][] topicSentimentCount;
	private int[] topicSentimentCountSum;
	private double alpha_sum;
	private double gamma_sum;
	private double[][] lambda_hat;
	private double[] lambda_hat_sum ;

	public LeftToRightEvaluation(Dataset testData, EvaluationModel evalModel){
		super(testData, evalModel);
	}

	@Override
	public String getTitle() {
		return "Sampling-LeftToRight";
	}

	@Override
	public double modelLogLikelihood(int doc) throws ToolException {
		if (doc % 1000 == 0) printStep(doc);

		Document document      = testData.getDocuments().get(doc);
		double logLikelihood   = 0.0;
		topicCount             = new int[evalModel.getTopicNum()];
		topicCountSum          = new int[1];
		sentimentCount         = new int[2];
		sentimentCountSum      = new int[1];
		topicSentimentCount    = new int[evalModel.getTopicNum()][2];
		topicSentimentCountSum = new int[evalModel.getTopicNum()];
		alpha_sum              = Utils.sum(evalModel.getAlphas());
		gamma_sum              = Utils.sum(evalModel.getGammas());
		lambda_hat             = new double[evalModel.getTopicNum()][2];
		lambda_hat_sum         = new double[evalModel.getTopicNum()];
		for (int topic = 0; topic < evalModel.getTopicNum(); topic ++) {
			lambda_hat[topic]     = evalModel.getLambdaHat(document.Ri, topic);
			lambda_hat_sum[topic] = Utils.sum(lambda_hat[topic]);
		}

		for (int position = 0; position < document.wordNum; position ++) {
			double likelihood = 0.0;
			for (int particle = 0; particle < PARTICLE_NUM; particle ++) {
				for (int pos = 0; pos < position; pos ++) {
					sampling(document, pos, true);
				}
				likelihood += calculateLikelihood(document, position);
				sampling(document, position, false);
			}
			likelihood    /= PARTICLE_NUM;
			logLikelihood += Math.log(likelihood);
		}
		return logLikelihood;
	}

	private void sampling(Document document, int pos,
					boolean notIncludeCurrent) {
		int word      = document.words[pos];
		int topic     = document.topics[pos];
		int sentiment = document.sentiments[pos];

		if (notIncludeCurrent) {
			topicCount[topic] --;
			topicCountSum[0] --;
			sentimentCount[sentiment == 0 ? 0 : 1] --;
			sentimentCountSum[0] --;
			if (sentiment > 0) {
				topicSentimentCount[topic][sentiment - 1] --;
				topicSentimentCountSum[topic] --;
			}
		}

		double[][] prob = new double[evalModel.getTopicNum()][evalModel.getSentimentNum()];
		double sum      = 0.0;
		for (topic = 0; topic < evalModel.getTopicNum(); topic ++) {
			double theta = topicCount[topic] + evalModel.getAlphas()[topic];
			for (sentiment = 0; sentiment < evalModel.getSentimentNum(); sentiment ++) {
				if (sentiment == 0) {
					sum += (
						prob[topic][sentiment] = 
							evalModel.getWordProb()[sentiment][topic][word] * theta *
							(sentimentCount[0] + evalModel.getGammas()[0])
						);
				} else {
					sum += (
						prob[topic][sentiment] =
							evalModel.getWordProb()[sentiment][topic][word] * theta *
							(sentimentCount[1] + evalModel.getGammas()[1]) *
							(topicSentimentCount[topic][sentiment - 1] + lambda_hat[topic][sentiment - 1]) /
							(topicSentimentCountSum[topic] + lambda_hat_sum[topic])
						);
				}
			}
		}

		double u       = Math.random();
		double prevSum = 0.0;
		for (topic = 0; topic < evalModel.getTopicNum(); topic ++) {
			for (sentiment = 0; sentiment < evalModel.getSentimentNum(); sentiment ++) {
				prob[topic][sentiment] /= sum;
				prob[topic][sentiment] += prevSum;
				prevSum                 = prob[topic][sentiment];

				if (prob[topic][sentiment] > u) break;
			}
			if (sentiment < evalModel.getSentimentNum()) {
				break;
			}
		}

		if (topic >= evalModel.getTopicNum() ||
				sentiment >= evalModel.getSentimentNum()) {
			// never reaches
			if (u >= prob[topic][sentiment]) {
				Utils.err(topic + ", u=" + u + ", sum=" + prevSum);
			}
		}

		topicCount[topic] ++;
		topicCountSum[0] ++;
		sentimentCount[sentiment == 0 ? 0 : 1] ++;
		sentimentCountSum[0] ++;
		if (sentiment > 0) {
			topicSentimentCount[topic][sentiment - 1] ++;
			topicSentimentCountSum[topic] ++;
		}

		document.topics[pos]     = topic;
		document.sentiments[pos] = sentiment;
	}

	private double calculateLikelihood(Document document, int pos) {
		double likelihood = 0.0;
		int word          = document.words[pos];
		for (int topic = 0; topic < evalModel.getTopicNum(); topic ++) {
			double theta = (topicCount[topic] + evalModel.getAlphas()[topic]) /
						(topicCountSum[0] + alpha_sum);
			for (int sentiment = 0; sentiment < evalModel.getSentimentNum(); sentiment ++) {
				if (sentiment == 0) {
					likelihood += evalModel.getWordProb()[sentiment][topic][word] * theta *
							(sentimentCount[0] + evalModel.getGammas()[0]) /
							(sentimentCountSum[0] + gamma_sum);
				} else {
					likelihood += evalModel.getWordProb()[sentiment][topic][word] * theta *
							(sentimentCount[1] + evalModel.getGammas()[1]) /
							(sentimentCountSum[0] + gamma_sum) *
							(topicSentimentCount[topic][sentiment - 1] + lambda_hat[topic][sentiment - 1]) /
							(topicSentimentCountSum[topic] + lambda_hat_sum[topic]);
				}
			}
		}
		return likelihood;
	}
}
