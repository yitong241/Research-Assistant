package com.uncc.air.general.eval;

import com.lhy.tool.ToolException;
import com.lhy.tool.util.Utils;
import com.uncc.topicmodel.data.Dataset;

/**
 * @author Huayu Li
 */
public abstract class Evaluation {
	protected static final String STRING_NOTI_STEP =
			"[%s] Calculate perplexity of %s-th document.";

	protected EvaluationModel evalModel = null;
	protected Dataset testData          = null;

	public Evaluation(Dataset testData, EvaluationModel evalModel) {
		this.evalModel = evalModel;
		this.testData  = testData;
	}

	public Evaluation() {
		// do nothing
	}

	public abstract double modelLogLikelihood (int doc) throws ToolException;
	public abstract String getTitle();

	public String getEvaluateResult() throws ToolException {
		double likelihood = modelLogLikelihood ();
		double perplexity = perplexity(likelihood);

		return String.format("[%s] LogLikelihood = %s, Perplexity = %s",
					getTitle(), likelihood, perplexity);
	}

	public double modelLogLikelihood () throws ToolException{
		double logLikelihood = 0.0;
		int tokenNum         = 0;
		for (int doc = 0; doc < testData.getDocumentSize(); doc ++) {
			logLikelihood += modelLogLikelihood(doc);
			tokenNum      += testData.getDocuments().get(doc).wordNum;
		}

		return logLikelihood / tokenNum;
	}

	public double perplexity(double likelihood) {
		return Math.exp(-1.0 * likelihood);
	}

	public void printStep(int doc) {
		Utils.println(String.format(STRING_NOTI_STEP, getTitle(), doc));
	}

	/*public static double exp(double exponent) {
		return Math.pow(2, exponent);
	}

	public static double log(double value) {
		return Math.log(value) / Math.log(2);
	}*/
}

