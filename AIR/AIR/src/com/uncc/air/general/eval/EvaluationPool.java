package com.uncc.air.general.eval;

import java.io.BufferedWriter;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;

import com.lhy.tool.ToolException;
import com.lhy.tool.util.Utils;
import com.uncc.air.AIRConstants;
import com.uncc.air.AIRException;
import com.uncc.topicmodel.TopicModelException;
import com.uncc.topicmodel.data.Dataset;

/**
 * @author Huayu Li
 */
public class EvaluationPool {
	private static final String STRING_LOAD_DATA = "Loading %s data...";

	public static final int INDEX_EM       = 0;
	public static final int INDEX_SAMPLING = 1;

	private List<Evaluation> evalList      = null;
	private List<Double> logLikelihoodList = null;

	public EvaluationPool(String title, File testDataFile,
			File outputPath, EvaluationModel evalModel,
				boolean isRestore) throws TopicModelException {
		this(title, evalModel.getDataset(testDataFile, evalModel.getDictionary(),
				new File(outputPath, AIRConstants.FILE_NAME_DIC_MAP),
					isRestore), evalModel);
	}

	public EvaluationPool(String title, Dataset testData,
						EvaluationModel evalModel)  {
		Utils.println(String.format(STRING_LOAD_DATA, title));

		if (testData != null) {
			testData.getParams().list(System.out);
		}

		evalList = new LinkedList<Evaluation>();
		//evalList.add(new LeftToRightEvaluation(testData, evalModel));
		evalList.add(new LeftToRightEMEvaluation(testData, evalModel));
	}

	public void evaluate() throws ToolException{
		if (logLikelihoodList == null) logLikelihoodList = new ArrayList<Double>();
		else logLikelihoodList.clear();

		for (Evaluation eval : evalList) {
			logLikelihoodList.add(eval.modelLogLikelihood());
		}
	}

	public double getLogLikelihood(int evalType) throws AIRException {
		if (logLikelihoodList == null || evalType < 0 ||
				evalType >= logLikelihoodList.size())
			throw new AIRException("No specified evaluation type.");

		return logLikelihoodList.get(evalType);
	}

	public double getPerplexity(int evalType) throws AIRException {
		if (logLikelihoodList == null || evalType < 0 ||
				evalType >= logLikelihoodList.size())
			throw new AIRException("No specified evaluation type.");

		return perplexity(logLikelihoodList.get(evalType));
	}

	public void outputEvaluation(BufferedWriter perplexityWriter,
				String printPrefix) throws IOException,
								ToolException {
		outputEvaluation(perplexityWriter, printPrefix, true);
	}

	public void outputEvaluation(BufferedWriter perplexityWriter,
			String printPrefix, boolean print) throws IOException,
							ToolException {
		if (! Utils.isEmpty(evalList)) {
			StringBuffer buffer = new StringBuffer();
			buffer.append(printPrefix).append("; ");

			for (int i = 0; i < evalList.size(); i ++) {
				buffer.append(String.format("[%s] LogLikelihood = %s, Perplexity = %s",
						evalList.get(i).getTitle(),
						logLikelihoodList.get(i),
						perplexity(logLikelihoodList.get(i))))
						.append("; ");
			}

			if (print) {
				Utils.writeAndPrint(perplexityWriter, buffer.toString(), true);
			} else {
				Utils.write(perplexityWriter, buffer.toString(), true);
			}
		}
	}

	private double perplexity(double logLikelihood) {
		return Math.exp(-1.0 * logLikelihood);
	}
}

