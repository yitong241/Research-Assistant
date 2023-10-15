package com.uncc.air.general.eval;

import com.uncc.topicmodel.TopicModelException;
import com.uncc.topicmodel.data.Dataset;
import com.uncc.topicmodel.data.Dictionary;
import java.io.File;

/**
 * @author Huayu Li
 */
public interface EvaluationModel {
	public int getTopicNum();
	public int getSentimentNum();
	public double[] getAlphas();
	public double[] getGammas();
	public double[] getLambdaHat(double rating, int topic);
	public double[][][] getWordProb();
	public Dictionary getDictionary();
	public Dataset getDataset(File datasetFile, Dictionary dictionary,
				File dicFile, boolean isRestore)
						throws TopicModelException;
}
