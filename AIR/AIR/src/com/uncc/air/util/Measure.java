package com.uncc.air.util;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import com.lhy.tool.ToolException;
import com.lhy.tool.util.Utils;
import com.uncc.air.AIRConstants;
import com.uncc.air.AIRException;
import com.uncc.air.ErrorMessage;
import com.uncc.air.data.AIRDataset;
import com.uncc.air.data.RatingScaler;
import com.uncc.topicmodel.TopicModelException;
import com.uncc.topicmodel.data.Dictionary;
import com.uncc.topicmodel.data.Document;

/**
 * @author Huayu Li
 */
public class Measure {

	public static String evaluate(File predictScoreFile,
				File groundtruthScoreFile,
				double[][] predictTopicWordDistribution,
				File groundtruthTopicWordDistributionFile,
				RatingScaler ratingScaler,
				boolean checkMissAspect,
				double defaultMissAspectValue)
						throws ToolException {
		final String outputFormat = "MSE = %s, PearsonCorrAspect = %s, " +
				"PearsonCorrReview=%s, " +
				"KendallTauRankCorr = %s, Misaspect = %s, " +
				"nDCGaspect = %s, KLDivergence=%s";
		final String defaultNull  = "--";

		double[][] groundtruthTopicWordDistribution = null;
		double[][] groundtruthScore                 = null;
		double[][] predictScore                     = null;

		if (groundtruthTopicWordDistributionFile != null) {
			groundtruthTopicWordDistribution = Utils.load2Array(
							groundtruthTopicWordDistributionFile,
							AIRConstants.SPACER_SPACE);
		}
		if (groundtruthScoreFile != null) {
			groundtruthScore = Utils.load2Array(groundtruthScoreFile,
						AIRConstants.SPACER_SPACE);
		}
		if (predictScoreFile != null) {
			predictScore = Utils.load2Array(predictScoreFile,
					AIRConstants.SPACER_SPACE);
			if (predictScore != null) {
				for (int doc = 0; doc < predictScore.length; doc ++) {
					for (int i = 0; i < predictScore[doc].length; i ++) {
						if (i == 0) continue; // skip the overall rating
						predictScore[doc][i] = cut(predictScore[doc][i], i - 1, ratingScaler);
					}
				}
			}
		}

		Double mse               = null;
		Double pearsonCorrAspect = null;
		Double pearsonCorrReview = null;
		Double kendallTauCorr    = null;
		Double misaspect         = null;
		Double nDCGaspect        = null;
		Double kldivergence      = null;
		try {
			if (predictScore != null && groundtruthScore != null) {
				mse               = mse(predictScore,
						groundtruthScore, checkMissAspect,
						defaultMissAspectValue);
				pearsonCorrAspect = pearsonCorrelationAspect(
						predictScore, groundtruthScore,
						checkMissAspect, defaultMissAspectValue);
				pearsonCorrReview = pearsonCorrelationReview(
						predictScore, groundtruthScore,
						checkMissAspect, defaultMissAspectValue);
				kendallTauCorr    = KendallTauRankCorrelation(
						predictScore, groundtruthScore,
						checkMissAspect, defaultMissAspectValue);
				misaspect         = Misaspect(predictScore,
						groundtruthScore, checkMissAspect,
						defaultMissAspectValue);
				nDCGaspect        = nDCG(predictScore,
						groundtruthScore, checkMissAspect,
						defaultMissAspectValue);
			}
			if (groundtruthTopicWordDistribution != null) {
				kldivergence = KLDivergence(predictTopicWordDistribution,
						groundtruthTopicWordDistribution);
			}
		} catch (ToolException e) {
			e.printStackTrace();
		}

		return String.format(outputFormat,
				(mse               == null ? defaultNull : mse),
				(pearsonCorrAspect == null ? defaultNull : pearsonCorrAspect),
				(pearsonCorrReview == null ? defaultNull : pearsonCorrReview),
				(kendallTauCorr    == null ? defaultNull : kendallTauCorr),
				(misaspect         == null ? defaultNull : misaspect),
				(nDCGaspect        == null ? defaultNull : nDCGaspect),
				(kldivergence      == null ? defaultNull : kldivergence));
	}

	public static double KLDivergence(double[][] predictTopicWordDistribution,
				File groundtruthWordDistributionFile) throws ToolException {
		return KLDivergence(predictTopicWordDistribution,
				Utils.load2Array(groundtruthWordDistributionFile,
						AIRConstants.SPACER_TAB));
	}

	public static double KLDivergence(File predictWordDistributionFile,
					File groundtruthWordDistributionFile)
							throws ToolException {
		return KLDivergence(Utils.load2Array(predictWordDistributionFile, AIRConstants.SPACER_TAB),
				Utils.load2Array(groundtruthWordDistributionFile, AIRConstants.SPACER_TAB));
	}

	public static double KLDivergence(double[][] predictTopicWordDistribution,
				double[][] groundtruthTopicWordDistribution)
							throws ToolException {
		if (Utils.isEmpty(predictTopicWordDistribution) ||
				Utils.isEmpty(groundtruthTopicWordDistribution) ||
				predictTopicWordDistribution.length != groundtruthTopicWordDistribution.length ||
				predictTopicWordDistribution[0].length != groundtruthTopicWordDistribution[0].length) {
			throw new ToolException ("Word distribution is empty or does not matach.");
		}

		//normalize(predictTopicWordDistribution);
		//normalize(groundtruthTopicWordDistribution);

		double[][] kldArray = new double[predictTopicWordDistribution.length][groundtruthTopicWordDistribution.length];
		for (int topic = 0; topic < predictTopicWordDistribution.length; topic ++) {
			for (int k = 0; k < groundtruthTopicWordDistribution.length; k ++) {
				kldArray[topic][k] = KLDivergence(
							predictTopicWordDistribution[topic],
							groundtruthTopicWordDistribution[k]);
			}
		}

		Utils.print(kldArray);

		int[][] assignment = KuhnMunkres.hgAlgorithm(kldArray, "min");
		double kldSum      = 0.0;
		Utils.print(assignment, ",");

		for (int i = 0; i < assignment.length; i ++) {
			kldSum += kldArray[assignment[i][0]][assignment[i][1]];
		}
		return kldSum;
	}

	private static double KLDivergence(double[] predictDistribution,
					double[] groundtruthDistribution)
							throws ToolException {
		if (Utils.isEmpty(predictDistribution) ||
				Utils.isEmpty(groundtruthDistribution) ||
				predictDistribution.length != groundtruthDistribution.length) {
			throw new ToolException ("Distributions do not match.");
		}
		double klDivergence = 0.0;
		for (int i = 0; i < predictDistribution.length; i ++) {
			if (predictDistribution[i] == 0.0) {
				predictDistribution[i] = Math.exp(-100.0);
			}
			if (groundtruthDistribution[i] == 0.0) {
				groundtruthDistribution[i] = Math.exp(-100.0);
			}
			klDivergence += groundtruthDistribution[i] * (
					Math.log(groundtruthDistribution[i]) -
					Math.log(predictDistribution[i]) );
		}
		return klDivergence;
	}

	
	// the predict score is what have been cut instead of the original one
	private static double KendallTauRankCorrelation(double[][] predictScore,
			double[][] groundtruthScore, boolean checkMissAspect,
				double defaultMissAspectValue) throws ToolException {
		if (Utils.isEmpty(predictScore) || Utils.isEmpty(groundtruthScore) ||
				predictScore.length != groundtruthScore.length ||
				predictScore[0].length != groundtruthScore[0].length) {
			throw new ToolException (ErrorMessage.ERROR_NO_ARGS);
		}

		double misorderRate = 0.0;
		double num          = 0.0;
		for (int doc = 0; doc < predictScore.length; doc ++) {
			double[] predict          = predictScore[doc];
			double[] groundtruth      = groundtruthScore[doc];
			double[] predictArray     = null;
			double[] groundtruthArray = null;
			if (checkMissAspect) {
				int count = 0;
				for (int i = 1; i < groundtruth.length; i ++) {
					if (groundtruth[i] != defaultMissAspectValue) {
						count ++;
					}
				}
				predictArray     = new double[count];
				groundtruthArray = new double[count];
				int index        = 0;
				for (int i = 1; i < groundtruth.length; i ++) {
					if (groundtruth[i] != defaultMissAspectValue) {
						predictArray[index]     = predict[i];
						groundtruthArray[index] = groundtruth[i];
						index ++;
					}
				}
			} else {
				predictArray     = Utils.copyOf(predictScore[doc],
					1, predictScore[doc].length - 1); // skip the overall.
				groundtruthArray = Utils.copyOf(groundtruthScore[doc],
					1, groundtruthScore[doc].length - 1); // skip the overall;
			}
			if (predictArray.length > 1) {
				misorderRate += KendallTauRankCorrelation(predictArray, groundtruthArray);
				num += 1;
			}
		}

		return misorderRate / num;
	}

	private static double KendallTauRankCorrelation(double[] predictArray,
				double[] groundtruthArray) throws ToolException {
		if (Utils.isEmpty(predictArray) ||
				Utils.isEmpty(groundtruthArray) ||
				predictArray.length != groundtruthArray.length) {
			throw new ToolException(ErrorMessage.ERROR_NO_ARGS);
		}

		double[][] pairs  = new double[predictArray.length][2];
		int concordantNum = 0;
		int discordantNum = 0;
		int totalNum      = 0;
		for (int i = 0; i < pairs.length; i ++) {
			pairs[i][0] = predictArray[i];
			pairs[i][1] = groundtruthArray[i];
		}
		for (int i = 0; i < pairs.length; i ++) {
			for (int j = i + 1; j < pairs.length; j ++) {
				if (isConcordant(pairs[i], pairs[j])) {
					concordantNum ++;
				} else
				if (isDiscordant(pairs[i], pairs[j])) {
					discordantNum ++;
				}
				totalNum ++;
			}
		}
		// debug
		/*Utils.println("concordantNum = " + concordantNum);
		Utils.println("discordantNum = " + discordantNum);
		Utils.println("totalNum      = " + totalNum);*/
		return (concordantNum - discordantNum + 0.0) / (totalNum + 0.0);
	}

	// the predict score is what have been cut instead of the original one
	private static double Misaspect(double[][] predictScore,
			double[][] groundtruthScore, boolean checkMissAspect,
			double defaultMissAspectValue) throws ToolException {
		if (Utils.isEmpty(predictScore) || Utils.isEmpty(groundtruthScore) ||
				predictScore.length != groundtruthScore.length ||
				predictScore[0].length != groundtruthScore[0].length) {
			throw new ToolException (ErrorMessage.ERROR_NO_ARGS);
		}

		double misorderRate = 0.0;
		double num          = 0.0;
		for (int doc = 0; doc < predictScore.length; doc ++) {
			double[] predict          = predictScore[doc];
			double[] groundtruth      = groundtruthScore[doc];
			double[] predictArray     = null;
			double[] groundtruthArray = null;
			if (checkMissAspect) {
				int count = 0;
				for (int i = 1; i < groundtruth.length; i ++) {
					if (groundtruth[i] != defaultMissAspectValue) {
						count ++;
					}
				}
				predictArray     = new double[count];
				groundtruthArray = new double[count];
				int index        = 0;
				for (int i = 1; i < groundtruth.length; i ++) {
					if (groundtruth[i] != defaultMissAspectValue) {
						predictArray[index]     = predict[i];
						groundtruthArray[index] = groundtruth[i];
						index ++;
					}
				}
			} else {
				predictArray     = Utils.copyOf(
					predictScore[doc], 1,
					predictScore[doc].length - 1); // skip the overall.
				groundtruthArray = Utils.copyOf(
					groundtruthScore[doc], 1,
					groundtruthScore[doc].length - 1); // skip the overall;
			}
			if (predictArray.length > 1) {
				misorderRate += Misaspect(predictArray, groundtruthArray);
				num += 1;
			}
		}	

		return misorderRate / num;
	}

	private static double Misaspect(double[] predictArray,
			double[] groundtruthArray) throws ToolException {
		if (Utils.isEmpty(predictArray) ||
				Utils.isEmpty(groundtruthArray) ||
				predictArray.length != groundtruthArray.length) {
			throw new ToolException(ErrorMessage.ERROR_NO_ARGS);
		}

		double[][] pairs  = new double[predictArray.length][2];
		int misorderedNum = 0;
		int totalNum      = 0;
		for (int i = 0; i < pairs.length; i ++) {
			pairs[i][0] = predictArray[i];
			pairs[i][1] = groundtruthArray[i];
		}
		for (int i = 0; i < pairs.length; i ++) {
			for (int j = i + 1; j < pairs.length; j ++) {
				if (isDiscordant(pairs[i], pairs[j])) {
					misorderedNum ++;
				}
				totalNum ++;
			}
		}
		// debug
		/*Utils.println("misorderedNum = " + misorderedNum + "; totalNum = " + totalNum +
				"; " + Utils.convertToString(predictArray, ",") +
				"; " + Utils.convertToString(groundtruthArray, ","));*/
		return (misorderedNum + 0.0) / (totalNum + 0.0);
	}

	private static boolean isConcordant(double[] pair1, double[] pair2) {
		return (pair1[0] > pair2[0] && pair1[1] > pair2[1]) ||
				(pair1[0] < pair2[0] && pair1[1] < pair2[1]);
	}

	private static boolean isDiscordant(double[] pair1, double[] pair2) {
		return (pair1[0] > pair2[0] && pair1[1] < pair2[1]) ||
				(pair1[0] < pair2[0] && pair1[1] > pair2[1]);
	}

	// the predict score is what have been cut instead of the original one
	private static double pearsonCorrelationAspect(double[][] predictScore,
			double[][] groundtruthScore, boolean checkMissAspect,
			double defaultMissAspectValue) throws ToolException {
		if (Utils.isEmpty(predictScore) || Utils.isEmpty(groundtruthScore) ||
				predictScore.length != groundtruthScore.length ||
				predictScore[0].length != groundtruthScore[0].length) {
			throw new ToolException (ErrorMessage.ERROR_NO_ARGS);
		}

		double pearsonCorr = 0.0;
		for (int doc = 0; doc < predictScore.length; doc ++) {
			double[] predict          = predictScore[doc];
			double[] groundtruth      = groundtruthScore[doc];
			double[] predictArray     = null;
			double[] groundtruthArray = null;
			if (checkMissAspect) {
				int count = 0;
				for (int i = 1; i < groundtruth.length; i ++) {
					if (groundtruth[i] != defaultMissAspectValue) {
						count ++;
					}
				}
				predictArray     = new double[count];
				groundtruthArray = new double[count];
				int index        = 0;
				for (int i = 1; i < groundtruth.length; i ++) {
					if (groundtruth[i] != defaultMissAspectValue) {
						predictArray[index]     = predict[i];
						groundtruthArray[index] = groundtruth[i];
						index ++;
					}
				}
			} else {
				predictArray = Utils.copyOf(predictScore[doc],
					1, predictScore[doc].length - 1); // skip the overall.
				groundtruthArray = Utils.copyOf(groundtruthScore[doc],
					1, groundtruthScore[doc].length - 1); // skip the overall;
			}
			pearsonCorr += pearsonCorrelation(predictArray, groundtruthArray);
		}
	
		return pearsonCorr / (predictScore.length + 0.0);
	}
	
	private static double pearsonCorrelationReview(double[][] predictScore,
			double[][] groundtruthScore, boolean checkMissAspect,
			double defaultMissAspectValue) throws ToolException {
		if (Utils.isEmpty(predictScore) || Utils.isEmpty(groundtruthScore) ||
				predictScore.length != groundtruthScore.length ||
				predictScore[0].length != groundtruthScore[0].length) {
			throw new ToolException (ErrorMessage.ERROR_NO_ARGS);
		}
	
		double pearsonCorr = 0.0;
		// skip the overall rating
		for (int topic = 1; topic < predictScore[0].length; topic ++) {
			double[] predictArray     = null;
			double[] groundtruthArray = null;
			if (checkMissAspect) {
				int count = 0;
				for (int doc = 0; doc < groundtruthScore.length; doc ++) {
					if (groundtruthScore[doc][topic] !=
							defaultMissAspectValue) {
						count ++;
					}
				}
				predictArray     = new double[count];
				groundtruthArray = new double[count];
				count            = 0;
				for (int doc = 0; doc < groundtruthScore.length; doc ++) {
					if (groundtruthScore[doc][topic] !=
							defaultMissAspectValue) {
						predictArray[count]     = predictScore[doc][topic];
						groundtruthArray[count] = groundtruthScore[doc][topic];
						count ++;
					}
				}
			} else {
				predictArray     = new double[predictScore.length];
				groundtruthArray = new double[predictScore.length];
				for (int doc = 0; doc < predictScore.length; doc ++) {
					predictArray[doc]     = predictScore[doc][topic];
					groundtruthArray[doc] = groundtruthScore[doc][topic]; 
				}
			}
			pearsonCorr += pearsonCorrelation(predictArray, groundtruthArray);
		}

		return pearsonCorr / (predictScore[0].length - 1.0);
	}

	public static double pearsonCorrelation(double[] x, double[] y)
						throws ToolException{
		if (Utils.isEmpty(x) || Utils.isEmpty(y) ||
					x.length != y.length) {
			throw new ToolException (ErrorMessage.ERROR_NO_ARGS);
		}
	
		double xMean = 0.0;
		double yMean = 0.0;
		double s_x   = 0.0;
		double s_y   = 0.0;
		for (int i = 0; i < x.length; i ++) {
			xMean += x[i];
			yMean += y[i];
		}
		xMean /= x.length + 0.0;
		yMean /= y.length + 0.0;

		for (int  i = 0; i < x.length; i ++) {
			s_x += (x[i] - xMean) * (x[i] - xMean);
			s_y += (y[i] - yMean) * (y[i] - yMean);
		}
	
		// handle special cases
		if (s_x == 0.0 && s_y == 0.0) {
			return 1.0;
		}
		if (s_x == 0.0) s_x = Math.exp(-100.0);
		if (s_y == 0.0) s_y = Math.exp(-100.0);

		double sum = 0.0;
		for (int i = 0; i < x.length; i ++) {
			sum += (x[i] - xMean) * (y[i] - yMean) / Math.sqrt(
					( s_x / (x.length - 1.0) ) * ( s_y / (x.length - 1.0) ) );
		}
		return sum / (x.length - 1);
	}

	// the predict score is what have been cut instead of the original one
	private static double nDCG(double[][] predictScore,
			double[][] groundtruthScore, boolean checkMissAspect,
			double defaultMissAspectValue) throws ToolException {
		if (Utils.isEmpty(predictScore) || Utils.isEmpty(groundtruthScore) ||
				predictScore.length != groundtruthScore.length ||
				predictScore[0].length != groundtruthScore[0].length) {
			throw new ToolException (ErrorMessage.ERROR_NO_ARGS);
		}

		double sum = 0.0;
		double num = 0.0;
		for (int doc = 0; doc < predictScore.length; doc ++) {
			double[] predict          = predictScore[doc];
			double[] groundtruth      = groundtruthScore[doc];
			double[] predictArray     = null;
			double[] groundtruthArray = null;
			if (checkMissAspect) {
				int count = 0;
				for (int i = 1; i < groundtruth.length; i ++) {
					if (groundtruth[i] != defaultMissAspectValue) {
						count ++;
					}
				}
				predictArray     = new double[count];
				groundtruthArray = new double[count];
				int index        = 0;
				for (int i = 1; i < groundtruth.length; i ++) {
					if (groundtruth[i] != defaultMissAspectValue) {
						predictArray[index]     = predict[i];
						groundtruthArray[index] = groundtruth[i];
						index ++;
					}
				}
			} else {
				predictArray     = Utils.copyOf(predictScore[doc],
					1, predictScore[doc].length - 1); // skip the overall.
				groundtruthArray = Utils.copyOf(groundtruthScore[doc],
					1, groundtruthScore[doc].length - 1); // skip the overall;
			}
			if (predictArray.length > 1) {
				sum += nDCG(predictArray, groundtruthArray);
				num += 1;
			}
		}

		return sum / num;
	}

	private static double nDCG(double[] predictArray,
				double[] groundtruthArray) throws ToolException {
		if (Utils.isEmpty(predictArray) ||
				Utils.isEmpty(groundtruthArray) ||
				predictArray.length != groundtruthArray.length) {
			throw new ToolException(ErrorMessage.ERROR_NO_ARGS);
		}

		ArrayList<Object[]> predictList = new ArrayList<Object[]>();
		for (int i = 0; i < groundtruthArray.length; i ++) {
			predictList.add(new Object[]{predictArray[i], i});
		}
		Collections.sort(predictList, new Comparator<Object[]>() {
			@Override
			public int compare(Object[] obj1, Object[] obj2) {
				Double d1 = (Double)obj1[0];
				Double d2 = (Double)obj2[0];
				return d2.compareTo(d1);
			}
		});

		Double[] idealGroundtruthArray = new Double[predictArray.length];
		for (int i = 0; i < idealGroundtruthArray.length; i ++) {
			idealGroundtruthArray[i] = groundtruthArray[i];
		}
		Arrays.sort(idealGroundtruthArray, new Comparator<Double>() {
			@Override
			public int compare(Double d1, Double d2) {
				return d2.compareTo(d1);
			}
		});

		double idcg = 0.0;
		double dcg  = 0.0;
		for (int i = 0; i < predictArray.length; i ++) {
			if (i == 0) {
				dcg  += groundtruthArray[(Integer)predictList.get(i)[1]];
				idcg += idealGroundtruthArray[i];
			} else {
				dcg  += groundtruthArray[(Integer)predictList.get(i)[1]] * Math.log(2.0) / Math.log(i + 1.0);
				idcg += idealGroundtruthArray[i] * Math.log(2.0) / Math.log(i + 1.0);
			}
			/*dcg  += (Math.pow(2.0, groundtruthArray[(Integer)predictList.get(i)[1]]) - 1.0 ) /
					(Math.log(i + 2) / Math.log(2.0));
			idcg += (Math.pow(2.0, idealGroundtruthArray[i]) - 1.0 ) /
					(Math.log(i + 2) / Math.log(2.0));*/
		}
		return dcg / idcg;
	}

	public static double calculateSematicCoherence(File trainFile,
				File wordProbFile, int topWordNum,
					boolean debug) throws TopicModelException,
						AIRException, ToolException {
		if (! Utils.exists(trainFile) || ! Utils.exists(wordProbFile)) {
			throw new AIRException(String.format(
					"File doesn't exisit.[TrainFile = %s][WordProbFile = %s]",
					trainFile == null ? null : trainFile.getAbsolutePath(),
					wordProbFile == null ? null : wordProbFile.getAbsolutePath()));
		}

		Dictionary dictionary    = new Dictionary();
		AIRDataset data          = new AIRDataset(dictionary, trainFile,
						null, false);
		topWordNum               = topWordNum > dictionary.getSize() ?
						dictionary.getSize() : topWordNum;
		double[][] wordProbArray = Utils.load2Array(wordProbFile, "\t");

		return calcaluateSematicCoherence(wordProbArray,
				data.getDocuments(), dictionary, topWordNum,
				debug);
		
	}

	public static double calcaluateSematicCoherence(double[][] wordProbArray,
					List<Document> documents,
					Dictionary dictionary, int topWordNum,
					boolean debug) throws AIRException {
		int[][] topWordArray = new int[wordProbArray.length][topWordNum];

		for (int topic = 0; topic < wordProbArray.length; topic ++) {
			double[] probs           = wordProbArray[topic];
			ArrayList<Object[]> list = new ArrayList<Object[]>();
			for (int i = 0; i < probs.length; i ++) {
				if (dictionary != null) {
					list.add(new Object[]{probs[i], i,
							dictionary.getWord(i)});
				} else {
					list.add(new Object[]{probs[i], i});
				}
			}
			Collections.sort(list, new Comparator<Object[]> () {
				@Override
				public int compare(Object[] obj1, Object[] obj2) {
					Double d1 = (Double)obj1[0];
					Double d2 = (Double)obj2[0];
					return d2.compareTo(d1);
				}
			});
			// debug
			if (dictionary != null && debug) {
				Utils.println("Topic - " + topic);
				for (int i = 0; i < topWordNum; i ++) {
					System.out.print(list.get(i)[2] + " ");
				}
				Utils.println("");
			}

			for (int i = 0; i < topWordNum; i ++) {
				topWordArray[topic][i] = (Integer)list.get(i)[1];
			}
		}

		double totalCoherence = 0.0;
		for (int topic = 0; topic < topWordArray.length; topic ++) {
			totalCoherence += calculateSemanticCoherence(
					topWordArray[topic], documents);
		}

		return (totalCoherence) / (topWordArray.length + 0.0);
	}

	public static double calculateSemanticCoherence(int[] topWords,
				List<Document> documents) throws AIRException {
		if (Utils.isEmpty(topWords) || Utils.isEmpty(documents)) {
			throw new AIRException("Empty Input in calculateSemanticCoherence.");
		}

		HashMap<String, Integer> tokenCountMap = new HashMap<String, Integer>();
		HashSet<Integer> topWordSet            = new HashSet<Integer>();
		for (int i = 0; i < topWords.length; i ++) {
			tokenCountMap.put(String.valueOf(topWords[i]), 0);
			topWordSet.add(topWords[i]);
			for (int j = i + 1; j < topWords.length; j ++) {
				tokenCountMap.put(getCoWordId(topWords[i], topWords[j]), 0);
			}
		}

		for (int doc = 0; doc < documents.size(); doc ++) {
			int uniqueWords[] =  documents.get(doc).uniqueWords;
			for (int i = 0; i < uniqueWords.length; i ++) {
				int wordId = uniqueWords[i];
				if (! topWordSet.contains(wordId)) continue;

				// single word
				String wordIdString = String.valueOf(wordId);
				tokenCountMap.put(wordIdString, tokenCountMap.get(wordIdString) + 1);
				//Utils.println(wordIdString + ", " + tokenCountMap.get(wordIdString));
				// co-words
				for (int j = i + 1; j < uniqueWords.length; j ++) {
					int anotherWordId = uniqueWords[j];
					if (! topWordSet.contains(anotherWordId)) continue;

					wordIdString = getCoWordId(wordId, anotherWordId);
					tokenCountMap.put(wordIdString,
							tokenCountMap.get(wordIdString) + 1);
				}
			}
		}

		double coherence = 0.0;
		for (int m = 1; m < topWords.length; m ++) {
			for (int l = 0; l < m; l ++) {
				int numl  = tokenCountMap.get(String.valueOf(topWords[l]));
				int numml = tokenCountMap.get(getCoWordId(
								topWords[m],
								topWords[l]));
				coherence += Math.log((numml + 1.0) / (numl + 0.0));
			}
		}
		return coherence;
	}
 
	private static String getCoWordId(int word1,  int word2)
						throws AIRException {
		int small = word1; // smaller one
		int large = word2; // larger one
		if (word1 > word2) {
			small = word2;
			large = word1;
		} else
		if (small == large) {
			throw new AIRException(String.format(
					"Two same tops exisit in top Words.[%s][%s]",
					small, large));
		}
		return small + "_" + large;
	}

	public static double mse(double[][] predictScore,
			double[][] groundtruthScore) throws ToolException {
		return mse(predictScore, groundtruthScore, false,
				AIRConstants.DEFAULT_MISS_ASPECT_RATING);
	}

	// the predict score is what have been cut instead of the original one
	public static double mse(double[][] predictScore,
			double[][] groundtruthScore, boolean checkMissAspect,
			double defaultMissAspectValue) throws ToolException {
		if (Utils.isEmpty(predictScore) || Utils.isEmpty(groundtruthScore) ||
				predictScore.length != groundtruthScore.length ||
				predictScore[0].length != groundtruthScore[0].length) {
			throw new ToolException (ErrorMessage.ERROR_NO_ARGS);
		}
		double mse = 0.0;
		int num    = 0;
		for (int doc = 0; doc < predictScore.length; doc ++) {
			double[] predict          = predictScore[doc];
			double[] groundtruth      = groundtruthScore[doc];
			double[] predictArray     = null;
			double[] groundtruthArray = null;
			if (checkMissAspect) {
				int count = 0;
				for (int i = 1; i < groundtruth.length; i ++) {
					if (groundtruth[i] != defaultMissAspectValue) {
						count ++;
					}
				}
				predictArray     = new double[count];
				groundtruthArray = new double[count];
				int index        = 0;
				for (int i = 1; i < groundtruth.length; i ++) {
					if (groundtruth[i] != defaultMissAspectValue) {
						predictArray[index]     = predict[i];
						groundtruthArray[index] = groundtruth[i];
						index ++;
					}
				}
			} else {
				predictArray     = Utils.copyOf(predict, 1,
					predict.length - 1); // skip the overall.
				groundtruthArray = Utils.copyOf(groundtruth,
					1, groundtruth.length - 1); // skip the overall;
			}
			if (predictArray.length > 0) {
				mse += mse(predictArray, groundtruthArray);
				num += predictArray.length;
			}
		}
		return mse / (num + 0.0);
	}

	private static double mse(double[] predictArray,
			double[] groundtruthArray) throws ToolException {
		if (Utils.isEmpty(predictArray) ||
				Utils.isEmpty(groundtruthArray) ||
				predictArray.length != groundtruthArray.length) {
			throw new ToolException(ErrorMessage.ERROR_NO_ARGS);
		}

		double sum = 0.0;
		for (int i = 0; i < predictArray.length; i ++) {
			sum += (predictArray[i] - groundtruthArray[i]) *
					(predictArray[i] - groundtruthArray[i]);
		}
		return sum;
	}


	public static String mse(File predictScoreFile, File trueScoreFile,
			int topicNum, RatingScaler ratingScaler,
			boolean checkMissAspect, double defaultMissAspectValue)
					throws ToolException {
		if (! predictScoreFile.exists() || ! trueScoreFile.exists()) {
			return null;
		}
		double[][] tratings = Utils.load2Array(trueScoreFile, " ");
		double[][] pratings = Utils.load2Array(predictScoreFile, " ");
		if (tratings[0].length != topicNum + 1) {
			throw new RuntimeException(String.format(
				"ParsedGroundtruthRatingNum=%s, SpecfiedTopicNum=%s",
				tratings[0].length, topicNum));
		}
		if (pratings[0].length != topicNum + 1) {
			throw new RuntimeException(String.format(
				"ParsedPredictRatingNum=%s, SpecifiedTopicNum=%s",
				pratings[0].length, topicNum));
		}
		if (tratings.length != pratings.length) {
			throw new RuntimeException(String.format(
				"length does not match: groundtruthRatingLen=%s, predictRatinglen=%s",
				tratings.length, pratings.length));
		}
		int totalNum = 0;
		int[] counts = new int[topicNum + 1];
		double[] sum = new double[topicNum + 1];
		for (int doc = 0; doc < tratings.length; doc ++) {
			double[] groundtruth = tratings[doc];
			double[] predict     = pratings[doc];
			if (groundtruth[0] != predict[0]) {
				// if the overall is not equal
				throw new RuntimeException(String.format(
					"Doc=%s, groundtruthOverall=%s, predictOverall=%s",
					doc, groundtruth[0], predict[0]));
			}
			for (int i = 1; i < groundtruth.length; i ++) {
				if (! checkMissAspect ||
						groundtruth[i] != defaultMissAspectValue) {
					double predictValue = cut(predict[i], i - 1, ratingScaler); 
					double trueValue    = groundtruth[i];
					sum[0]         += (predictValue - trueValue) *
							(predictValue - trueValue);
					sum[i] += (predictValue - trueValue) *
							(predictValue - trueValue);
					counts[i] ++;
					totalNum ++;
				}
			}
		}
		Utils.println("TotleNum = " + totalNum +
				", revNum=" + tratings.length +
				", TrueTotalNum=" + (totalNum));

		StringBuffer buffer = new StringBuffer();
		for (int i = 0; i < sum.length; i ++) {
			if (i == 0) {
				buffer.append("Overall MSE = " + (sum[i] / (totalNum + 0.0)))
					.append(com.lhy.tool.Constants.LINE_SEPARATOR);
			} else {
				buffer.append(String.format("Topic - %s, MSE = %s",
					(i - 1), sum[i] / (counts[i] + 0.0)));
				if (i != sum.length - 1) {
					buffer.append(com.lhy.tool.Constants.LINE_SEPARATOR);
				}
			}
		}
		Utils.println(buffer);
		return buffer.toString();
	}

	private static double cut(double score, int topicIndex,
					RatingScaler ratingScaler) {
		if (ratingScaler != null) return ratingScaler.cut(score, topicIndex);
	
		/*double v = score;
		if (v > 5.0) return 5.0;
		if (v < 1.0) return 1.0;
		return v;*/
		return score;
	}

	public static void selfmse(File trueScoreFile, File outputPath,
				int topicNum, boolean checkMissAspect,
				double defaultMissAspectValue) throws ToolException {
		try {
			int[][] hisCount   =null;
			{
				int[][] tratings = Utils.load2IntArray(trueScoreFile, " ");
				int min = Integer.MAX_VALUE;
				int max = -1 * Integer.MAX_VALUE;
				for (int i = 0; i < tratings.length; i ++) {
					for (int j = 0; j < tratings[i].length; j ++) {
						int v = tratings[i][j];
						if (v < min) min = v;
						if (v > max) max = v;
					}
				}
				hisCount = new int[topicNum][max - min + 1];
			}
			double[][] tratings = Utils.load2Array(trueScoreFile, " ");
			if (tratings[0].length != topicNum + 1) {
				throw new RuntimeException(String.format(
					"ParsedGroundtruthRatingNum=%s, SpecfiedTopicNum=%s",
					tratings[0].length, topicNum));
			}
			int totalNum      = 0;
			double overallMse = 0.0;
			for (int doc = 0; doc < tratings.length; doc ++) {
				double[] groundtruth = tratings[doc];
				double overall       = groundtruth[0];
				for (int i = 1; i < groundtruth.length; i ++) {
					if (! checkMissAspect || groundtruth[i] !=
							defaultMissAspectValue){
						double diff = Math.abs(groundtruth[i] - overall);
						overallMse += diff * diff;

						int diffIndex = ((int)diff);
						if ((diffIndex - diff) != 0.0) {
							Utils.err("Difference is not an Integer.");
						}
						hisCount[i - 1][diffIndex]++;
						totalNum ++;
					}
				}
			}

			double[] means = new double[topicNum]; 
			for (int topic = 0; topic < hisCount.length; topic ++) {
				HashMap<String, Integer> countMap = new HashMap<String, Integer>();
				double sum     = 0.0;
				int totalCount = 0;
				for (int i = 0; i < hisCount[topic].length; i ++) {
					countMap.put(String.valueOf(i), hisCount[topic][i]);
					totalCount += hisCount[topic][i];
					sum        += hisCount[topic][i] * i;
				}
				means[topic] = sum / totalCount;
			}

			Utils.println("TotleNum = " + totalNum +
					", revNum=" + tratings.length +
					", TrueTotalNum=" + (tratings.length * topicNum));
			Utils.println("Self Overall MSE = " + (overallMse / (totalNum + 0.0)));
			Utils.println("Mean : " + Utils.convertToString(means, " , "));
			
		} catch (NumberFormatException e) {
			e.printStackTrace();
		}
	}
}
