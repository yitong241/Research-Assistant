package com.uncc.air;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.Properties;
import java.util.StringTokenizer;

import com.csvreader.CsvWriter;
import com.lhy.tool.util.Utils;
import com.uncc.air.data.RatingScaler;
import com.uncc.topicmodel.data.Dataset;
import com.uncc.topicmodel.data.Dictionary;

/**
 * @author Huayu Li
 */
public abstract class Model {
	private static final boolean IS_BIGRAM = false;

	protected static final String FORMAT_MODEL_NAME =
			"Lambda=%.2f,Gamma=%.2f,%.2f,Topic=%s";

	public static final int SENTIMENT_NUM = 3;

	public int    TOPIC_NUM;
	protected int TOP_WORD_NUM;

	public double[][] betas      = null;
	public double[] betaSum      = null;
	public Dictionary dictionary = null;
	public Dataset    data       = null;
	
	protected ArrayList<Integer>[] keywords = null;

	public abstract Properties getParameters();
	public abstract String getPrefix();

	protected String format(String s) {
		return String.format(getPrefix(), s);
	}

	public File getScoreFile(File outputPath) {
		return new File(outputPath, format(AIRConstants.FILE_NAME_SCORE));
	}

	protected boolean storeScore(double[][][] omega, File scoreOutputFile,
				RatingScaler ratingScaler, boolean print) {
		if (omega != null && scoreOutputFile != null) {
			BufferedWriter writer = null;
			try {
				if (print) System.out.print("Parameter score is saving... ");
				writer = Utils.createBufferedWriter(scoreOutputFile);
				for (int i = 0; i < omega.length; i ++) {
					StringBuffer buffer = new StringBuffer();
					buffer.append(data.getDocuments().get(i).rating)
					.append(AIRConstants.SPACER_TAB);
					for (int k = 0; k < omega[i].length; k ++) {
						buffer.append(String.format("%.3f", ratingScaler.recoverRating(omega[i][k][0], k)));
						if (k != omega[i].length - 1) {
							buffer.append(AIRConstants.SPACER_TAB);
						}
					}
					Utils.write(writer, buffer.toString(), true);
				}
				return true;
			} catch (IOException e) {
				e.printStackTrace();
			} finally {
				Utils.cleanup(writer);
				if (print) System.out.println("end!");
			}
		}

		return false;
	}

	// Saves the words according their probabilities
	@SuppressWarnings("unchecked")
	protected boolean storeTopWords(File topwordCsvFile, boolean isWithCount,
				double[][][] phi, int[][][] topicSentimentWord) {
		if (topwordCsvFile == null) return false;

		CsvWriter csvWriter            = null;
		ArrayList<Object[]>[] topWords = new ArrayList[SENTIMENT_NUM + 1];
		double[][] sumWordProb         = sumWordProb(phi);
		int[][] sumTopicWordCount      = sumTopicWordCount(topicSentimentWord);
			
		try {
			csvWriter = Utils.createCsvWriter(topwordCsvFile);
			for (int topic = 0; topic < TOPIC_NUM; topic ++) {
				csvWriter.write("Topic-" + topic);
				csvWriter.endRecord();
				for (int sentiment = 0; sentiment < topWords.length; sentiment ++) {
					if (sentiment < SENTIMENT_NUM) {
						topWords[sentiment] = isWithCount ?
								getTopWordsWithCount(topicSentimentWord[topic][sentiment]) :
								getTopWordsWithPropability(phi[sentiment][topic],
										topicSentimentWord == null ? null :
											topicSentimentWord[topic][sentiment]);
					} else {
						// print top word via sum all sentiment
						topWords[sentiment] = isWithCount ?
								getTopWordsWithCount(sumTopicWordCount[topic]) :
								getTopWordsWithPropability(sumWordProb[topic],
										sumTopicWordCount == null ?
											null : sumTopicWordCount[topic]);
					}
				}

				for (int i = 0; i < TOP_WORD_NUM; i ++) {
					for (int sentiment = 0; sentiment < topWords.length; sentiment ++) {
						for (int j = 0; j < topWords[sentiment].get(i).length; j ++) {
							csvWriter.write(topWords[sentiment].get(i)[j].toString());
						}
						if (sentiment != topWords.length - 1) {
							csvWriter.write("");
						}
					}
					csvWriter.endRecord();
				}
			}

			return true;
		} catch (IOException e) {
			e.printStackTrace();
		} finally {
			Utils.cleanup(csvWriter);
		}

		return false;
	}

	private ArrayList<Object[]> getTopWordsWithPropability(double[] phi,
					int[] topicSentimentWord) {
		if (phi == null) return null;

		ArrayList<Object[]> list = new ArrayList<Object[]>();
		for (int i = 0; i < phi.length; i ++) {
			Object[] object = null;
			if (topicSentimentWord != null) {
				object = new Object[]{dictionary.getWord(i), phi[i],
					topicSentimentWord[i]};
			} else {
				object = new Object[]{dictionary.getWord(i), phi[i]};
			}
			list.add(object);
		}
		Collections.sort(list, new Comparator<Object[]>() {
			@Override
			public int compare(Object[] obj1, Object[] obj2) {
				return ((Double)obj1[1]).compareTo(
						((Double)obj2[1])) * (-1);
			}
		});

		ArrayList<Object[]> result = new ArrayList<Object[]>();
		int maxNum = (TOP_WORD_NUM > dictionary.getSize() ? dictionary.getSize() : TOP_WORD_NUM);
		for (int i = 0; i < maxNum; i ++) {
			result.add(list.get(i));
		}

		return result;
	}

	private ArrayList<Object[]> getTopWordsWithCount(int[] topicSentimentWord) {
		if (topicSentimentWord == null) return null;

		ArrayList<Object[]> list = new ArrayList<Object[]>();
		for (int i = 0; i < topicSentimentWord.length; i ++) {
			Object[] object = {dictionary.getWord(i),
						topicSentimentWord[i]};
			list.add(object);
		}
		Collections.sort(list, new Comparator<Object[]>() {
			@Override
			public int compare(Object[] obj1, Object[] obj2) {
				return ((Integer)obj2[1]).compareTo((Integer)obj1[1]);
			}
		});

		ArrayList<Object[]> result = new ArrayList<Object[]>();
		int maxNum = (TOP_WORD_NUM > dictionary.getSize() ? dictionary.getSize() : TOP_WORD_NUM);
		for (int i = 0; i < maxNum; i ++) {
			result.add(list.get(i));
		}

		return result;
	}

	protected void initBeta(double betaInit, ArrayList<Integer>[] keywordList)
						throws AIRException {
		betas = new double[TOPIC_NUM][dictionary.getSize()];
		if (keywordList != null) {
			if (keywordList.length != TOPIC_NUM) {
				throw new AIRException(String.format(
						ErrorMessage.ERROR_NOT_MATCH_KEYWORD,
						keywordList.length,
						TOPIC_NUM));
			} else {
				betas = initArrayWithKeyword(betaInit, null, keywordList);
			}
		} else {
			for (int topic = 0; topic < TOPIC_NUM; topic ++) {
				for (int word = 0; word < dictionary.getSize(); word ++) {
					betas[topic][word] = betaInit;
				}
			}
		}

		betaSum = getBetaSum();
	}

	protected double[][] initArrayWithKeyword(double initValue,
					double[][] initValueArray,
					ArrayList<Integer>[] keywordList) {
		double[][] array = new double[TOPIC_NUM][dictionary.getSize()];
		for (int topic = 0; topic < TOPIC_NUM; topic ++) {
			for (int word = 0; word < dictionary.getSize(); word ++) {
				if (initValueArray != null) {
					array[topic][word] = initValueArray[topic][word];
				} else {
					array[topic][word] = initValue;
				}
			}
		}
		for (int topic = 0; topic < TOPIC_NUM; topic ++) {
			for (Integer keywordId : keywordList[topic]) {
				array[topic][keywordId] = 1.0;
				for (int k = 0; k < TOPIC_NUM; k ++) {
					// avoid the same keyword in different topic
					if (k == topic || array[k][keywordId] == 1.0) continue;

					array[k][keywordId] = Math.exp(-100);
				}
			}
		}
		return array;
	}

	

	@SuppressWarnings("unchecked")
	public static ArrayList<Integer>[] loadKeywords(File keywordFile,
				Dictionary dictionary) throws AIRException {
		if (! Utils.exists(keywordFile)) return null;

		Utils.println("Keyword File : " + keywordFile.getAbsolutePath());
		BufferedReader reader = null;
		try {
			ArrayList<ArrayList<Integer>> list = new ArrayList<ArrayList<Integer>>();
			reader      = Utils.createBufferedReader(keywordFile);
			String line = null;
			while ((line = reader.readLine()) != null) {
				if (Utils.isEmpty((line = line.trim()))) continue;

				StringTokenizer tokenizer  = new StringTokenizer(line);
				ArrayList<Integer> keywords = new ArrayList<Integer>();
				while (tokenizer.hasMoreElements()) {
					String keyword  = tokenizer.nextToken();
					if (IS_BIGRAM) {
						boolean isFound = false;
						for (String dicWord : dictionary.getWords()) {
							if (containWord(keyword, dicWord, "_")) {
								keywords.add(dictionary.getWordId(dicWord));
								isFound = true;
							}
						}
						if (! isFound) {
							Utils.err("Cannot find keywod in dictionary: keyword = " + keyword);
							//throw new RTMException("Cannot find keywod in dictionary: keyword = " + keyword);
						}
					} else {
						if (! dictionary.contains(keyword)) {
							Utils.err("Cannot find keywod in dictionary: keyword = " + keyword);
							//throw new RTMException("Cannot find keywod in dictionary: keyword = " + keyword);
						} else {
							keywords.add(dictionary.getWordId(keyword));
						}
					}
				}
				if (! keywords.isEmpty()) {
					list.add(keywords);
				}
			}

			// print out
			/*for (ArrayList<Integer> subList : list) {
				StringBuffer buffer = new StringBuffer();
				for (Integer elem : subList) {
					buffer.append(elem).append(" ");
				}
				Utils.println(subList.size());
				//Utils.println(buffer);
			}*/

			return list.isEmpty() ? null : list.toArray(new ArrayList[0]);
		} catch (IOException e) {
			e.printStackTrace();
		} finally {
			Utils.cleanup(reader);
		}

		return null;
	}

	public static double[][] sumWordProb(double[][][] phi) {
		if (phi == null) return null;

		double[][] wordProb  = new double[phi[0].length][phi[0][0].length];
		double[] wordProbSum = new double[phi[0].length];

		for (int sentiment = 0; sentiment < phi.length; sentiment ++) {
			for (int topic = 0; topic < phi[sentiment].length; topic ++) {
				for (int word = 0; word < phi[sentiment][topic].length; word ++) {
					wordProb[topic][word] += phi[sentiment][topic][word];
					wordProbSum[topic]    += phi[sentiment][topic][word];
				}
			}
		}
		for (int topic = 0; topic < wordProb.length; topic ++) {
			for (int word = 0; word < wordProb[topic].length; word ++) {
				wordProb[topic][word] /= wordProbSum[topic];
			}
		}

		return wordProb;
	}

	private int[][] sumTopicWordCount(int[][][] topicSentimentWord) {
		if (topicSentimentWord == null) return null;

		int[][] topicWordCount  = new int[topicSentimentWord.length][topicSentimentWord[0][0].length];
		for (int topic = 0; topic < topicSentimentWord.length; topic ++) {
			for (int sentiment = 0; sentiment < topicSentimentWord[topic].length; sentiment ++) {
				for (int word = 0; word < topicSentimentWord[topic][sentiment].length; word ++) {
					topicWordCount[topic][word] += topicSentimentWord[topic][sentiment][word];
				}
			}
		}
		return topicWordCount;
	}

	private double[] getBetaSum() {
		double[] betaSum = new double[TOPIC_NUM];
		for (int topic = 0; topic < TOPIC_NUM; topic ++) {
			double sum = 0.0;
			for (int word = 0; word < dictionary.getSize(); word ++) {
				sum += betas[topic][word];
			}
			betaSum[topic] = sum;
		}
		return betaSum;
	}

	private static boolean containWord(String child, String parent,
							String spacer) {
		String[] words = parent.split(spacer);
		for (String word : words) {
			if (word.equals(child)) return true;
		}
		return false;
	}
}
