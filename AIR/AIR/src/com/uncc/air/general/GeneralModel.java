package com.uncc.air.general;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.Properties;

import com.lhy.tool.util.Utils;
import com.uncc.topicmodel.TopicModelException;
import com.uncc.topicmodel.data.Dataset;
import com.uncc.topicmodel.data.Dictionary;
import com.uncc.air.AIRConstants;
import com.uncc.air.AIRException;
import com.uncc.air.ErrorMessage;
import com.uncc.air.Model;
import com.uncc.air.ParamManager;
import com.uncc.air.data.AIRDataset;
import com.uncc.air.general.eval.EvaluationModel;
import com.uncc.air.util.AIRUtils;

/**
 * @author Huayu Li
 */
public class GeneralModel extends Model implements Dictionary.DictionaryListener,
						EvaluationModel {
	public  static final double ALPHA_EST_ERROR_TOLERANCE  = 1.0e-4;
	public  static final double LIKELIHOOD_ERROR_TOLERANCE = 1.0e-4;
	public  static final double LAMBDA_EST_ERROR_TOLERANCE = 1.0e-4;

	protected int burninIter    = 0;
	protected int alphaEstIter  = 0;
	protected int lambdaEstIter = 0;

	protected int burninIterNum ;
	protected int estimateIterNum;
	protected int maxAlphaEstIterNum;
	protected int maxLambdaEstIterNum;

	protected double betaInit;
	protected double lambdaInit;
	protected double[] lambdas = null;
	protected double[] alphas  = null;
	protected double[] gammas  = null;//{9000.0, 1000.0};//{8.19, 3.95};//{2.5, 1.0};

	protected int[][][] topicSentimentWord      = null;
	protected int[][] topicSentimentWordSum     = null;
	protected int [][] documentTopic            = null;
	protected int [] documentTopicSum           = null;
	protected int[][] documentSentiment         = null; // only 2 sentiments, 0 means s = 0, 1 means s = 1,2
	protected int[] documentSentimentSum        = null;
	protected int[][][] documentTopicSentiment  = null; // only 2 sentiments, 0 means s = 1, 1 means s = 2
	protected int[][] documentTopicSentimentSum = null;

	protected double[][][] phi   = null;
	protected double[][] theta   = null;
	protected double[][][] omega = null;
	protected double[][]   t     = null;

	private File modelOutputPath = null;

	public GeneralModel(File datasetFile, ParamManager paramManager)
					throws AIRException, TopicModelException {
		init(datasetFile, paramManager);
	}

	private void init(File datasetFile, ParamManager paramManager)
					throws AIRException, TopicModelException {
		this.modelOutputPath = paramManager.getModelOutputPath();

		if (paramManager.isRestore()) {
			loadModel(datasetFile, paramManager.getModelOutputPath());
		} else {
			initNewModel(datasetFile, paramManager);
		}
		dictionary.addDictionaryListener(this);
	}

	@Override
	public void dictionaryChanged() {
		int topicSentimentWordNew[][][] = new int[TOPIC_NUM][SENTIMENT_NUM][dictionary.getSize()];
		for (int topic = 0; topic < TOPIC_NUM; topic ++) {
			for (int sentiment = 0; sentiment < SENTIMENT_NUM; sentiment ++) {
				for (int word = 0; word < topicSentimentWord[topic][sentiment].length; word ++) {
					topicSentimentWordNew[topic][sentiment][word] = topicSentimentWord[topic][sentiment][word];
				}
			}
		}
		topicSentimentWord = topicSentimentWordNew;

		try {
			initBeta(betaInit, keywords);
		} catch (AIRException e) {
			e.printStackTrace();
		}
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
	public double[] getAlphas() {
		return alphas;
	}

	@Override
	public double[] getGammas() {
		return gammas;
	}

	@Override
	public double[] getLambdaHat(double rating, int topic) {
		double[] lambdaHats = new double[2];
		lambdaHats[0] = lambdas[topic] * rating;
		lambdaHats[1] = lambdas[topic] * (1 - rating);
		return lambdaHats;
	}

	@Override
	public double[][][] getWordProb() {
		return phi;
	}

	@Override
	public Dictionary getDictionary() {
		return dictionary;
	}

	@Override
	public Dataset getDataset(File datasetFile, Dictionary dictionary,
					File dicFile, boolean isRestore)
							throws TopicModelException {
		return new AIRDataset(dictionary, datasetFile, dicFile, isRestore);
	}

	@Override
	public Properties getParameters() {
		Properties props = new Properties();

		props.setProperty(AIRConstants.STRING_ALPHAS,               String.valueOf(Utils.convertToString(alphas, AIRConstants.SPACER_COMMA)));
		props.setProperty(AIRConstants.STRING_BETA_INIT,            String.valueOf(betaInit));
		props.setProperty(AIRConstants.STRING_GAMMAS,               String.valueOf(Utils.convertToString(gammas, AIRConstants.SPACER_COMMA)));
		props.setProperty(AIRConstants.STRING_LAMBDA,               String.valueOf(lambdaInit));
		props.setProperty(AIRConstants.STRING_BURNIN_ITER_NUM,      String.valueOf(burninIterNum));
		props.setProperty(AIRConstants.STRING_BURNIN_ITER,          String.valueOf(burninIter));
		props.setProperty(AIRConstants.STRING_EST_ITER_NUM,         String.valueOf(estimateIterNum));
		props.setProperty(AIRConstants.STRING_MAX_ALPHA_EST_I_NUM,  String.valueOf(maxAlphaEstIterNum));
		props.setProperty(AIRConstants.STRING_MAX_LAMBDA_EST_I_NUM, String.valueOf(maxLambdaEstIterNum));
		props.setProperty(AIRConstants.STRING_ALPHA_EST_ITER,       String.valueOf(alphaEstIter));
		props.setProperty(AIRConstants.STRING_DIC_NUM,              String.valueOf(dictionary.getSize()));
		props.setProperty(AIRConstants.STRING_TOPIC_NUM,            String.valueOf(TOPIC_NUM));
		props.setProperty(AIRConstants.STRING_DOCUMENT_NUM,         String.valueOf(data.getDocumentSize()));
		props.setProperty(AIRConstants.STRING_TOP_WORD_NUM,         String.valueOf(TOP_WORD_NUM));
		props.setProperty(AIRConstants.STRING_TOKEN_NUM,            String.valueOf(data.getTermNum()));
		props.setProperty(AIRConstants.STRING_MODEL_OUTPUT_PATH,    modelOutputPath.getAbsolutePath());
		props.setProperty(AIRConstants.STRING_KEYWORD_FILE,         keywords == null ? "" : "YES");
	
		props.putAll(data.getParams());

		return props;
	}

	@Override
	public String getPrefix() {
		return "AIR_%s";
	}

	private void initNewModel(File datasetFile, ParamManager paramManager)
					throws AIRException, TopicModelException {
		parseParamArguments(datasetFile, paramManager);

		initBeta(betaInit, keywords);

		topicSentimentWord        = new int[TOPIC_NUM][SENTIMENT_NUM][dictionary.getSize()];
		topicSentimentWordSum     = new int[TOPIC_NUM][SENTIMENT_NUM];
		documentTopic             = new int[data.getDocumentSize()][TOPIC_NUM];
		documentTopicSum          = new int[data.getDocumentSize()];
		documentSentiment         = new int[data.getDocumentSize()][2];
		documentSentimentSum      = new int[data.getDocumentSize()];
		documentTopicSentiment    = new int[data.getDocumentSize()][TOPIC_NUM][2];
		documentTopicSentimentSum = new int[data.getDocumentSize()][TOPIC_NUM];
		for (int i = 0; i < data.getDocumentSize(); i ++) {
			for (int j = 0; j < data.getDocuments().get(i).wordNum; j ++) {
				int word = data.getDocuments().get(i).words[j];

				int topic = getRandomTopicIndex(word);
				data.getDocuments().get(i).topics[j] = topic;
	
				int sentimentIndex = getRandomSentimentIndex(data.getDocuments().get(i).Ri);
				data.getDocuments().get(i).sentiments[j] = sentimentIndex;

				topicSentimentWord[topic][sentimentIndex][word] ++;
				topicSentimentWordSum[topic][sentimentIndex] ++;
				documentTopic[i][topic] ++;
				documentTopicSum[i] ++;
				documentSentiment[i][sentimentIndex == 0 ? 0 : 1] ++;
				documentSentimentSum[i] ++;
				if (sentimentIndex > 0) {
					documentTopicSentiment[i][topic][sentimentIndex - 1] ++;
					documentTopicSentimentSum[i][topic] ++;
				}
			}
		}
	}

	public boolean storeModel(boolean print) {
		boolean res = true;

		File paramOutputFile = new File(modelOutputPath,
					format(AIRConstants.FILE_NAME_PARAM));
		File tfwOutputFile   = new File(modelOutputPath,
					format(AIRConstants.FILE_NAME_WORD_TOPIC));
		File sfwOutputFile   = new File(modelOutputPath,
					format(AIRConstants.FILE_NAME_WORD_SEN));

		if (! Utils.save(getParameters(), paramOutputFile, print)) {
			Utils.err(ErrorMessage.ERROR_STORE_PARAMS);
			if (res) res = false;
		}
		if (! storeTopicForEachWord(tfwOutputFile)) {
			Utils.err(ErrorMessage.ERROR_STORE_TOPIC_FOR_WORD);
			if (res) res = false;
		}
		if (! storeSentimentForEachWord(sfwOutputFile)) {
			Utils.err(ErrorMessage.ERROR_STORE_SEN_FOR_WORD);
			if (res) res = false;
		}

		return res;
	}

	public boolean storeEstimateResult(boolean print) {
		boolean res = storeModel(print);

		File topwordOutputFile         = new File(modelOutputPath,
					format(AIRConstants.FILE_NAME_TOP_WORD));
		File tOutputFile               = new File(modelOutputPath,
					format(AIRConstants.FILE_NAME_PARAM_T));
		File thetaOutputFile           = new File(modelOutputPath,
					format(AIRConstants.FILE_NAME_PARAM_THETA));
		File docTopicOutputFile        = new File(modelOutputPath,
					format(AIRConstants.FILE_NAME_DOC_TOPIC));
		File docSenOutputFile          = new File(modelOutputPath,
					format(AIRConstants.FILE_NAME_DOC_SEN));
		File topicSenWordSumOutputFile = new File(modelOutputPath,
					format("TopicSentimentWordSum"));

		if (! storeTopWords(topwordOutputFile, false, phi,
						topicSentimentWord)) {
			Utils.err(ErrorMessage.ERROR_STORE_TOP_WORD);
			if (res) res = false;
		}
		if (! storeScore(omega, getScoreFile(modelOutputPath),
				AIRConstants.CONFIG_MANAGER.getRatingScaler(), print)) {
			Utils.err(ErrorMessage.ERROR_STORE_SCORE);
			if (res) res = false;
		}
		if (! Utils.save(t, AIRConstants.SPACER_TAB, tOutputFile, print)) {
			Utils.err(ErrorMessage.ERROR_STORE_PARAM_T);
			if (res) res = false;
		}
		if (! Utils.save(theta, AIRConstants.SPACER_TAB,
						thetaOutputFile, print)) {
			Utils.err(ErrorMessage.ERROR_STORE_PARAM_THETA);
			if (res) res = false;
		}
		if (! storeTopicSentimentWord(modelOutputPath)) {
			Utils.err(ErrorMessage.ERROR_STORE_TOPIC_SEN_WORD_COUNT);
			if (res) res = false;
		}
		if (! Utils.save(documentTopic, AIRConstants.SPACER_TAB,
						docTopicOutputFile, print)) {
			Utils.err(ErrorMessage.ERROR_STORE_DOC_TOPIC_COUNT);
			if (res) res = false;
		}
		if (! storeDocumentTopicSentiment(modelOutputPath)) {
			Utils.err(ErrorMessage.ERROR_STORE_DOC_TOPIC_SEN_COUNT);
			if (res) res = false;
		}
		if (! Utils.save(documentSentiment, AIRConstants.SPACER_TAB,
						docSenOutputFile, print)) {
			Utils.err(ErrorMessage.ERROR_STORE_DOC_SEN_COUNT);
			if (res) res = false;
		}
		if (! Utils.save(topicSentimentWordSum, AIRConstants.SPACER_TAB,
						topicSenWordSumOutputFile, print)) {
			Utils.err(ErrorMessage.ERROR_STORE_TOPIC_SEN_COUNT);
			if (res) res = false;
		}

		for (int sentiment = 0; sentiment < SENTIMENT_NUM + 1; sentiment ++) {
			File phiOutputFile = new File(modelOutputPath, format(
						AIRConstants.FILE_NAME_PARAM_PHI +
							"_" + sentiment));
			if (sentiment < SENTIMENT_NUM) {
				Utils.save(phi[sentiment], AIRConstants.SPACER_TAB,
						phiOutputFile, true);
			} else {
				Utils.save(sumWordProb(phi), AIRConstants.SPACER_TAB,
						phiOutputFile, true);
			}
		}

		return res;
	}

	private int getRandomTopicIndex(int word) {
		double[] probs = new double[TOPIC_NUM];
		double   sum   = 0.0;
		double random  = Math.random();
		int      topic = 0;
		for (topic = 0; topic < TOPIC_NUM; topic ++) {
			sum += (probs[topic] = betas[topic][word]);
		}
		for (topic = 0; topic < TOPIC_NUM; topic ++) {
			probs[topic] /= sum;
			if (topic != 0) probs[topic] += probs[topic - 1];
			if (probs[topic] > random) {
				break;
			}
		}
		if (topic >= TOPIC_NUM) {
			// never reaches
			Utils.err("Failed to sample topic : sum = " + sum + " , random = " + random);
		}

		return topic;
	}

	private int getRandomSentimentIndex(double rating) {
		double[] p    = new double[SENTIMENT_NUM];
		int sentiment = 0;

		p[0] = gammas[0] / (gammas[0] + gammas[1]);
		p[1] = (1 - p[0]) * rating;
		p[2] = (1 - p[0]) * (1 - rating);

		double u = Math.random();
		for (sentiment = 0; sentiment < p.length; sentiment ++) {
			if (sentiment != 0) {
				p[sentiment] += p[sentiment - 1];
			}
			if (u < p[sentiment]) break;
		}

		if (sentiment >= SENTIMENT_NUM) {
			// never reaches
			sentiment = SENTIMENT_NUM - 1;
			Utils.err("Error occurs when initing sentiment index.");
		}

		return sentiment;
	}

	private void parseParamArguments(File datasetFile,
			ParamManager paramManager) throws AIRException,
						TopicModelException {
		if (paramManager == null) {
			throw new AIRException(ErrorMessage.ERROR_NO_ARGS);
		}

		// init dictionary
		dictionary = new Dictionary();
		// load data set
		data = getDataset(datasetFile, dictionary, new File(
					paramManager.getModelOutputPath(),
						AIRConstants.FILE_NAME_DIC_MAP),
								false);
		data.getParams().list(System.out);

		if (paramManager.getTopicNum() != null) {
			this.TOPIC_NUM = paramManager.getTopicNum();
		} else {
			throw new AIRException(ErrorMessage.ERROR_NO_TOPIC_NUM);
		}

		if (paramManager.getTopWordNum() != null) {
			this.TOP_WORD_NUM = paramManager.getTopWordNum();
		} else {
			throw new AIRException(ErrorMessage.ERROR_NO_TOP_WORD_NUM);
		}

		if (paramManager.getBurninIterNum() != null) {
			this.burninIterNum = paramManager.getBurninIterNum();
		} else {
			throw new AIRException(ErrorMessage.ERROR_NO_BURNIN_ITER_NUM);
		}

		if (paramManager.getEstimateIterNum() != null) {
			this.estimateIterNum = paramManager.getEstimateIterNum();
		} else {
			throw new AIRException(ErrorMessage.ERROR_NO_EST_ITER_NUM);
		}

		if (paramManager.getMaxAlphaEstNum() != null) {
			this.maxAlphaEstIterNum = paramManager.getMaxAlphaEstNum();
		} else {
			throw new AIRException(ErrorMessage.ERROR_NO_MAX_ALPHA_EST_ITER_NUM);
		}

		if (paramManager.getMaxLambdaEstNum() != null) {
			this.maxLambdaEstIterNum = paramManager.getMaxLambdaEstNum();
		} else
		if (paramManager.isOptimizeLambda()) {
			throw new AIRException(ErrorMessage.ERROR_NO_MAX_LAMBDA_EST_ITER_NUM);
		}
		
		if (paramManager.getGammas() != null &&
				paramManager.getGammas().length == 2) {
			this.gammas = Arrays.copyOf(paramManager.getGammas(),
						paramManager.getGammas().length);
		} else {
			throw new AIRException(ErrorMessage.ERROR_NO_GAMMA);
		}

		if (paramManager.getLambda() != null) {
			this.lambdaInit = paramManager.getLambda();
			this.lambdas    = new double[TOPIC_NUM];
			for (int topic = 0; topic < TOPIC_NUM; topic ++) {
				lambdas[topic] = lambdaInit;
			}
			//updateLambdaHat();
		} else {
			throw new AIRException(ErrorMessage.ERROR_NO_LAMBDA);
		}

		if (paramManager.getBetaInit() != null) {
			this.betaInit = paramManager.getBetaInit();
		} else {
			throw new AIRException(ErrorMessage.ERROR_NO_BETA);
		}

		if (paramManager.getKeywordFile() != null) {
			keywords = loadKeywords(paramManager.getKeywordFile(),
								dictionary);
		}

		alphas = new double[TOPIC_NUM];
		for (int topic = 0; topic < alphas.length; topic ++) {
			alphas[topic] = 2.0 / TOPIC_NUM; //2.0 / TOPIC_NUM; //50/TOPIC_NUM; //0.01; // 0.1;
		}
	}

	private boolean storeTopicSentimentWord(File path) {
		BufferedWriter writer = null;
		try {
			writer = Utils.createBufferedWriter(new File(path,
						format(AIRConstants.FILE_NAME_TOP_SEN_W)));

			for (int word = 0; word < dictionary.getSize(); word ++) {
				StringBuffer buffer = new StringBuffer();
				for (int topic = 0; topic < topicSentimentWord.length; topic ++) {
					for (int sentiment = 0; sentiment < topicSentimentWord[topic].length; sentiment ++) {
						buffer.append(topicSentimentWord[topic][sentiment][word]);
						if (sentiment != topicSentimentWord[topic].length - 1) {
							buffer.append(AIRConstants.SPACER_COMMA);
						}
					}
					if (topic != topicSentimentWord.length - 1) {
						buffer.append(AIRConstants.SPACER_TAB);
					}
				}
				writer.write(buffer.toString());
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

	private boolean storeDocumentTopicSentiment(File path) {
		BufferedWriter writer = null;
		try {
			writer = Utils.createBufferedWriter(new File(path,
					format(AIRConstants.FILE_NAME_DOC_TOP_SEN)));
			for (int doc = 0; doc < documentTopicSentiment.length; doc ++) {
				StringBuffer buffer = new StringBuffer();
				for (int topic = 0; topic < documentTopicSentiment[doc].length; topic ++) {
					for (int sentiment = 0; sentiment < documentTopicSentiment[doc][topic].length; sentiment ++) {
						buffer.append(documentTopicSentiment[doc][topic][sentiment]);
						if (sentiment != documentTopicSentiment[doc][topic].length - 1) {
							buffer.append(AIRConstants.SPACER_COMMA);
						}
					}
					if (topic != documentTopicSentiment[doc].length - 1) {
						buffer.append(AIRConstants.SPACER_TAB);
					}
				}
				writer.write(buffer.toString());
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

	private boolean storeTopicForEachWord(File outputFile) {
		BufferedWriter writer = null;
		try {
			writer = Utils.createBufferedWriter(outputFile);
			for (int doc = 0; doc < data.getDocumentSize(); doc ++) {
				StringBuffer buffer = new StringBuffer();
				for (int i = 0; i < data.getDocuments().get(doc).wordNum; i ++) {
					buffer.append(data.getDocuments().get(doc).topics[i]);
					if (i != data.getDocuments().get(doc).wordNum - 1) {
						buffer.append(" ");
					}
				}
				writer.write(buffer.toString());
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

	private boolean storeSentimentForEachWord(File outputFile) {
		BufferedWriter writer = null;
		try {
			writer = Utils.createBufferedWriter(outputFile);
			for (int doc = 0; doc < data.getDocumentSize(); doc ++) {
				StringBuffer buffer = new StringBuffer();
				for (int i = 0; i < data.getDocuments().get(doc).wordNum; i ++) {
					buffer.append(data.getDocuments().get(doc).sentiments[i]);
					if (i != data.getDocuments().get(doc).wordNum - 1) {
						buffer.append(" ");
					}
				}
				writer.write(buffer.toString());
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

	public void loadModel(File datasetFile, File path)
					throws AIRException, TopicModelException {
		// init dictionary
		if (dictionary == null) {
			dictionary = new Dictionary();
		}
		dictionary.addDictionaryListener(this);
		// load data set
		data = getDataset(datasetFile, dictionary, new File(path,
					AIRConstants.FILE_NAME_DIC_MAP), true);

		if (! loadParameters(path)) throw new AIRException ("Failed to load model parameters");

		if (! loadTopicForEachWord(path)) {
			throw new AIRException("Failed to parse topic_word.");
		}

		if (! loadSentimentForEachWord(path)) {
			throw new AIRException("Failed to parse sentiment_word."); 
		}

		topicSentimentWord        = new int[TOPIC_NUM][SENTIMENT_NUM][dictionary.getSize()];
		topicSentimentWordSum     = new int[TOPIC_NUM][SENTIMENT_NUM];
		documentTopic             = new int[data.getDocumentSize()][TOPIC_NUM];
		documentTopicSum          = new int[data.getDocumentSize()];
		documentSentiment         = new int[data.getDocumentSize()][2];
		documentSentimentSum      = new int[data.getDocumentSize()];
		documentTopicSentiment    = new int[data.getDocumentSize()][TOPIC_NUM][2];
		documentTopicSentimentSum = new int[data.getDocumentSize()][TOPIC_NUM];

		for (int i = 0; i < data.getDocumentSize(); i ++) {
			for (int j = 0; j < data.getDocuments().get(i).wordNum; j ++) {
				int topic          = data.getDocuments().get(i).topics[j];
				int sentimentIndex = data.getDocuments().get(i).sentiments[j];
				int word           = data.getDocuments().get(i).words[j];

				topicSentimentWord[topic][sentimentIndex][word] ++;
				topicSentimentWordSum[topic][sentimentIndex] ++;
				documentTopic[i][topic] ++;
				documentTopicSum[i] ++;
				documentSentiment[i][sentimentIndex == 0 ? 0 : 1] ++;
				documentSentimentSum[i] ++;
				if (sentimentIndex > 0) {
					documentTopicSentiment[i][topic][sentimentIndex - 1] ++;
					documentTopicSentimentSum[i][topic] ++;
				}
			}
		}
	}

	private boolean loadParameters(File path) throws AIRException {
		File inputFile = new File(path, AIRConstants.FILE_NAME_PARAM);
		if (inputFile.exists()) {
			BufferedReader reader = null;
			try {
				reader = Utils.createBufferedReader(inputFile);
				Properties props = new Properties();
				props.load(reader);

				if (Integer.parseInt(props.getProperty(
						AIRConstants.STRING_DIC_NUM)) !=
							dictionary.getSize()) {
					throw new AIRException("The size of loaded dictionay is different from the one parsed from parameter file.");
				}
				int documentNum = Integer.parseInt(props.getProperty(
						AIRConstants.STRING_DOCUMENT_NUM));
				if (documentNum != data.getDocumentSize()) {
					throw new AIRException("The size of loaded document is different from the one parsed from parameter file.");
				}

				/*beta             = Double.parseDouble(props.getProperty(
							AIRConstants.STRING_BETA));*/
				/*lambda             = Double.parseDouble(props.getProperty(
							AIRConstants.STRING_LAMBDA));*/
				burninIterNum       = Integer.parseInt(props.getProperty(
							AIRConstants.STRING_BURNIN_ITER_NUM));
				estimateIterNum     = Integer.parseInt(props.getProperty(
							AIRConstants.STRING_EST_ITER_NUM));
				burninIter          = Integer.parseInt(props.getProperty(
							AIRConstants.STRING_BURNIN_ITER));
				maxAlphaEstIterNum  = Integer.parseInt(props.getProperty(
							AIRConstants.STRING_MAX_ALPHA_EST_I_NUM));
				maxLambdaEstIterNum = Integer.parseInt(props.getProperty(
							AIRConstants.STRING_MAX_LAMBDA_EST_I_NUM));
				alphaEstIter        = Integer.parseInt(props.getProperty(
							AIRConstants.STRING_ALPHA_EST_ITER));
				TOPIC_NUM           = Integer.parseInt(props.getProperty(
							AIRConstants.STRING_TOPIC_NUM));
				TOP_WORD_NUM          = Integer.parseInt(props.getProperty(
							AIRConstants.STRING_TOP_WORD_NUM));
				gammas              = AIRUtils.parseGammas(AIRConstants.STRING_GAMMAS);

				String salpha = props.getProperty(AIRConstants.STRING_ALPHAS);
				double[] dalpha = Utils.parseDoubleArray(salpha, ",");
				if (dalpha != null && dalpha.length == TOPIC_NUM) {
					alphas = dalpha;
				} else {
					throw new AIRException ("Failed to load alpha value.");
				}

				return true;
			}catch (NullPointerException e) {
				e.printStackTrace();
			} catch (NumberFormatException e) {
				e.printStackTrace();
			} catch (IOException e) {
				e.printStackTrace();
			} finally {
				Utils.cleanup(reader);
			}
		}
		return false;
	}

	private boolean loadTopicForEachWord(File path) throws AIRException {
		BufferedReader topicReader = null;
		//CsvReader
		try {
			topicReader = Utils.createBufferedReader(new File(path,
						AIRConstants.FILE_NAME_WORD_TOPIC));
			String line = null;
			int doc = 0;
			while ((line = topicReader.readLine()) != null) {
				if (Utils.isEmpty((line=line.trim()))) continue;

				String[] array = line.split(" ");
				int word       = 0;
				for (String stopic : array) {
					if (Utils.isEmpty(stopic)) continue;

					int topic = Integer.parseInt(stopic);
					if (topic < TOPIC_NUM && topic >= 0) {
						data.getDocuments().get(doc).topics[word] = topic;
					} else {
						throw new AIRException("TopicWord format is not correct: topic assinment is not corrrect.");
					}
					word ++;
				}
				if (word != data.getDocuments().get(doc).wordNum) {
					throw new AIRException("TopicWord format is not correct: word number is not matched.");
				}
				doc ++;
			}
			if (doc != data.getDocumentSize()) {
				throw new AIRException("TopicWord format is not correct: document number is not matched.");
			}
			return true;
		} catch (NumberFormatException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		} finally {
			Utils.cleanup(topicReader);
		}
		return false;
	}

	private boolean loadSentimentForEachWord(File path) throws AIRException {
		BufferedReader sentimentReader = null;
		try {
			sentimentReader = Utils.createBufferedReader(new File(path,
								AIRConstants.FILE_NAME_WORD_SEN));
			int doc         = 0;
			String line     = null;
			while ((line = sentimentReader.readLine()) != null) {
				if (Utils.isEmpty((line=line.trim()))) continue;

				String[] array = line.split(" ");
				int word       = 0;
				for (String ssentiment : array) {
					if (Utils.isEmpty(ssentiment)) continue;

					int sentiment = Integer.parseInt(ssentiment);
					if (sentiment < SENTIMENT_NUM && sentiment >= 0) {
						data.getDocuments().get(doc).sentiments[word] = Integer.parseInt(ssentiment);
					} else {
						throw new AIRException("SentimentWord format is not correct: sentiment assinment is not corrrect.");
					}
					word ++;
				}
				if (word != data.getDocuments().get(doc).wordNum) {
					throw new AIRException("SentimentWord format is not correct: word number is not matched.");
				}
				doc ++;
				
			}
			if (doc != data.getDocumentSize()) {
				throw new AIRException("SentimentWord format is not correct: document number is not matched.");
			}
			return true;
		} catch (NumberFormatException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		} finally {
			Utils.cleanup(sentimentReader);
		}
		return false;
	}
}
