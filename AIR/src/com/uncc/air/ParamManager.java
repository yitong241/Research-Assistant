package com.uncc.air;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Properties;

import com.lhy.tool.util.Utils;
import com.uncc.air.data.RateBeerDataset;
import com.uncc.air.data.RatingScaler;
import com.uncc.air.util.AIRUtils;

/**
 * @author Huayu Li
 */
public class ParamManager {
	public static final int IMPLEMENT_METHOD_GENERAL_SAMPLING = 0;
	public static final int IMPLEMENT_METHOD_GENERAL_MAP      = 1;
	public static final int IMPLEMENT_METHOD_GENERAL_VB       = 2;
	public static final int IMPLEMENT_METHOD_SHORT_REVIEW_MAP = 3;

	public static final String FORMAT_PARAMETERS         =
			"[Environment Setting]" 			  + com.lhy.tool.Constants.LINE_SEPARATOR +
			AIRConstants.STRING_MODEL_OUTPUT_PATH    + "=%s"  + com.lhy.tool.Constants.LINE_SEPARATOR +
									    com.lhy.tool.Constants.LINE_SEPARATOR +
			AIRConstants.STRING_DATA_TRAIN           + "=%s"  + com.lhy.tool.Constants.LINE_SEPARATOR +
			AIRConstants.STRING_DATA_VALIDATION      + "=%s"  + com.lhy.tool.Constants.LINE_SEPARATOR +
			AIRConstants.STRING_DATA_TEST            + "=%s"  + com.lhy.tool.Constants.LINE_SEPARATOR +
									    com.lhy.tool.Constants.LINE_SEPARATOR +
			AIRConstants.STRING_IS_USE_RATEBEER_DATA + "=%s"  + com.lhy.tool.Constants.LINE_SEPARATOR +
									    com.lhy.tool.Constants.LINE_SEPARATOR +
			"[Model Setting]"    				  + com.lhy.tool.Constants.LINE_SEPARATOR +
			AIRConstants.STRING_IMPLEMENT_METHOD_NOTE         + com.lhy.tool.Constants.LINE_SEPARATOR +
			AIRConstants.STRING_IMPLEMENT_METHOD     + "=%s"  + com.lhy.tool.Constants.LINE_SEPARATOR +
									    com.lhy.tool.Constants.LINE_SEPARATOR +
			AIRConstants.STRING_IS_RESTORE           + "=%s"  + com.lhy.tool.Constants.LINE_SEPARATOR +
			AIRConstants.STRING_IS_OPTIMIZE_LAMBDA   + "=%s"  + com.lhy.tool.Constants.LINE_SEPARATOR +
									    com.lhy.tool.Constants.LINE_SEPARATOR +
			AIRConstants.STRING_TOPIC_NUM            + "=%s"  + com.lhy.tool.Constants.LINE_SEPARATOR +
			AIRConstants.STRING_LAMBDA               + "=%s"  + com.lhy.tool.Constants.LINE_SEPARATOR +
			AIRConstants.STRING_GAMMAS               + "=%s"  + com.lhy.tool.Constants.LINE_SEPARATOR +
			AIRConstants.STRING_BETA_INIT            + "=%s"  + com.lhy.tool.Constants.LINE_SEPARATOR +
			AIRConstants.STRING_TOP_WORD_NUM         + "=%s"  + com.lhy.tool.Constants.LINE_SEPARATOR +
									    com.lhy.tool.Constants.LINE_SEPARATOR +
			AIRConstants.STRING_BURNIN_ITER_NUM      + "=%s"  + com.lhy.tool.Constants.LINE_SEPARATOR +
			AIRConstants.STRING_EST_ITER_NUM         + "=%s"  + com.lhy.tool.Constants.LINE_SEPARATOR +
			AIRConstants.STRING_MAX_ALPHA_EST_I_NUM  + "=%s"  + com.lhy.tool.Constants.LINE_SEPARATOR +
			AIRConstants.STRING_MAX_LAMBDA_EST_I_NUM + "=%s"  + com.lhy.tool.Constants.LINE_SEPARATOR +
									    com.lhy.tool.Constants.LINE_SEPARATOR +
			AIRConstants.STRING_KEYWORD_FILE         + "=%s"  + com.lhy.tool.Constants.LINE_SEPARATOR +
			AIRConstants.STRING_GROUNDTRUTH_FILE     + "=%s"  + com.lhy.tool.Constants.LINE_SEPARATOR +
			AIRConstants.STRING_TW_TOPIC_WORD_FILE   + "=%s"  + com.lhy.tool.Constants.LINE_SEPARATOR +
									    com.lhy.tool.Constants.LINE_SEPARATOR +
			"[Left To Right - Gibbs Sampling Setting]"        + com.lhy.tool.Constants.LINE_SEPARATOR +
			AIRConstants.STRING_PARTICLE_NUM         + "=%s"  + com.lhy.tool.Constants.LINE_SEPARATOR +
									    com.lhy.tool.Constants.LINE_SEPARATOR +
			"[DEBUG]"                                         + com.lhy.tool.Constants.LINE_SEPARATOR +
			AIRConstants.STRING_DEBUG_CONVERGENE     + "=%s"  + com.lhy.tool.Constants.LINE_SEPARATOR +
									    com.lhy.tool.Constants.LINE_SEPARATOR +
			"[Greedy Search]"                                 + com.lhy.tool.Constants.LINE_SEPARATOR +
			AIRConstants.STRING_IS_GREED_SEARCH      + "=%s"  + com.lhy.tool.Constants.LINE_SEPARATOR +
									    com.lhy.tool.Constants.LINE_SEPARATOR +
			AIRConstants.STRING_IS_USE_MULTI_THREAD  + "=%s"  + com.lhy.tool.Constants.LINE_SEPARATOR +
			AIRConstants.STRING_THREAD_NUM           + "=%s"  + com.lhy.tool.Constants.LINE_SEPARATOR +
									    com.lhy.tool.Constants.LINE_SEPARATOR +
			AIRConstants.STRING_TOPIC_NUM_ARRAY      + "=%s"  + com.lhy.tool.Constants.LINE_SEPARATOR +
			AIRConstants.STRING_LAMBDA_ARRAY         + "=%s"  + com.lhy.tool.Constants.LINE_SEPARATOR +
			AIRConstants.STRING_GAMMAS_ARRAY         + "=%s"  + com.lhy.tool.Constants.LINE_SEPARATOR;

	private Integer topicNum                 = null;
	private Integer particleNum              = null;
	private Integer topWordNum               = null;
	private Integer burninIterNum            = null;
	private Integer estimateIterNum          = null;
	private Integer maxAlphaEstIterNum       = null;
	private Integer maxLambdaEstIterNum      = null;
	private Integer threadNum                = null;
	private int[] topicNumArray              = null;
	private File modelOutputPath             = null;
	private File dataTrainFile               = null;
	private File dataTestFile                = null;
	private File dataValidationFile          = null;
	private File keywordFile                 = null;
	private File groundtruthScoreFile        = null;
	private File groundtruthTopicWordDisFile = null;
	private Double lambda                    = null;
	private Double betaInit                  = null;
	private double[] lambdaArray             = null;
	private double[] gammas                  = null;
	private double[][] gammasArray           = null;
	private boolean debugConvergence         = false;
	private boolean isRestore                = false;
	private boolean isOptimizeLambda         = false;
	private boolean isGreedSearch            = false;
	private boolean isUseMultiThread         = false;
	private boolean isUseRatebeerData        = false;
	private RatingScaler ratingScaler        = null;
	private int implementMethod              = IMPLEMENT_METHOD_GENERAL_SAMPLING;

	public ParamManager() {}

	public ParamManager(double[] gammas, double lambda,  Integer topicNum) {
		this.gammas   = gammas;
		this.topicNum = topicNum;
		this.lambda   = lambda;
	}

	public Integer getMaxAlphaEstNum() {
		return maxAlphaEstIterNum;
	}

	public Integer getMaxLambdaEstNum() {
		return maxLambdaEstIterNum;
	}

	public Double getBetaInit() {
		return betaInit;
	}

	public Integer getBurninIterNum() {
		return burninIterNum;
	}

	public Integer getEstimateIterNum() {
		return estimateIterNum;
	}

	public File getDataTestFile() {
		return dataTestFile;
	}

	public File getDataTrainFile() {
		return dataTrainFile;
	}

	public File getDataValidationFile() {
		return dataValidationFile;
	}

	public double[] getGammas() {
		return gammas;
	}

	public double[][] getGammasArray() {
		return gammasArray;
	}

	public File getGroundtruthTopicWordDistributionFile() {
		return groundtruthTopicWordDisFile;
	}
	
	public File getGroundtruthScoreFile() {
		return groundtruthScoreFile;
	}

	public int getImplementMethod() {
		return implementMethod;
	}

	public Double getLambda() {
		return lambda;
	}

	public double[] getLambdaArray() {
		return lambdaArray;
	}

	public File getKeywordFile() {
		return keywordFile;
	}

	public File getModelOutputPath() {
		return modelOutputPath;
	}

	public Integer getParticleNum() {
		return particleNum;
	}

	public Integer getThreadNum() {
		return threadNum;
	}

	public Integer getTopicNum() {
		return topicNum;
	}

	public int[] getTopicNumArray() {
		return topicNumArray;
	}

	public Integer getTopWordNum() {
		return topWordNum;
	}

	public RatingScaler getRatingScaler() {
		return ratingScaler;
	}

	public boolean isDebugConvergence() {
		return debugConvergence;
	}

	public boolean isGreedSearch() {
		return isGreedSearch;
	}

	public boolean isOptimizeLambda() {
		return isOptimizeLambda;
	}

	public boolean isRestore() {
		return isRestore;
	}

	public boolean isUseMultiThread() {
		return isUseMultiThread;
	}

	public void setBetaInit(double betaInit) {
		this.betaInit = betaInit;
	}

	public void setEstimateIterNum (int estimateIterNum) {
		this.estimateIterNum = estimateIterNum;
	}

	public void setBurninIterNum(int burninIterNum) {
		this.burninIterNum = burninIterNum;
	}

	public void setDataTestFile(File dataTestFile) {
		this.dataTestFile = dataTestFile;
	}

	public void setDataTestFile(String sdataTestFile) {
		if (! Utils.isEmpty(sdataTestFile)) {
			this.dataTestFile = new File(sdataTestFile);
		}
	
		if (! Utils.exists(this.dataTestFile)) {
			Utils.println(ErrorMessage.ERROR_NO_TEST_FILE);
		}
	}

	public void setDataTrainFile(File dataTrainFile) {
		this.dataTrainFile = dataTrainFile;
	}

	public void setDataTrainFile(String sdataTrainFile) {
		if (! Utils.isEmpty(sdataTrainFile)) {
			this.dataTrainFile = new File(sdataTrainFile);
		}

		if (! Utils.exists(this.dataTrainFile)) {
			System.out.println(dataTrainFile.getAbsolutePath());
			Utils.err(ErrorMessage.ERROR_NO_TRAIN_FILE);
		}
	}

	public void setDataValidationFile(File dataValidationFile) {
		this.dataValidationFile = dataValidationFile;
	}

	public void setDataValidationFile(String sdataValidationFile) {
		if (! Utils.isEmpty(sdataValidationFile)) {
			this.dataValidationFile = new File(sdataValidationFile);
		}

		if (! Utils.exists(this.dataValidationFile)) {
			Utils.println(ErrorMessage.ERROR_NO_VALIDATION_FILE);
		}
	}

	public void setDebugConvergence(boolean debugConvergence) {
		this.debugConvergence = debugConvergence;
	}

	public void setGammas(double[] gammas) {
		this.gammas = gammas;
	}

	public void setGammasArray(double[][] gammasArray) {
		this.gammasArray = gammasArray;
	}

	public void setGreedSearch(boolean isGreedSearch) {
		this.isGreedSearch = isGreedSearch;
	}

	public void setGroundtruthTopicWordDistributionFile(
				File groundtruthTopicWordDisFile) {
		this.groundtruthTopicWordDisFile = groundtruthTopicWordDisFile;
	}

	public void setGroundtruthTopicWordDistributionFile(
			String groundtruthTopicWordDisFile) {
		if (! Utils.isEmpty(groundtruthTopicWordDisFile)) {
			this.groundtruthTopicWordDisFile = new File(
						groundtruthTopicWordDisFile);
			if (! this.groundtruthTopicWordDisFile.exists()) {
				Utils.err(String.format(
						ErrorMessage.ERROR_EMPTY_GROUNDTRUTH_TW_FILE,
						this.groundtruthTopicWordDisFile.getAbsolutePath()));
			}
		}
	}

	public void setGroundTruthFile(File groundtruthScoreFile) {
		this.groundtruthScoreFile = groundtruthScoreFile;
	}

	public void setGroundTruthFile(String groundtruthScoreFile) {
		if (! Utils.isEmpty(groundtruthScoreFile)) {
			this.groundtruthScoreFile = new File(groundtruthScoreFile);
			if (! this.groundtruthScoreFile.exists()) {
				Utils.err(String.format(ErrorMessage.ERROR_EMPTY_GROUNDTRUTH_FILE,
						this.groundtruthScoreFile.getAbsolutePath()));
			}
		}
	}

	public void setImplementMethod(int implementMethod) {
		this.implementMethod = implementMethod;
	}

	public void setLambda(double lambda) {
		this.lambda = lambda;
	}

	public void setLambdaArray(double[] lambdaArray) {
		this.lambdaArray = lambdaArray;
	}

	public void setKeywordFile(File keywordFile) {
		this.keywordFile = keywordFile;
	}

	public void setKeywordFile(String keywordFile) {
		if (! Utils.isEmpty(keywordFile)) {
			this.keywordFile = new File(keywordFile);
			if (! this.keywordFile.exists()) {
				Utils.err(String.format(ErrorMessage.ERROR_EMPTY_KEYWORD_FILE,
						this.keywordFile.getAbsolutePath()));
			}
		}
	}

	public void setMaxAlphaEstIterNum(int maxAlphaEstIterNum) {
		this.maxAlphaEstIterNum = maxAlphaEstIterNum;
	}

	public void setMaxLambdaEstIterNum(int maxLambdaEstIterNum) {
		this.maxLambdaEstIterNum = maxLambdaEstIterNum;
	}

	public void setModelOutputPath(String smodelOutputPath) {
		if (! Utils.isEmpty(smodelOutputPath)) {
			this.modelOutputPath = new File(smodelOutputPath);
		}
	}

	public void setModelOutputPath(File modelOutputPath) {
		this.modelOutputPath = modelOutputPath;
	}

	public void setOptimizeLambda(boolean isOptimizeLambda) {
		this.isOptimizeLambda = isOptimizeLambda;
	}

	public void setParticleNum(int particleNum) {
		this.particleNum = particleNum;
	}

	public void setRestore(boolean isRestore) {
		this.isRestore = isRestore;
	}

	public void setThreadNum(int threadNum) {
		this.threadNum = threadNum;
	}

	public void setTopicNum(int topicNum) {
		this.topicNum = topicNum;
	}

	public void setTopicNumArray(int[] topicNumArray) {
		this.topicNumArray = topicNumArray;
	}

	public void setTopWordNum(int topWordNum) {
		this.topWordNum = topWordNum;
	}

	public void setRatingScaler(RatingScaler ratingScaler) {
		this.ratingScaler = ratingScaler;
	}

	public void setUseMultiThread(boolean isUseMultiThread) {
		this.isUseMultiThread = isUseMultiThread;
	}

	// must set after setting keyword
	public void setUseRateBeerData(boolean isUseRatebeerData) {
		this.isUseRatebeerData = isUseRatebeerData;
		if (this.isUseRatebeerData) {
			this.ratingScaler = new RateBeerDataset((this.keywordFile != null));
		}
	}

	@Override
	public String toString() {
		return String.format(FORMAT_PARAMETERS,
					Utils.wrapPath(modelOutputPath),
					Utils.wrapPath(dataTrainFile),
					Utils.wrapPath(dataValidationFile),
					Utils.wrapPath(dataTestFile),
					isUseRatebeerData,
					implementMethod,
					isRestore,
					isOptimizeLambda,
					topicNum,
					lambda,
					Utils.convertToString(gammas, ", "),
					betaInit,
					topWordNum,
					burninIterNum,
					estimateIterNum,
					maxAlphaEstIterNum,
					maxLambdaEstIterNum,
					Utils.wrapPath(keywordFile),
					Utils.wrapPath(groundtruthScoreFile),
					Utils.wrapPath(groundtruthTopicWordDisFile),
					particleNum,
					debugConvergence,
					isGreedSearch,
					isUseMultiThread,
					threadNum,
					topicNumArray == null ? "" : Utils.convertToString(topicNumArray, AIRConstants.SPACER_SEMICOLON),
					lambdaArray == null ? "" : Utils.convertToString(lambdaArray, AIRConstants.SPACER_SEMICOLON),
					wrapGammasArray(gammasArray));
	}

	public void load(File inputFile) throws AIRException {
		if (! Utils.exists(inputFile)) {
			throw new AIRException(String.format("Config file doesn't exist. [ %s ]",
					inputFile == null ? inputFile : inputFile.getAbsolutePath()));
		}

		BufferedReader confReader = null;
		try {
			confReader       = Utils.createBufferedReader(inputFile);
			Properties props = new Properties();
			props.load(confReader);

			String sprop = props.getProperty(AIRConstants.STRING_MODEL_OUTPUT_PATH);
			if (! Utils.isEmpty(sprop)) {
				setModelOutputPath(sprop);
			}

			sprop = props.getProperty(AIRConstants.STRING_DATA_TRAIN);
			if (! Utils.isEmpty(sprop)) {
				setDataTrainFile(sprop);
			}

			sprop = props.getProperty(AIRConstants.STRING_DATA_VALIDATION);
			if (! Utils.isEmpty(sprop)) {
				setDataValidationFile(sprop);
			}

			sprop = props.getProperty(AIRConstants.STRING_DATA_TEST);
			if (! Utils.isEmpty(sprop)) {
				setDataTestFile(sprop);
			}

			setGammas(AIRUtils.parseGammas(props.getProperty(AIRConstants.STRING_GAMMAS)));

			sprop = props.getProperty(AIRConstants.STRING_LAMBDA);
			if (! Utils.isEmpty(sprop)) {
				setLambda(AIRUtils.getArithmeticResult(sprop));
			}

			sprop = props.getProperty(AIRConstants.STRING_BETA_INIT);
			if (! Utils.isEmpty(sprop)) {
				setBetaInit(Double.parseDouble(sprop));
			}

			sprop = props.getProperty(AIRConstants.STRING_KEYWORD_FILE);
			if (! Utils.isEmpty(sprop)) {
				setKeywordFile(sprop);
			}

			sprop = props.getProperty(AIRConstants.STRING_GROUNDTRUTH_FILE);
			if (! Utils.isEmpty(sprop)) {
				setGroundTruthFile(sprop);
			}

			sprop = props.getProperty(AIRConstants.STRING_TW_TOPIC_WORD_FILE);
			if (! Utils.isEmpty(sprop)) {
				setGroundtruthTopicWordDistributionFile(sprop);
			}

			sprop = props.getProperty(AIRConstants.STRING_PARTICLE_NUM);
			if (! Utils.isEmpty(sprop)) {
				setParticleNum(Integer.parseInt(sprop));
			}

			sprop = props.getProperty(AIRConstants.STRING_TOPIC_NUM);
			if (! Utils.isEmpty(sprop)) {
				setTopicNum(Integer.parseInt(sprop));
			}

			sprop = props.getProperty(AIRConstants.STRING_TOP_WORD_NUM);
			if (! Utils.isEmpty(sprop)) {
				setTopWordNum(Integer.parseInt(sprop));
			}

			sprop = props.getProperty(AIRConstants.STRING_BURNIN_ITER_NUM);
			if (! Utils.isEmpty(sprop)) {
				setBurninIterNum(Integer.parseInt(sprop));
			}

			sprop = props.getProperty(AIRConstants.STRING_EST_ITER_NUM);
			if (! Utils.isEmpty(sprop)) {
				setEstimateIterNum(Integer.parseInt(sprop));
			}

			sprop = props.getProperty(AIRConstants.STRING_MAX_ALPHA_EST_I_NUM);
			if (! Utils.isEmpty(sprop)) {
				setMaxAlphaEstIterNum(Integer.parseInt(sprop));
			}

			sprop = props.getProperty(AIRConstants.STRING_MAX_LAMBDA_EST_I_NUM);
			if (! Utils.isEmpty(sprop)) {
				setMaxLambdaEstIterNum(Integer.parseInt(sprop));
			}

			sprop = props.getProperty(AIRConstants.STRING_THREAD_NUM);
			if (! Utils.isEmpty(sprop)) {
				setThreadNum(Integer.parseInt(sprop));
			}

			sprop = props.getProperty(AIRConstants.STRING_IMPLEMENT_METHOD);
			if (! Utils.isEmpty(sprop)) {
				implementMethod = Integer.parseInt(sprop);
				if (implementMethod < IMPLEMENT_METHOD_GENERAL_SAMPLING ||
						implementMethod > IMPLEMENT_METHOD_SHORT_REVIEW_MAP) {
					throw new AIRException(ErrorMessage.ERROR_NO_IMPLEMENT_METHOD);
				}
			}

			sprop = props.getProperty(AIRConstants.STRING_DEBUG_CONVERGENE);
			if (Utils.isTrue(sprop)) {
				debugConvergence = true;
			} else {
				debugConvergence = false;
			}

			sprop = props.getProperty(AIRConstants.STRING_IS_RESTORE);
			if (Utils.isTrue(sprop)) {
				isRestore = true;
			} else {
				isRestore = false;
			}

			sprop = props.getProperty(AIRConstants.STRING_IS_OPTIMIZE_LAMBDA);
			if (Utils.isTrue(sprop)) {
				isOptimizeLambda = true;
			} else {
				isOptimizeLambda = false;
			}
		
			sprop = props.getProperty(AIRConstants.STRING_IS_USE_RATEBEER_DATA);
			if (Utils.isTrue(sprop)) {
				this.setUseRateBeerData(true);
			} else {
				this.setUseRateBeerData(false);
			}

			sprop = props.getProperty(AIRConstants.STRING_IS_GREED_SEARCH);
			if (Utils.isTrue(sprop)) {
				isGreedSearch = true;
			} else {
				isGreedSearch = false;
			}

			if (isGreedSearch) {
				sprop = props.getProperty(AIRConstants.STRING_IS_USE_MULTI_THREAD);
				if (Utils.isTrue(sprop)) {
					isUseMultiThread = true;
				} else {
					isUseMultiThread = false;
				}

				sprop = props.getProperty(AIRConstants.STRING_TOPIC_NUM_ARRAY);
				if (! Utils.isEmpty(sprop)) {
					setTopicNumArray(Utils.parseIntArray(sprop, AIRConstants.SPACER_SEMICOLON));
				}

				sprop = props.getProperty(AIRConstants.STRING_LAMBDA_ARRAY);
				if (! Utils.isEmpty(sprop)) {
					setLambdaArray(Utils.parseDoubleArray(sprop, AIRConstants.SPACER_SEMICOLON));
				}

				sprop = props.getProperty(AIRConstants.STRING_GAMMAS_ARRAY);
				setGammasArray(parseGammasArray(sprop));
			}

		} catch (NumberFormatException e) {
			throw new AIRException (e.toString());
		} catch (IOException e) {
			throw new AIRException (e.toString());
		} finally {
			Utils.cleanup(confReader);
		}
	}

	public boolean store(File outputFile) {
		if (outputFile != null) {
			BufferedWriter writer = null;
			try {
				writer = Utils.createBufferedWriter(outputFile);

				Utils.write(writer, toString(), true);

				return true;
			} catch (IOException e) {
				e.printStackTrace();
			} finally {
				Utils.cleanup(writer);
			}
		}

		return false;
	}

	@Override
	public ParamManager clone() {
		ParamManager paramManager = new ParamManager();

		paramManager.setTopicNum(topicNum);
		paramManager.setParticleNum(particleNum);
		paramManager.setTopWordNum(topWordNum);
		paramManager.setBurninIterNum(burninIterNum);
		paramManager.setEstimateIterNum(estimateIterNum);
		paramManager.setMaxAlphaEstIterNum(maxAlphaEstIterNum);
		paramManager.setMaxLambdaEstIterNum(maxLambdaEstIterNum);
		paramManager.setThreadNum(threadNum);
		paramManager.setTopicNumArray(topicNumArray);
		paramManager.setModelOutputPath(modelOutputPath);
		paramManager.setDataTrainFile(dataTrainFile);
		paramManager.setDataTestFile(dataTestFile);
		paramManager.setDataValidationFile(dataValidationFile);
		paramManager.setKeywordFile(keywordFile);
		paramManager.setGroundTruthFile(groundtruthScoreFile);
		paramManager.setGroundtruthTopicWordDistributionFile(groundtruthTopicWordDisFile);
		paramManager.setLambda(lambda);
		paramManager.setBetaInit(betaInit);
		paramManager.setLambdaArray(lambdaArray);
		paramManager.setGammas(gammas);
		paramManager.setGammasArray(gammasArray);
		paramManager.setDebugConvergence(debugConvergence);
		paramManager.setRestore(isRestore);
		paramManager.setOptimizeLambda(isOptimizeLambda);
		paramManager.setGreedSearch(isGreedSearch);
		paramManager.setUseMultiThread(isUseMultiThread);
		paramManager.setUseRateBeerData(isUseRatebeerData);
		paramManager.setRatingScaler(ratingScaler);
		paramManager.setImplementMethod(implementMethod);

		return paramManager;
	}

	private static String wrapGammasArray(double[][] gammasArray) {
		if (gammasArray == null || gammasArray.length == 0) return "";

		StringBuffer buffer = new StringBuffer();
		for (int i = 0; i < gammasArray.length; i ++) {
			for (int j = 0; j < gammasArray[i].length; j ++) {
				buffer.append(gammasArray[i][j]);
				if (j != gammasArray[i].length - 1) {
					buffer.append(AIRConstants.SPACER_COMMA);
				}
			}
			if (i != gammasArray.length - 1) {
				buffer.append(AIRConstants.SPACER_SEMICOLON);
			}
		}
		return buffer.toString();
	}

	private double[][] parseGammasArray(String content)
				throws NumberFormatException, AIRException {
		if (Utils.isEmpty(content)) return null;

		String[] elements        = content.split(AIRConstants.SPACER_SEMICOLON);
		ArrayList<double[]> list = new ArrayList<double[]>();
		for (String element : elements) {
			list.add(AIRUtils.parseGammas(element));
		}

		return list.toArray(new double[0][0]);
	}
}
