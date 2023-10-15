package com.uncc.air;

import java.io.File;

import com.lhy.tool.util.Utils;
import com.uncc.air.ParamManager;
import com.uncc.air.util.AIRUtils;

/**
 * @author Huayu Li
 */
public final class AIRConstants {
	private static final String FILE_NAME_AIR         = "AIR";
	private static final String FILE_NAME_OUTPUT      = "output";
	private static final File PATH_CURRENT_RUNNING    =
				new File(System.getProperty("user.dir"));

	public static final String FILE_NAME_MODEL        = "model";
	public static final String FILE_NAME_CONFIGURE    = "configure";
	public static final String FILE_NAME_REV_TRAIN    = "reviews_train";
	public static final String FILE_NAME_REV_VALID    = "reviews_validation";
	public static final String FILE_NAME_REV_TEST     = "reviews_test";
	public static final String FILE_NAME_DIC_MAP      = "dictionary_map";
	public static final String FILE_NAME_PARAM        = "parameters";
	public static final String FILE_NAME_TOP_WORD     = "top_words.csv";
	public static final String FILE_NAME_SCORE        = "score";
	public static final String FILE_NAME_PARAM_T      = "param_t";
	public static final String FILE_NAME_PARAM_THETA  = "param_theta";
	public static final String FILE_NAME_PARAM_PHI    = "param_phi";
	public static final String FILE_NAME_TOP_SEN_W    = "topic_sentiment_word";
	public static final String FILE_NAME_DOC_TOPIC    = "document_topic";
	public static final String FILE_NAME_DOC_TOP_SEN  = "document_topic_sentiment";
	public static final String FILE_NAME_DOC_SEN      = "document_sentiment";
	public static final String FILE_NAME_WORD_TOPIC   = "word_topic";
	public static final String FILE_NAME_WORD_SEN     = "word_sentiment";
	public static final String FILE_NAME_ALPHA_EST    = "alpha_estimate";
	public static final String FILE_NAME_LAMBDA_EST   = "lambda_estimate";
	public static final String FILE_NAME_PERPLEXITY   = "perplexity";
	public static final String FILE_NAME_REVIEW_STOP_WORD_TABLE   = "review_stop_word_table";
	public static final String FILE_NAME_REVIEW_STEM_WORD_TABLE   = "review_stem_word_table";
	public static final String FILE_NAME_REVIEW_UPPER_LOWER_TABLE = "review_upper_lower_table"; 
	public static final String FILE_NAME_REVIEW_OTHER_LANGUAGE    = "review_other_language";
	public static final String FILE_NAME_REVIEW_OUTLIER_WORD      = "review_outlier_word";

	public static final String FILE_NAME_WORD_PROB      = "WordProb";
	
	public static final String CHARSET_NAME                = "utf-8";
	public static final String LINE_SEPARATOR              = System.getProperty("line.separator", "\n");
	public static final String SPACER_COMMA                = ",";
	public static final String SPACER_TAB                  = "\t";
	public static final String SPACER_SEMICOLON            = ";";
	public static final String SPACER_SPACE                = " ";
	public static final String STRING_DEBUG_CONVERGENE     = "DebugConvergence";
	public static final String STRING_BETA_INIT            = "BetaInit";
	public static final String STRING_GAMMAS               = "Gammas";
	public static final String STRING_GAMMAS_ARRAY         = "GammasArray";
	public static final String STRING_ALPHAS               = "Alphas";
	public static final String STRING_ALPHA_INIT           = "AlphaInit";
	public static final String STRING_LAMBDA               = "Lambda";
	public static final String STRING_LAMBDA_ARRAY         = "LambdaArray";
	public static final String STRING_TOPIC_NUM            = "TopicNum";
	public static final String STRING_TOPIC_NUM_ARRAY      = "TopicNumArray";
	public static final String STRING_BURNIN_ITER_NUM      = "BurninIterNum";
	public static final String STRING_BURNIN_ITER          = "BurninIter";
	public static final String STRING_EST_ITER_NUM         = "EstimateIterNum";
	public static final String STRING_MAX_ALPHA_EST_I_NUM  = "MaxAlphaEstimateIterNum";
	public static final String STRING_MAX_LAMBDA_EST_I_NUM = "maxLambdaEstimateIterNum";
	public static final String STRING_ALPHA_EST_ITER       = "AlphaEstimateIter";
	public static final String STRING_TOP_WORD_NUM         = "TopWordNum";
	public static final String STRING_DIC_NUM              = "DictionaryNum";
	public static final String STRING_DOCUMENT_NUM         = "ReviewNum";
	public static final String STRING_TOKEN_NUM            = "TokenNum";
	public static final String STRING_PARTICLE_NUM         = "ParticleNum";
	public static final String STRING_MODEL_OUTPUT_PATH    = "ModelOutputPath";
	public static final String STRING_DATA_INPUT_PATH      = "DataInputPath";
	public static final String STRING_DATA_TRAIN           = "TrainFile";
	public static final String STRING_DATA_VALIDATION      = "ValidationFile";
	public static final String STRING_DATA_TEST            = "TestFile";
	public static final String STRING_KEYWORD_FILE         = "KeywordFile";
	public static final String STRING_GROUNDTRUTH_FILE     = "GroundtruthScoreFile";
	public static final String STRING_TW_TOPIC_WORD_FILE   = "GroundtruthTopicWordDistributionFile";
	public static final String STRING_IS_RESTORE           = "IsRestore";
	public static final String STRING_IS_OPTIMIZE_LAMBDA   = "IsOptimizeLambda";
	public static final String STRING_IS_GREED_SEARCH      = "EnableGreedSearch";
	public static final String STRING_IS_USE_MULTI_THREAD  = "EnableUseMultiThread";
	public static final String STRING_THREAD_NUM           = "ThreadNum";
	public static final String STRING_MAX_EM_ITER_NUM      = "MaxEMIterNum";
	public static final String STRING_EM_THRESHOLD         = "EMThreshold";
	public static final String STRING_E_THRESHOLD          = "EThreshold";
	public static final String STRING_ALPHA_THRESHOLD      = "AlphaThreshold";
	public static final String STRING_IS_USE_RATEBEER_DATA = "IsUseRateBeerData";
	public static final String STRING_IMPLEMENT_METHOD     = "ImplementMethod";
	public static final char DELIMITER                     = ',';
	public static final double MAX_RATING                  = 5.0;
	public static final double MIN_RATING                  = 1.0;
	public static final int DEFAULT_MISS_ASPECT_RATING     = -1;

	public static final String STRING_IMPLEMENT_METHOD_NOTE =
					"# Implement Method\r\n" +
					"# General_Model_Via_Sampling=0\r\n" +
					"# General_Model_Via_MAP=1\r\n"       +
					"# General_Model_Via_VB=2\r\n"      +
					"# Short_Rev_Model_Via_MAP=3";

	private static final File DEFAULT_PATH_OUTPUT            = 
				new File((PATH_CURRENT_RUNNING.getParentFile().exists() ?
					new File(PATH_CURRENT_RUNNING.getParentFile(), FILE_NAME_OUTPUT) :
					new File(PATH_CURRENT_RUNNING, FILE_NAME_OUTPUT)),
						FILE_NAME_AIR);
	private static final File DEFAULT_FILE_KEYWORD           = null;
	private static final File DEFAULT_FILE_GROUNDTRUTH_SCORE = null;
	private static final File DEFAULT_FILE_GROUNDTRUTH_TW    = null;
	private static final boolean DEFAULT_DEBUG_CONVERGENCE   = false;
	private static final boolean DEFAULT_IS_RESTORE          = false;
	private static final boolean DEFAULT_IS_OPTIMIZE_LAMBDA  = false;
	private static final boolean DEFAULT_IS_GREED_SEARCH     = false;
	private static final boolean DEFAULT_IS_USE_MULTI_THREAD = false;
	private static final double DEFAULT_LAMBDA               = 1.0;
	private static final double DEFAULT_BETA_INIT            = 0.01;
	private static final double[] DEFAULT_GAMMAS             = {2.5, 1.0};
	private static final int DEFAULT_TOPIC_NUM               = 10;
	private static final int DEFAULT_PARTICLE_NUM            = 100;
	private static final int DEFAULT_TOP_WORD_NUM            = 20;
	private static final int DEFAULT_BURNIN_ITER_NUM         = 2000;
	private static final int DEFAULT_ESTIMATE_ITER_NUM       = 200;
	private static final int DEFAULT_MAX_ALPHA_EST_ITER_NUM  = 5;
	private static final int DEFAULT_MAX_LAMBDA_EST_ITER_NUM = 1;
	private static final int DEFAULT_THREAD_NUM              = 4;

	public static File PATH_OUTPUT            = DEFAULT_PATH_OUTPUT;
	public static File FILE_CONFIGURE         = null;
	public static ParamManager CONFIG_MANAGER = null;
				
	static {
		String sprop = System.getProperty("PATH_CONFIGURE");
		if (! Utils.isEmpty(sprop)) {
			FILE_CONFIGURE = new File(new File(sprop), FILE_NAME_CONFIGURE);
		} else {
			FILE_CONFIGURE = new File(PATH_CURRENT_RUNNING, FILE_NAME_CONFIGURE);
		}

		initConfig();
	}


	private static void initConfig() {
		/*if (System.getProperty("PATH_OUTPUT") != null) {
			PATH_OUTPUT = new File(System.getProperty("PATH_OUTPUT"));
		}*/
		if (! FILE_CONFIGURE.exists()) {
			CONFIG_MANAGER = new ParamManager();

			CONFIG_MANAGER.setModelOutputPath(PATH_OUTPUT);
			CONFIG_MANAGER.setDataTrainFile("(path)/TrainFile");
			CONFIG_MANAGER.setDataValidationFile("(path)/ValidationFile");
			CONFIG_MANAGER.setDataTestFile("(path)/TestFile");
			CONFIG_MANAGER.setTopicNum(DEFAULT_TOPIC_NUM);
			CONFIG_MANAGER.setLambda(DEFAULT_LAMBDA);
			CONFIG_MANAGER.setGammas(DEFAULT_GAMMAS);
			CONFIG_MANAGER.setBetaInit(DEFAULT_BETA_INIT);
			CONFIG_MANAGER.setTopWordNum(DEFAULT_TOP_WORD_NUM);
			CONFIG_MANAGER.setBurninIterNum(DEFAULT_BURNIN_ITER_NUM);
			CONFIG_MANAGER.setEstimateIterNum(DEFAULT_ESTIMATE_ITER_NUM);
			CONFIG_MANAGER.setMaxAlphaEstIterNum(DEFAULT_MAX_ALPHA_EST_ITER_NUM);
			CONFIG_MANAGER.setMaxLambdaEstIterNum(DEFAULT_MAX_LAMBDA_EST_ITER_NUM);
			CONFIG_MANAGER.setKeywordFile(DEFAULT_FILE_KEYWORD);
			CONFIG_MANAGER.setGroundTruthFile(DEFAULT_FILE_GROUNDTRUTH_SCORE);
			CONFIG_MANAGER.setGroundtruthTopicWordDistributionFile(DEFAULT_FILE_GROUNDTRUTH_TW);
			CONFIG_MANAGER.setParticleNum(DEFAULT_PARTICLE_NUM);
			CONFIG_MANAGER.setDebugConvergence(DEFAULT_DEBUG_CONVERGENCE);
			CONFIG_MANAGER.setRestore(DEFAULT_IS_RESTORE);
			CONFIG_MANAGER.setOptimizeLambda(DEFAULT_IS_OPTIMIZE_LAMBDA);
			CONFIG_MANAGER.setGreedSearch(DEFAULT_IS_GREED_SEARCH);
			CONFIG_MANAGER.setUseMultiThread(DEFAULT_IS_USE_MULTI_THREAD);
			CONFIG_MANAGER.setThreadNum(DEFAULT_THREAD_NUM);
			CONFIG_MANAGER.setTopicNumArray(null);
			CONFIG_MANAGER.setLambdaArray(null);
			CONFIG_MANAGER.setGammasArray(null);
			CONFIG_MANAGER.setUseRateBeerData(false);

			if (! CONFIG_MANAGER.store(FILE_CONFIGURE)) {
				Utils.err("Failed to store configure file.");
			}
		} else {
			CONFIG_MANAGER = new ParamManager();
			try {
				CONFIG_MANAGER.load(FILE_CONFIGURE);

				if (CONFIG_MANAGER.getModelOutputPath() != null) {
					PATH_OUTPUT = CONFIG_MANAGER.getModelOutputPath();
				} else {
					CONFIG_MANAGER.setModelOutputPath(PATH_OUTPUT);
				}
			} catch (AIRException e) {
				e.printStackTrace();
				System.exit(-1);
			}
		}

		// override some properties
		try {
			String sprop = System.getProperty("TOPIC_NUM");
			if (sprop != null) {
				CONFIG_MANAGER.setTopicNum(Integer.parseInt(sprop));
			}

			sprop = System.getProperty("LAMBDA");
			if (sprop != null) {
				CONFIG_MANAGER.setLambda(AIRUtils.getArithmeticResult(sprop));
			}

			sprop = System.getProperty("GAMMAS");
			if (sprop != null) {
				CONFIG_MANAGER.setGammas(AIRUtils.parseGammas(sprop));
			}
		} catch (NumberFormatException e) {
			e.printStackTrace();
			System.exit(-1);
			
		} catch (AIRException e) {
			e.printStackTrace();
			System.exit(-1);
		}

		Utils.println("================== Configure ==================");
		Utils.println(CONFIG_MANAGER);
		Utils.println("===============================================");
	}
}
