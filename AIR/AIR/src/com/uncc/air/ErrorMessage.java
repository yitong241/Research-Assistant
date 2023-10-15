package com.uncc.air;

/**
 * @author Huayu Li
 */
public class ErrorMessage {
	public static final String ERROR_NO_ARGS                    =
			"No argument specified.";
	public static final String ERROR_NO_TOPIC_NUM               =
			"No topic number specified.";
	public static final String ERROR_NO_TOP_WORD_NUM            =
			"No top word number specified.";
	public static final String ERROR_NO_BURNIN_ITER_NUM         =
			"No burnin iteration number specified.";
	public static final String ERROR_NO_EST_ITER_NUM            =
			"No estimate iteration number specified.";
	public static final String ERROR_NO_MAX_ALPHA_EST_ITER_NUM  =
			"No max alpha estimate iteration number specified.";
	public static final String ERROR_NO_MAX_LAMBDA_EST_ITER_NUM =
			"No max lambda estimate iteration number specified.";
	public static final String ERROR_NO_LAMBDA                  =
			"No lambda specified.";
	public static final String ERROR_NO_BETA                    =
			"No beta specified.";
	public static final String ERROR_NO_VALIDATION_FILE         =
			"No validation file specified.";
	public static final String ERROR_NO_TEST_FILE               =
			"No test file specified.";
	public static final String ERROR_NO_TRAIN_FILE              =
			"No train file specified.";
	public static final String ERROR_NO_DATA_INPUT_PATH         =
			"No data input path specified.";
	public static final String ERROR_NO_OUTPUT_PATH             =
			"No output path specified.";
	public static final String ERROR_NO_GAMMA                   =
			"No gammas specified or format is not correct.";
	public static final String ERROR_EMPTY_FILE                 =
			"Empty file.[%s]";

	public static final String ERROR_STORE_PARAMS               =
			"Failed to store parameters.";
	public static final String ERROR_STORE_TOP_WORD             =
			"Failed to store top words.";
	public static final String ERROR_STORE_SCORE                =
			"Failed to store score.";
	public static final String ERROR_STORE_PARAM_T              =
			"Failed to store parameter t.";
	public static final String ERROR_STORE_PARAM_THETA          =
			"Failed to store parameter theta.";
	public static final String ERROR_STORE_TOPIC_SEN_WORD_COUNT =
			"Failed to store topic_sentiment_word count.";
	public static final String ERROR_STORE_DOC_TOPIC_COUNT      =
			"Failed to store document topic count.";
	public static final String ERROR_STORE_DOC_TOPIC_SEN_COUNT  = 
			"Failed to store document_topic_sentiment count.";
	public static final String ERROR_STORE_DOC_SEN_COUNT        =
			"Failed to store document_sentiment count.";
	public static final String ERROR_STORE_TOPIC_SEN_COUNT      =
			"Failed to store topic sentiment count.";
	public static final String ERROR_STORE_TOPIC_FOR_WORD       =
			"Failed to store topic for each word.";
	public static final String ERROR_STORE_SEN_FOR_WORD         =
			"Failed to store sentiment for each word.";
	public static final String ERROR_STORE_PHI                  =
			"Failed to store param phi %s.";
	public static final String ERROR_STORE_GAMMA                =
			"Failed to store gamma.";
	public static final String ERROR_STORE_ALPHA                =
			"Failed to store alpha.";
	public static final String ERROR_STORE_ALPHA_INIT           =
			"Failded to store alpha_init.";
	public static final String ERROR_STORE_WORD_PROB            =
			"Failed to store word_prob.";
	public static final String ERROR_STORE_WORD_PROB_INIT       =
			"Failed to store word_prob_init.";
	
	public static final String ERROR_EMPTY_GROUNDTRUTH_FILE     =
			"Groundtruth score file doesn't exist. [ %s ]";
	public static final String ERROR_EMPTY_KEYWORD_FILE         =
			"Keyword file doesn't exist. [ %s ]";
	public static final String ERROR_EMPTY_GROUNDTRUTH_TW_FILE  =
			"Groundtruth word distribution file doesn't exist. [ %s ]";
	public static final String ERROR_NOT_MATCH_KEYWORD          =
			"Keyword topic number is not matched: keywordTopicNum = %s, topicNum = %s";

	public static final String ERROR_NO_IMPLEMENT_METHOD        =
			"No specified implement method.";
	
}
