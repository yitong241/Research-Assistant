package com.uncc.air;

import com.lhy.tool.ToolException;
import com.lhy.tool.util.Utils;
import com.uncc.air.general.AIR_GS;
import com.uncc.air.general.AIR_VB;
import com.uncc.air.general.MultiThread;
import com.uncc.air.shortreview.AIRS_MAP;
import com.uncc.topicmodel.TopicModelException;

/**
 * @author Huayu Li
 */
public class Main {
	
	public Main(ParamManager paramManager) {

		if (paramManager == null) {
			paramManager = AIRConstants.CONFIG_MANAGER;
		}

		if (paramManager.getImplementMethod() == ParamManager
				.IMPLEMENT_METHOD_GENERAL_SAMPLING &&
					paramManager.isGreedSearch()) {
			greedSearch(paramManager);
		} else {
			buildModel(paramManager);
		}
	}

	private void buildModel(ParamManager paramManager) {
		try {
			Estimator estimator = null;
			switch (paramManager.getImplementMethod()) {
				case ParamManager.IMPLEMENT_METHOD_GENERAL_SAMPLING:
					estimator = new AIR_GS(paramManager, null);
					break;
				case ParamManager.IMPLEMENT_METHOD_GENERAL_VB:
					estimator = new AIR_VB(paramManager, null);
					break;
				case ParamManager.IMPLEMENT_METHOD_SHORT_REVIEW_MAP:
					estimator = new AIRS_MAP(paramManager);
					break;
				default:
					// never reaches;
					Utils.err(ErrorMessage.ERROR_NO_IMPLEMENT_METHOD);
					return;
			}
			estimator.estimate();
		} catch (ToolException e) {
			e.printStackTrace();
		} catch (AIRException e) {
			e.printStackTrace();
		} catch (TopicModelException e) {
			e.printStackTrace();
		}
	}

	private void greedSearch(ParamManager paramManager) {
		if (paramManager.isUseMultiThread()) {
			try {
				new MultiThread(paramManager).run();
			} catch (AIRException e) {
				e.printStackTrace();
			}
		} else {
			if (Utils.isEmpty(paramManager.getTopicNumArray()) ||
					Utils.isEmpty(paramManager.getLambdaArray()) ||
					Utils.isEmpty(paramManager.getGammasArray())) {
				Utils.err("No parameters specified.");
				return;
			}
			try {
			for (int topicNum : paramManager.getTopicNumArray()) {
				for (double lambda : paramManager.getLambdaArray()) {
					for (double[] gammas : paramManager.getGammasArray()) {
						ParamManager param = paramManager.clone();
						param.setTopicNum(topicNum);
						param.setLambda(lambda);
						param.setGammas(gammas);
						AIR_GS estimator = new AIR_GS(param, null);
						estimator.estimate();
						
					}
				}
			}
			} catch (ToolException e) {
				e.printStackTrace();
			} catch (AIRException e) {
				e.printStackTrace();
			} catch (TopicModelException e) {
				e.printStackTrace();
			}
		}
	}

	public static void main(String[] args) {
		new Main(null);
	}

}
