package com.uncc.air.general;

import com.lhy.tool.ToolException;
import com.lhy.tool.util.Utils;
import com.uncc.air.AIRException;
import com.uncc.air.ParamManager;
import com.uncc.topicmodel.TopicModelException;

/**
 * @author Huayu Li
 */
public class MultiThread {
	private static final String THREAD_NAME  = "[Thread %s]";

	private ParamManager paramManager        = null;
	private Resource     resource            = null;
	private Object       LOCK                = new Object();
	private int currentIndex                 = 0;

	public MultiThread(ParamManager paramManager) throws AIRException {
		if (paramManager == null) {
			throw new AIRException("No parameters specified.");
		}

		this.paramManager = paramManager;
		this.resource     = new Resource();
	}

	public void run() {
		RunEstimator[] threads = new RunEstimator[paramManager.getThreadNum()];
		boolean[] status       = new boolean[threads.length];
		int checkCount         = 0;

		for (int i = 0; i < threads.length; i ++) {
			threads[i] = new RunEstimator(String.format(THREAD_NAME, i));
			threads[i].start();
			status[i]  = true;

			Utils.sleep(1 * 1000);
		}

		while (true) {
			for (int i = 0; i < threads.length; i ++) {
				if (! threads[i].isAlive() && status[i]) {
					status[i] = false;
					checkCount ++;
				}
			}
			if (checkCount == threads.length) break;
			
			Utils.sleep(5 * 1000);
		}

		StringBuffer buffer = new StringBuffer();
		for (int i = 0; i < threads.length; i ++) {
			if (threads[i].getIndex() < resource.getSize()) {
				buffer.append(String.format("%s failed to complete."))
					.append(com.lhy.tool.Constants.LINE_SEPARATOR);
			}
		}
		if (buffer.length() == 0) {
			Utils.println(String.format("%s threads completes successfully.", threads.length));
		} else {
			Utils.err(buffer);
		}
	}

	private class RunEstimator extends Thread {
		@SuppressWarnings("unused")
		private boolean inProgress = false;
		private ParamManager param = null;
		private int index          = -1;
		private int topicNum;
		private double lambda;
		private double[] gammas;

		public RunEstimator(String threadName) {
			if (! Utils.isEmpty(threadName)) {
				setName(threadName);
			}
		}

		@Override
		public void run() {
			while (true) {
				inProgress = true;

				synchronized (LOCK) {
					if ((index = currentIndex) >= resource.getSize()) break;

					topicNum = resource.getTopicNum(currentIndex);
					lambda   = resource.getLambda(currentIndex);
					gammas   = resource.getGammas(currentIndex);

					param = paramManager.clone();
					param.setTopicNum(topicNum);
					param.setLambda(lambda);
					param.setGammas(gammas);

					currentIndex ++;
				}

				try {
					AIR_GS estimator = new AIR_GS(param, getName());
					estimator.estimate();
				} catch (ToolException e) {
					e.printStackTrace();
				} catch (AIRException e) {
					e.printStackTrace();
				} catch (TopicModelException e) {
					e.printStackTrace();
				}

				inProgress = false;
			}
			inProgress = false;
		}

		public int getIndex() {
			return index;
		}
	}

	private class Resource {
		private int[] topicNumArray    = null;
		private double[] lambdaArray   = null;
		private double[][] gammasArray = null;

		public Resource() throws AIRException {
			if (Utils.isEmpty(paramManager.getTopicNumArray()) ||
					Utils.isEmpty(paramManager.getLambdaArray()) ||
					Utils.isEmpty(paramManager.getGammasArray())) {
				throw new AIRException ("No parameters specified.");
			}

			int size      = paramManager.getTopicNumArray().length *
				 		paramManager.getLambdaArray().length *
						paramManager.getGammasArray().length;
			topicNumArray = new int[size];
			lambdaArray   = new double[size];
			gammasArray   = new double[size][paramManager.getGammasArray()[0].length];

			Utils.println("Total possible combination is " + size);

			int index = 0;
			for (int topicNum : paramManager.getTopicNumArray()) {
				for (double lambda : paramManager.getLambdaArray()) {
					for (double[] gammas : paramManager.getGammasArray()) {
						topicNumArray[index] = topicNum;
						lambdaArray[index]   = lambda;
						gammasArray[index]   = gammas;
						Utils.println("topicNum = " + topicNumArray[index] +
								"; lambda = " + lambdaArray[index] +
								"; gammas = " + Utils.convertToString(gammasArray[index], ","));
						index ++;
					}
				}
			}
		}

		public int getTopicNum(int index) {
			return topicNumArray[index];
		}

		public double getLambda(int index) {
			return lambdaArray[index];
		}
	
		public double[] getGammas(int index) {
			return gammasArray[index];
		}
	
		public int getSize() {
			return topicNumArray.length;
		}
	}
}
