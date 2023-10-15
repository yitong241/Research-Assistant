package com.uncc.air.util;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.StringTokenizer;

import org.apache.commons.math3.special.Gamma;
import org.apache.commons.math3.util.FastMath;

import com.uncc.air.AIRException;
import com.uncc.lda.est.em.LDANewtonRaphson;
import com.lhy.tool.ToolException;
import com.lhy.tool.util.Utils;

/**
 * @author Huayu Li
 */
public class DirichletParamsEstimator {

	@SuppressWarnings("unused")
	private static final double digmma_1 = -0.577215664901532860606512090082;
	@SuppressWarnings("unused")
	private static final double S        = 1.0e-5;
	@SuppressWarnings("unused")
	private static final double C        = 8.0 * 5.0;

	/*
	 * K is the length of the element of probabilities
	 */
	public static double[] estimate(LinkedList<double[]> probabilities,
			int K, double errorThreshold, File outputPath,
					int debug) throws AIRException{
		if (! check(probabilities, K)) {
			throw new AIRException ("Estimating input format is not correct.");
		}

		double[][] res = new double[probabilities.size()][K];
		for (int i = 0; i < probabilities.size(); i ++) {
			for (int j = 0; j < probabilities.get(i).length; j ++) {
				res[i][j] = probabilities.get(i)[j];
			}
		}

		return estimate(res, K, errorThreshold, outputPath, debug);
		//return estimateViaHessian(res, K, errorThreshold, debug);
	}

	/*
	 * K is the length of the element of probabilities
	 */
	public static double[] estimate(double[][] probabilities,
				int K, double errorThreshold,
				File outputPath, int debug) throws AIRException{
		if (probabilities == null || probabilities[0].length != K) {
			throw new AIRException ("Estimating input format is not correct.");
		}

		List<double[]> data = new LinkedList<double[]>();
		double[] aveLogPs   = new double[K];
		double[] alpha      = new double[K];

		for (int i = 0; i < aveLogPs.length; i ++) {
			aveLogPs[i] = averageLogP(probabilities, i);
		}
		// init
		for (int i = 0; i < alpha.length; i ++) {
			alpha[i] = Math.random();
		}
		
		optimize(alpha, aveLogPs, errorThreshold, data, debug);

		double[][] dataArray = new double[data.size()][2];
		for (int i = 0; i < data.size(); i ++) {
			dataArray[i][0] = data.get(i)[0];
			dataArray[i][1] = data.get(i)[1];
			
		}
		if (outputPath != null) {
			AIRChartFactory.saveXYLineChart("Estimate Dirichlet Params, Object Function",
				"Iteration Number", "Object Fuction Value", dataArray,
				new File(outputPath, "dir_params_iter.jpg"),
					800, 400);
		}

		return alpha;
	}

	public static double[] estimateViaHessian(double[][] probabilities,
					int K, double errorThreshold, int debug)
							throws AIRException{
		if (probabilities == null || probabilities[0].length != K) {
			throw new AIRException ("Estimating input format is not correct.");
		}

		double[] LogPs = new double[K];
		double[] alpha = new double[K];

		for (int i = 0; i < LogPs.length; i ++) {
			LogPs[i] = averageLogP(probabilities, i) * probabilities.length;
		}
		// init
		for (int i = 0; i < alpha.length; i ++) {
			alpha[i] = 1.0 / alpha.length;//Math.random();
		}
	
		try {
			new LDANewtonRaphson().estimate(alpha, LogPs,
					probabilities.length, errorThreshold, debug);
		} catch (ToolException e) {
			e.printStackTrace();
		}

		return alpha;
	}

	public static void optimize(double[] alpha, double[] constants,
			double errorThreshold, List<double[]> data, int debug)
						throws AIRException {
		int iter           = 0;
		double alphaError  = Double.MAX_VALUE;
		double fValueError = Double.MAX_VALUE;
		double fValue      = objectFunction(constants, alpha, alpha.length);

		if (debug >= 1) {
			Utils.println("Init fValue = " + fValue);
		}
		do {
			double[] alpha_old = Arrays.copyOf(alpha, alpha.length);
			double fValueOld   = fValue;

			for (int k = 0 ; k < alpha.length; k ++) {
				double y = Gamma.digamma(sum(alpha)) + constants[k];
				alpha[k] = solveDigamma(y);
				//Utils.println(alpha[0] + ","+ alpha[1]);
			}
			fValue = objectFunction(constants, alpha, alpha.length);

			if (data != null) {
				data.add(new double[]{iter, fValue});
			}

			try {
				alphaError  = Utils.sumOfDiffAbs(alpha, alpha_old);
				fValueError = (fValue - fValueOld) / fValueOld;
			} catch (ToolException e) {
				throw new AIRException(e.toString());
			}

			if (debug >= 2) {
				Utils.println(String.format("Iter = %s, argError = %s, fValueError = %s, fValue = %s, args = %s",
						iter, alphaError, fValueError,
						fValue, Utils.convertToString(alpha, ",")));
			}
			iter ++;
		} while (alphaError > errorThreshold ||
				fValueError > errorThreshold || iter <= 2);

		if (debug >= 1) {
			Utils.println(String.format("Error = %s, args = %s", alphaError,
					Utils.convertToString(alpha, ",")));
		}
	}

	private static double objectFunction(double[] aveLogPs,
					double[] alpha_new,  int K) {
		double[] logGamma  = new double[K];
		double[] alphaLogP = new double[K];
		for (int i = 0; i < K; i ++) {
			logGamma[i]  = Gamma.logGamma(alpha_new[i]);
			alphaLogP[i] = (alpha_new[i] - 1) * aveLogPs[i];
		}

		return Gamma.logGamma(sum(alpha_new)) - sum(logGamma) + sum(alphaLogP);
	}

	private static double averageLogP(double[][] probabilities,
								int k) {
		double sum = 0.0;
		for (int i = 0; i < probabilities.length; i ++) {
			sum += Math.log(probabilities[i][k]);
		}
		return sum / probabilities.length;
	}

	private static boolean check(LinkedList<double[]> probabilities, int K) {
		if (! Utils.isEmpty(probabilities) && K > 0) {
			for (double[] prob : probabilities) {
				if (prob == null || prob.length < K) return false;
			}
			return true;
		}

		return false;
	}

	private static double sum(double[] array) {
		double sum = 0.0;
		if (array != null) {
			for (double element : array) {
				sum += element;
			}
		}
		return sum;
	}

	/*
	 * Solve the function digamma(x) = y when given y.
	 */
	private static double solveDigamma(double y) {
		// solve digamma(x) = y without any constraints
		/*int iterMax = 10000;
		double e    = 1.0e-14;
		double x    = 1;
		for (int iter = 0; iter < iterMax && (Math.abs(Gamma.digamma(x) - y) > e); iter ++) {
			x = x - (Gamma.digamma(x) - y) / Gamma.trigamma(x);
		}

		return x;*/

		// solve digamma(x) = y with constraints that y > 0
		int iterMax    = 10000;
		double e       = 1.0e-14;
		double mappedX = 1;
		double x       = FastMath.exp(1);
		for (int iter = 0; iter < iterMax && (Math.abs(Gamma.digamma(x) - y) > e); iter ++) {
			mappedX -= (Gamma.digamma(x) - y) /
					( Gamma.trigamma(x) * FastMath.exp(mappedX));
			x        = FastMath.exp(mappedX);
		}

		return x;

		// proximate way !!
		/*if (y >= -2.22) {
			return FastMath.exp(y) + 0.5;
		} else {
			return -1.0 / (y + Gamma.digamma(1));
		}*/
	}

	/*
	 * digamma(1) = -0.5772156677920679
	 */
	/*public static double digamma(double x) {
		if (x > 0 && x <= S) {
			return digmma_1 - 1.0 / x;
		}
		if (x >= C) {
			double x_x = 1.0 / (x * x);
			//	      1       1        1         1
			// log(x) -  --- - ------ + ------- - -------
			//           2 x   12 x^2   120 x^4   252 x^6
			return Math.log(x) - 1.0 / (2.0 * x) - x_x *( 1.0 / 12.0 - x_x  * ( 1.0 / 120.0 + x_x * 1.0 / 252.0));
		}

		return digamma(x + 1) - 1.0 / x;
	}

	public static double trigamma(double x) {
		double inv2 = 1.0 / (x * x);

		if (x > 0 && x <= S) {
			return  inv2;
		}
		if (x >= C) {
			double inv3 = 1.0 / (x * x * x);
			//  1       1       1         1         1
			// --- +  ----- + ------ - ------- + -------
			//  x     2 x^2   6 x^3    30 x^5     42 x^7
			return 1.0 / x + 0.5 * inv2 + inv3 * (1.0 / 6.0 - inv2 *(1.0 / 30.0 +  inv2 * (1.0 / 42)));
		}

		return trigamma(1 + x) + inv2;
	}*/

	private static LinkedList<double[]> getProbs(File file, int lineNum) {
		if (!Utils.exists(file)) return null;

		BufferedReader reader = null;
		try {
			reader                        = Utils.createBufferedReader(file);
			String line                   = null;
			int count                     = 0;
			LinkedList<double[]> probList = new LinkedList<double[]>();
			while ((line = reader.readLine()) != null) {
				count ++;
				StringTokenizer tokenizer = new StringTokenizer(line);
				int index             = 0;
				double sentimentCount = 0;
				double totalCount     = 0;
				while (tokenizer.hasMoreElements()) {
					String token = tokenizer.nextToken();
					index ++;
					if (index < 2) continue;

					if (token.indexOf(":") != -1) {
						sentimentCount ++;
						try {
							Integer.parseInt(token.split(":")[1]);
						} catch (NumberFormatException e) {
							Utils.err("Line: " + count);
							e.printStackTrace();
						}
					}
					totalCount ++;
				}

				double[] probs = new double[2];
				probs[1] = sentimentCount / totalCount;
				probs[0] = 1 - probs[1];
				probList.add(probs);

				// print
				//Utils.println(probs[0] + "," + probs[1]);

				if (count >= lineNum) break;
			}

			return probList;
		} catch (IOException e) {
			e.printStackTrace();
		} finally {
			Utils.cleanup(reader);
		}

		return null;
	}

	@SuppressWarnings("unused")
	private static void drawDistribution(LinkedList<double[]> probs,
							File outputPath) {
		if (Utils.isEmpty(probs)) return;

		HashMap<Double, Integer> countMap = new HashMap<Double, Integer>();
		for (double[] ps : probs) {
			double p = ((double)Math.round(ps[0] * 10)) / 10.0;//((int)(ps[0] * 10)) / 10.0;
			if (countMap.containsKey(p)) {
				countMap.put(p, countMap.get(p) + 1);
			} else {
				countMap.put(p, 1);
			}
		}

		String[] domains = new String[countMap.size()];
		Integer[] values = new Integer[countMap.size()];

		toDomainAndValue(countMap, domains, values);

		File imageFile = new File(outputPath, "Estimate_Dirichlet_Distribution.jpeg");
		AIRChartFactory.saveHistogram(String.format("Estimate Dirichlet Distribution Histogram : Sample %s", probs.size()),
					"Probability", "Count", domains, values,
						imageFile, 500, 500);
	}

	private static void toDomainAndValue(HashMap<Double, Integer> map,
			String[] domains, Integer[] values) {
		List<Object[]> list = new ArrayList<Object[]>();
		for (Double key : map.keySet()) {
			list.add(new Object[]{key, map.get(key)});
		}
		Collections.sort(list, new Comparator<Object[]>() {
			@Override
			public int compare(Object[] obj1, Object[] obj2) {
				return ((Double)obj1[0]).compareTo((Double)obj2[0]);
			}
		});
		for (int i = 0; i < list.size(); i ++) {
			domains[i] = String.format("%.1f", list.get(i)[0]);
			values[i]  = (Integer)list.get(i)[1];
		}
	}
}
