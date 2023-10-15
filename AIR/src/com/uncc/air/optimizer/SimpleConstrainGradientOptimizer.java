package com.uncc.air.optimizer;

import java.util.Arrays;

import com.lhy.tool.optimizer.LBFGS;

import org.apache.commons.math3.util.FastMath;

import com.lhy.tool.ToolException;
import com.lhy.tool.util.Utils;
import com.uncc.air.AIRException;

/**
 * Object function min f(xi) where sum(xi) = 1 & 0=< xi <=1
 * @author Huayu Li
 */
public abstract class SimpleConstrainGradientOptimizer  {
	protected double[] args       = null;
	protected double[] mappedArgs   = null;
	private double errorThreshold = 0.0;
	private double argTolerance   = 0.0;

	public SimpleConstrainGradientOptimizer (double[] args,
				double errorThreshold, double argTolerance)
							throws AIRException {
		if (Utils.isEmpty(args) || errorThreshold < 0.0) {
			throw new AIRException("Illegal arguments.");
		}

		this.args           = args;
		this.mappedArgs     = new double[args.length];
		this.errorThreshold = errorThreshold;
		this.argTolerance   = argTolerance;
	}

	/*
	 * objectFunction is a 1-Dimension array where the first element is
	 * the value of object function.
	 * 
	 * If objectGradients is not null and objectFunction is null,
	 * it requires to calculate the object function's gradients.
	 * 
	 * If objectGradients is null and objectFunction is not null,
	 * it requires to calculate the value of the object function.
	 * 
	 * If both objectGradients and objectFunction are not null,
	 * it requires to calculate the value of gradients and function.
	 */
	public abstract void calculateObjectGradients(double[] objectGradients,
						double[] objectFunction);

	public boolean optimize(int debug) throws ToolException, AIRException {
		return optimizeViaLBGS(debug);
	}

	private boolean optimizeViaLBGS(int debug) throws AIRException, ToolException {
		initMappedArgs();
		double argsError   = Double.MAX_VALUE;
		double funcError   = Double.MAX_VALUE;
		double[] funcRes   = new double[1];
		double[] gradients = new double[args.length];
		int[] iflag        = {0};
		int[] iprint       = {-1, 0};
		int m              = 5;
		int iter           = 0;
		double[] diags     = new double[mappedArgs.length];

		calculateMappedGradients(gradients, funcRes);

		try {
			do {
				double[] argsOld   = Arrays.copyOf(args, args.length);
				double funcResOld  = funcRes[0];
				

				LBFGS.lbfgs(mappedArgs.length , m , mappedArgs , funcResOld,
						gradients , false , diags , iprint , argTolerance, 1e-20 , iflag );

				updateArgs();
				calculateMappedGradients(gradients, funcRes);

				argsError = Utils.sumOfDiffAbs(args, argsOld);
				funcError = (funcResOld - funcRes[0]) / Math.abs(funcResOld);

				if (debug >= 1) {
					Utils.println("iter = " + iter + "; iflag = " + iflag[0] +
						"; funRes = " + funcRes[0] + "; argsError = " +
						argsError + "; funcError = " + funcError +
							"; args=" + Utils.convertToString(args, ","));
				}
	
				iter ++;
			} while ( iflag[0] != 0 || argsError > errorThreshold ||
					funcError > errorThreshold);
	
			if (debug >= 2) {
				Utils.println("iter = " + iter + "; iflag = " + iflag[0] +
					"; funRes = " + funcRes[0] + "; argsError = " +
					argsError + "; funcError = " + funcError +
						"; args=" + Utils.convertToString(args, ","));
			}

			if (iflag[0] == 0) {
				return true;
			} else
			if (argsError <= errorThreshold ||
					funcError <= errorThreshold) {
				/*
				 *  if the error of arguments or function is during
				 *  the tolerance, we also consider it converges.
				 */
				return true;
			}
		} catch (com.lhy.tool.optimizer.LBFGS.ExceptionWithIflag e) {

			/*
			 *  if the error of arguments or function is during
			 *  the tolerance, we also consider it converges.
			 */
			/*if (isNaN(args)) {
				Utils.println("FunRes = " + funcRes + "; ArgsError = " +
					argsError + "; FuncError = " + funcError +
						"; Args=" + Utils.convertToString(args, ",") +
						"; MappedArgs=" + Utils.convertToString(mappedArgs, ","));

				wrapArgs();

				 Utils.print(args);
			}*/
			if (iter >= 150 || debug >= 1) {
				Utils.println("iter = " + iter + "; iflag = " + iflag[0] +
					"; funRes = " + funcRes[0] + "; argsError = " +
					argsError + "; funcError = " + funcError +
						"; args=" + Utils.convertToString(args, ","));
			}

			if (argsError <= errorThreshold || funcError <= errorThreshold) return true;
			
			throw new AIRException(e.toString());
		}

		Utils.err("Not Converged.");
		return false;
	}

	private boolean optimizeViaGradientDecent() throws ToolException {
		initMappedArgs();

		double argsError   = Double.MAX_VALUE;
		double funcError   = Double.MAX_VALUE;
		double step        = 1.0;
		double[] funcRes   = new double[1];
		double[] gradients = new double[args.length];

		calculateMappedGradients(gradients, funcRes);

		do {
			double[] argsOld       = Arrays.copyOf(args, args.length);
			double[] mappedArgsOld = Arrays.copyOf(mappedArgs, mappedArgs.length);
			double[] gradientsOld  = Arrays.copyOf(gradients, gradients.length);
			double funcResOld      = funcRes[0];

			for (int i = 0; i < mappedArgs.length; i ++) {
				mappedArgs[i] -= step * gradients[i];
			}
			
			updateArgs();
			calculateMappedGradients(gradients, funcRes);

			while (funcRes[0] >= funcResOld && step != 0) {
				step /= 2.0;
				/*Utils.println("Reducing step size to " + step + ": funcRes = " +
						funcRes + ", funcResOld = " + funcResOld);*/

				args       = Arrays.copyOf(argsOld, argsOld.length);
				mappedArgs = Arrays.copyOf(mappedArgsOld, mappedArgsOld.length);
				gradients  = Arrays.copyOf(gradientsOld, gradientsOld.length);
				for (int i = 0; i < args.length; i ++) {
					mappedArgs[i] -= step * gradients[i];
				}
				updateArgs();
				calculateMappedGradients(gradients, funcRes);
			}

			argsError = Utils.sumOfDiffAbs(args, argsOld);
			funcError = (funcRes[0] - funcResOld) / Math.abs(funcResOld);

			/*Utils.println("funRes = " + funcRes + "; argsError = " +
					argsError + "; funcError = " + funcError +
						"; args=" + Utils.convertToString(args, ","));*/
			
		} while (argsError > errorThreshold || funcError > errorThreshold);

		Utils.println("\tfunRes = " + funcRes + "; argsError = " +
				argsError + "; funcError = " + funcError +
					"; args=" + Utils.convertToString(args, ","));
		return true;
	}

	private void calculateMappedGradients(double[] objectGradients,
						double[] objectFunction) {
		calculateObjectGradients(objectGradients, objectFunction);

		if (objectGradients != null) {
			double[] gradients = new double[args.length];
			double[] tmpValue  = new double[args.length];
			double sum         = 0.0;
			for (int i = 0; i < args.length; i ++) {
				tmpValue[i] = FastMath.exp(mappedArgs[i]);
				sum        += tmpValue[i];
			}
			double deonimator = sum * sum;
			for (int i = 0; i < args.length; i ++) {
				for (int j = 0; j < args.length; j ++) {
					double chainGradient = (i == j ?
							tmpValue[i] * (sum - tmpValue[i]) / deonimator :
							-1.0 * tmpValue[i] * tmpValue[j] / deonimator );
					gradients[i] += chainGradient * objectGradients[j];
				}
			}
			for (int i = 0; i < args.length; i ++) {
				objectGradients[i] = gradients[i];
			}
		}
	}

	private void initMappedArgs() {
		for (int i = 0; i < mappedArgs.length; i ++) {
			mappedArgs[i] = FastMath.log(args[i]);
		}
	}

	private void updateArgs() {
		double sum      = 0.0;
		double max      = 0.0;
		double maxIndex = 0;
		for (int i = 0; i < mappedArgs.length; i ++) {
			args[i] = FastMath.exp(mappedArgs[i]);
			if (args[i] > max) {
				max = args[i];
				maxIndex = i;
			}
		}
		for (int i = 0; i < mappedArgs.length; i ++) {
			if (i != maxIndex) {
				sum += args[i] / max;
			}
		}
		sum = (sum + 1.0) * max;
		for (int i = 0; i < args.length; i ++) {
			args[i] /= sum;
		}
	}

	public boolean isNaN(double[] array) {
		for (double elem : array) {
			if (Double.isNaN(elem)) return true;
		}
		return false;
	}

	protected void wrapArgs() {
		double sum    = 0.0;
		//double[] args = new double[mappedArgs.length];
		for (int i = 0; i < mappedArgs.length; i ++) {
			args[i] = FastMath.exp(mappedArgs[i]);
			if (Double.isInfinite(args[i])) {
				if (mappedArgs[i] < 0) {
					args[i] = FastMath.exp(-100);
				} else {
					args[i] = Double.MAX_VALUE / mappedArgs.length;
				}
			}
			sum += args[i];
		}
		for (int i = 0; i < args.length; i ++) {
			args[i] /= sum;

			if (args[i] == 0.0) args[i] = FastMath.exp(-100.0);
		}
	}
}

