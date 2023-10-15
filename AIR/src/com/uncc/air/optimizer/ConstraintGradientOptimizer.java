package com.uncc.air.optimizer;

import java.util.Arrays;

import com.lhy.tool.ToolException;
import com.lhy.tool.util.Utils;
import com.uncc.air.AIRException;

/**
 * @author Huayu Li
 */
public abstract class ConstraintGradientOptimizer {
	protected double[] args       = null;
	private double M              = 1.0;
	private double errorThreshold = 0.0;

	public ConstraintGradientOptimizer(double[] args,
				double errorThreshold) throws AIRException {
		if (Utils.isEmpty(args) || M <= 0) {
			throw new AIRException("Illegal arguments.");
		}

		this.args           = args;
		this.errorThreshold = errorThreshold;
	}

	public abstract double getObjectFunction();
	public abstract double[] getObjectGradients();

	public void optimize() throws ToolException {
		double argsError = Double.MAX_VALUE;
		double funcError = Double.MAX_VALUE;
		double funcRes   = getOptimizeObject();
		double increase  = 10.0;
		
		do {
			System.out.println("M = " + M);

			optimizeArgs();

			M = increase * M;
			if (M >= 10) break;
		} while (!convergence());
		Utils.println("funRes = " + funcRes + "; argsError = " +
				argsError + "; funcError = " + funcError +
					"; args=" + Utils.convertToString(args, ","));
	}

	private void optimizeArgs() throws ToolException {
		double argsError = Double.MAX_VALUE;
		double funcError = Double.MAX_VALUE;
		double step      = 1.0;
		double funcRes   = getOptimizeObject();
		do {
			double[] argsOld   = Arrays.copyOf(args, args.length);
			double funcResOld  = funcRes; 
			double[] gradients = getGradients();

			for (int i = 0; i < args.length; i ++) {
				args[i] -= step * gradients[i];
			}
			funcRes   = getOptimizeObject();

			while (funcRes >= funcResOld && step != 0) {
				step /= 2.0;
				/*Utils.println("Reducing step size to " + step + ": funcRes = " +
						funcRes + ", funcResOld = " + funcResOld);*/

				args      = Arrays.copyOf(argsOld, argsOld.length);
				gradients = getGradients();
				for (int i = 0; i < args.length; i ++) {
					args[i] -= step * gradients[i];
				}
				funcRes   = getOptimizeObject();
			}

			argsError = Utils.sumOfDiffAbs(args, argsOld);
			funcError = Math.abs((funcRes - funcResOld) / funcRes);

			/*Utils.println("\tfunRes = " + funcRes + "; argsError = " +
					argsError + "; funcError = " + funcError +
						"; args=" + Utils.convertToString(args, ","));*/
			
		} while (argsError > errorThreshold || funcError > errorThreshold);

		Utils.println("\tfunRes = " + funcRes + "; argsError = " +
				argsError + "; funcError = " + funcError +
					"; args=" + Utils.convertToString(args, ","));
	}

	private boolean convergence() {
		for (int i = 0; i < args.length; i ++) {
			if (Math.abs(args[i] * (-1.0)) > errorThreshold) return false;
			if (Math.abs(args[i] - 1) > errorThreshold) return false;
		}
		return true;
	}

	private double getOptimizeObject() {
		double sum1 = 0.0;
		double sum2 = 0.0;
		for (int i = 0; i < args.length; i ++) {
			double x   = (args[i] < 0 ?  args[i] : 0);
			double x_1 = (1 - args[i] <0 ? 1 - args[i] : 0);
			sum1 += M * x *x + M * x_1 * x_1;
			sum2 += args[i];
		}
		return getObjectFunction() + sum1 + (sum2 - 1) * (sum2 - 1);
	}

	private double[] getGradients() throws ToolException {
		double[] objectGradients = getObjectGradients();
		double[] gradients       = new double[args.length];

		for (int i = 0; i < gradients.length; i ++) {
			gradients[i] = objectGradients[i] +
					(args[i] < 0 ? 2 * M * args[i] : 0) +
					((1 - args[i]) < 0 ? 2 * M * (args[i] - 1) : 0) +
					2 * M * (Utils.sum(args) - 1);
		}

		return gradients;
	}
}
