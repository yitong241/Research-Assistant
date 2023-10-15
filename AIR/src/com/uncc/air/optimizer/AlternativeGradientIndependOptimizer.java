package com.uncc.air.optimizer;

import java.util.Arrays;

import com.lhy.tool.optimizer.LBFGS;

import org.apache.commons.math3.util.FastMath;

import com.lhy.tool.ToolException;
import com.lhy.tool.util.Utils;
import com.uncc.air.AIRException;

/**
 * @author Huayu Li
 */
public abstract class AlternativeGradientIndependOptimizer {
	protected double[] args       = null;
	private double[] mappedArgs   = null;
	private double errorThreshold = 0.0;

	public AlternativeGradientIndependOptimizer(double[] args,
				double errorThreshold) throws AIRException {
		if (Utils.isEmpty(args) || errorThreshold < 0.0) {
			throw new AIRException("Illegal arguments.");
		}

		this.args           = args;
		this.mappedArgs     = new double[args.length];
		this.errorThreshold = errorThreshold;

		initMappedArgs();
	}

	public abstract double getObjectFunction();
	public abstract double[] getObjectGradients();

	public boolean optimize() throws ToolException, AIRException {
		//return optimizeViaGradientDecent();
		return optimizeViaLBGS();
	}

	private boolean optimizeViaLBGS() throws AIRException, ToolException {
		double argsError = Double.MAX_VALUE;
		double funcError = Double.MAX_VALUE;
		double funcRes   = getObjectFunction();
		int[] iflag      = {0};
		int[] iprint     = {-1, 0};
		int m            = 5;
		double[] diags   = new double[mappedArgs.length];
		try {
			do {
				double[] argsOld   = Arrays.copyOf(args, args.length);
				double funcResOld  = funcRes;
				double[] gradients = getGradients();
				

				LBFGS.lbfgs(mappedArgs.length , m , mappedArgs , funcResOld,
						gradients , false , diags , iprint , 1.0e-2, 1e-20 , iflag );

				updateArgs();
				funcRes   = getObjectFunction();

				argsError = Utils.sumOfDiffAbs(args, argsOld);
				funcError = (funcResOld - funcRes) / Math.abs(funcRes);

			} while ( iflag[0] != 0 || argsError > errorThreshold ||
					funcError > errorThreshold);
	
			/*Utils.println("funRes = " + funcRes + "; argsError = " +
					argsError + "; funcError = " + funcError +
						"; args=" + Utils.convertToString(args, ","));*/

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
			if (argsError <= errorThreshold || funcError <= errorThreshold) return true;

			/*Utils.println("FunRes = " + funcRes + "; ArgsError = " +
					argsError + "; FuncError = " + funcError +
						"; Args=" + Utils.convertToString(args, ",") +
						"; MappedArgs=" + Utils.convertToString(mappedArgs, ","));*/
			throw new AIRException(e.toString());
		}

		Utils.err("Not Converged.");
		return false;
	}

	private boolean optimizeViaGradientDecent() throws ToolException {
		double argsError = Double.MAX_VALUE;
		double funcError = Double.MAX_VALUE;
		double step      = 1.0;
		double funcRes   = getObjectFunction();
		do {
			double[] gradients     = getGradients();
			double[] argsOld       = Arrays.copyOf(args, args.length);
			double[] mappedArgsOld = Arrays.copyOf(mappedArgs, mappedArgs.length);
			double[] gradientsOld  = Arrays.copyOf(gradients, gradients.length);
			double funcResOld      = funcRes;

			for (int i = 0; i < mappedArgs.length; i ++) {
				mappedArgs[i] -= step * gradients[i];
			}
			updateArgs();
			funcRes = getObjectFunction();

			while (funcRes >= funcResOld && step != 0) {
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
				funcRes = getObjectFunction();
			}

			argsError = Utils.sumOfDiffAbs(args, argsOld);
			funcError = Math.abs((funcRes - funcResOld) / funcRes);

			/*Utils.println("funRes = " + funcRes + "; argsError = " +
					argsError + "; funcError = " + funcError +
						"; args=" + Utils.convertToString(args, ","));*/
			
		} while (argsError > errorThreshold || funcError > errorThreshold);

		Utils.println("\tfunRes = " + funcRes + "; argsError = " +
				argsError + "; funcError = " + funcError +
					"; args=" + Utils.convertToString(args, ","));
		return true;
	}

	private double[] getGradients() throws ToolException {
		double[] objectGradients = getObjectGradients();
		double[] gradients       = new double[args.length];
		double[] tmpValue        = new double[args.length];
		double sum               = 0.0;
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
		return gradients;
	}

	private void initMappedArgs() {
		for (int i = 0; i < mappedArgs.length; i ++) {
			mappedArgs[i] = FastMath.log(args[i]);
		}
	}

	private void updateArgs() {
		double sum = 0.0;
		for (int i = 0; i < mappedArgs.length; i ++) {
			args[i] = FastMath.exp(mappedArgs[i]);
			sum    += args[i];
		}
		for (int i = 0; i < args.length; i ++) {
			args[i] /= sum;
		}
	}
}

