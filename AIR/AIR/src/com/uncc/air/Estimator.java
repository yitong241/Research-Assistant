package com.uncc.air;

import com.lhy.tool.ToolException;

/**
 * @author Huayu Li
 */
public interface Estimator {
	public void estimate() throws ToolException, AIRException;
}
