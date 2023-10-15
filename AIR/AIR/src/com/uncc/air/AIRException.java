package com.uncc.air;

/**
 * @author Huayu Li
 */
public class AIRException extends Exception {
	private static final long serialVersionUID = 2L;
	public AIRException(String message) {
		super(message);
	}
	public AIRException(String format, Object...args) {
		super(String.format(format, args));
	}
}
