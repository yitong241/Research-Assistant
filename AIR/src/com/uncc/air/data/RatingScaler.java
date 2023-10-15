package com.uncc.air.data;

/**
 * @author Huayu Li
 */
public interface RatingScaler {
	public double scaleRating(double rating);
	public double recoverRating(double rating, int topicIndex);
	public double cut(double rating, int topicIndex);
	public double getOverallMaxRating();
	public double getOverallMinRating();
}
