package com.uncc.air.data;

/**
 * @author Huayu Li
 */
public class RateBeerDataset implements RatingScaler {
	private static final int[] RATING_MAX_VALUES = {
						5, // appearance
						10, // aroma
						5, // palate
						10}; // taste
						//20}; // overall

	private boolean isRatingMappedToSpecifiedAspect = false;

	public RateBeerDataset(boolean isRatingMappedToSpecifiedAspect) {
		this.isRatingMappedToSpecifiedAspect = isRatingMappedToSpecifiedAspect;
	}

	@Override
	public double scaleRating(double rating) {
		return rating / 20.0 - 0.025;
	}

	@Override
	public double recoverRating(double rating, int topicIndex) {
		if (isRatingMappedToSpecifiedAspect) {
			return (rating + 0.025) * RATING_MAX_VALUES[topicIndex];
		} else {
			return rating;
		}
	}

	@Override
	public double cut(double rating, int topicIndex) {
		if (isRatingMappedToSpecifiedAspect) {
			if (rating < 1.0) return 1.0;
			else
			if (rating > RATING_MAX_VALUES[topicIndex]) {
				return RATING_MAX_VALUES[topicIndex];
			}
		}
		return rating;
	}

	@Override
	public double getOverallMaxRating() {
		return 20.0;
	}

	@Override
	public double getOverallMinRating() {
		return 1.0;
	}
}
