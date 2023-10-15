package com.uncc.air.data;

import java.util.ArrayList;
import java.util.HashMap;

import com.lhy.tool.util.Utils;
import com.uncc.topicmodel.data.Document;

/**
 * @author Huayu Li
 */
public class AIRDocument extends Document {
	private HashMap<Integer, Integer> uniqueWordMap = null;;

	public AIRDocument(ArrayList<Integer> wordIds, double rating,
						RatingScaler ratingScaler) {
		if (Utils.isEmpty(wordIds)) {
			words            = new int[0];
			topics           = new int[0];
			sentiments       = new int[0];
			uniqueWords      = new int[0];
			uniqueWordCounts = new int[0];
		} else {
			words         = new int[wordIds.size()];
			uniqueWordMap = new HashMap<Integer, Integer>();

			for (int i = 0; i < wordIds.size(); i ++) {
				words[i] = wordIds.get(i);
				if (uniqueWordMap.containsKey(words[i])) {
					uniqueWordMap.put(words[i], uniqueWordMap.get(words[i]) + 1);
				} else {
					uniqueWordMap.put(words[i], 1);
				}
			}

			this.topics     = new int[words.length];
			this.sentiments = new int[words.length];
			this.wordNum    = words.length;
			this.rating     = rating;
			this.Ri         = ratingScaler.scaleRating(rating);

			Integer[] keys   = uniqueWordMap.keySet().toArray(new Integer[0]);
			uniqueWords      = new int[keys.length];
			uniqueWordCounts = new int[keys.length];
			for (int i = 0; i < keys.length; i ++) {
				uniqueWords[i]      = keys[i];
				uniqueWordCounts[i] = uniqueWordMap.get(uniqueWords[i]);
			}
		}
	}

	@Override
	public int getWordNum() {
		return words.length;
	}

	@Override
	public int getUniqueWordNum() {
		return uniqueWords.length;
	}

	@Override
	public int getUniqueWordCount(int wordId) {
		return uniqueWordMap.get(wordId);
	}
}
