package com.uncc.air.data;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;

import com.lhy.tool.util.Utils;
import com.uncc.topicmodel.TopicModelException;
import com.uncc.topicmodel.data.Dataset;
import com.uncc.topicmodel.data.Dictionary;
import com.uncc.air.AIRConstants;

/**
 * @author Huayu Li
 */
public class AIRDataset extends Dataset implements RatingScaler {
	public static final char[] SPEC_CHARS = {'-'};

	public AIRDataset(Dictionary dictionary, File datasetFile,
				File dicOutputFile, boolean isRestore)
							throws TopicModelException {
		super(datasetFile, dicOutputFile, dictionary, isRestore);
	}

	@Override
	public double scaleRating(double rating) {
		if (rating < 0 || rating > 5) throw new RuntimeException(
			String.format("Rating % is out of range: 1-5.", rating));
		return (rating - 0.5) / 5.0;
	}

	@Override
	public double recoverRating(double rating, int topicIndex) {
		return rating * 5.0 + 0.5;
	}

	@Override
	public double cut(double rating, int topicIndex) {
		double v = rating;
		if (v > 5.0) return 5.0;
		if (v < 1.0) return 1.0;
		return v;
	}

	@Override
	public double getOverallMaxRating() {
		return 5.0;
	}

	@Override
	public double getOverallMinRating() {
		return 1.0;
	}

	@Override
	public void create(File datasetFile, File dicOutputFile,
				Dictionary dictionary) throws TopicModelException {
		parse(datasetFile, dictionary, false);
		dictionary.store(dicOutputFile);
	}

	@Override
	public void init() {
		if (AIRConstants.CONFIG_MANAGER.getRatingScaler() == null) {
			AIRConstants.CONFIG_MANAGER.setRatingScaler(this);
		}
	};

	@Override
	public void restore(File datasetFile, File dicInputFile,
				Dictionary dictionary) throws TopicModelException {
		dictionary.load(dicInputFile);
		parse(datasetFile, dictionary, true);
	}

	private void parse(File datasetFile, Dictionary dictionary,
					boolean isRestore) throws TopicModelException {
		if (! Utils.exists(datasetFile)) 
			throw new TopicModelException("Dataset file doesn't exist. [" + datasetFile.getAbsolutePath() + "]");

		BufferedReader reader = null;
		try {
			reader      = Utils.createBufferedReader(datasetFile);
			String line = null;
			while ((line = reader.readLine()) != null) {
				if ((line = line.trim()).equals("")) continue;

				@SuppressWarnings("unused")
				String revIndex = null;
				String text     = "";
				int rating      = 0;
				int s    = line.indexOf(" ");
				revIndex = line.substring(0, s);
				line     = line.substring(s + 1);
				s        = line.indexOf(" ");
				try {
					if (s < 0) {
						rating = Integer.parseInt(line);
					} else {
						rating      = Integer.parseInt(line.substring(0, s));
						s           = line.indexOf(" ");
						text        = line.substring(s + 1);
					}
				} catch (NumberFormatException e) {
					throw new TopicModelException(e.toString());
				}
	
				addDocument(rating, text, dictionary, isRestore);
			}
			if (dictionary == null || dictionary.isEmpty()) {
				throw new TopicModelException("Dictionary is Empty!");
			}
		} catch (IOException e) {
			e.printStackTrace();
		} finally {
			Utils.cleanup(reader);
		}
	}

	private void addDocument(int rating, String text, Dictionary dictionary,
					boolean isRestore) throws TopicModelException {
		if (Utils.isEmpty(text)) return;

		if (rating < AIRConstants.CONFIG_MANAGER.getRatingScaler().getOverallMinRating() ||
				rating > AIRConstants.CONFIG_MANAGER.getRatingScaler().getOverallMaxRating()) {
			throw new TopicModelException ("Rating is out of range.");
		}

		String[] words             = text.trim().split(" ");
		ArrayList<Integer> wordIds = new ArrayList<Integer>();
		for (String word : words) {
			if (Utils.isEmpty(word = word.trim())) continue;

			if (isRestore) {
				// restore dataset
				if (dictionary.contains(word)) {
					wordIds.add(dictionary.getWordId((word)));
				} else {
					throw new TopicModelException("Failed to locate the word[" + word + "] in the dictionary.");
				}
			} else {
				// load a new dataset
				int wordId = dictionary.getSize();
				if (dictionary.contains(word)) {
					wordId = dictionary.getWordId(word);
				} else {
					dictionary.add(word, wordId);
				}
				wordIds.add(wordId);
			}
		}

		getDocuments().add(new AIRDocument(wordIds, rating,
				AIRConstants.CONFIG_MANAGER.getRatingScaler()));
	}
}
