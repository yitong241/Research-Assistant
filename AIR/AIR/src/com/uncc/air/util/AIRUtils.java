package com.uncc.air.util;

import com.lhy.tool.util.Utils;
import com.uncc.air.AIRConstants;
import com.uncc.air.AIRException;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;

/**
 * @author Huayu Li
 */
public class AIRUtils {

	/*
	 * Parses reviews. The review line format is as the following:
	 * ReviewId Overrall_Rating Text
	 * 
	 * The returned array is as the format {ReviewId, Overall_Rating, Text}
	 */
	public static String[] parseReviews(String line) {
		if (Utils.isEmpty(line)) return null;

		line            = line.trim();
		String text     = "";
		int rating      = 0;
		int s           = line.indexOf(" ");
		String reviewId = line.substring(0, s);
		line         = line.substring(s + 1);
		s               = line.indexOf(" ");
		if (s < 0) {
			rating = Integer.parseInt(line);
		} else {
			rating      = Integer.parseInt(line.substring(0, s));
			s           = line.indexOf(" ");
			text        = line.substring(s + 1);
		}
		return new String[]{reviewId, String.valueOf(rating), text};
	}

	public static double[] parseGammas(String text) throws AIRException {
		double[] gammas = Utils.parseDoubleArray(removeDoubleQuotation(text),
							AIRConstants.SPACER_COMMA);
		if (gammas != null && gammas.length == 2) {
			return gammas;
		}
		throw new AIRException(String.format("Gammas format is not correct. [gammas=%s]", text));
	}

	public static String removeDoubleQuotation(String text) {
		if (text != null) {
			while (text.startsWith("\"")) {
				text = text.substring(1);
			}
			while (text.lastIndexOf('"') == text.length() - 1) {
				text = text.substring(0, text.length() - 1);
			}
		}
		return text;
	}

	public static double getArithmeticResult(String s) throws AIRException,
						NumberFormatException {
		String error = null;
		try {
			return Double.parseDouble(s);
		} catch (NumberFormatException e) {
			error = e.getMessage() + "\n" + e.getCause();
		}
		
		String[] array = s.split("\\*");
		if (array.length != 2) throw new AIRException(error);
		
		return Double.parseDouble(array[0]) * Double.parseDouble(array[1]);
	}

	public static String[] getSortedKeyInDescent(HashMap<String, Integer> map) {
		if (map == null || map.isEmpty()) return null;

		ArrayList<Object[]> list = new ArrayList<Object[]>();
		String[]            keys = new String[map.size()];
		for (String key : map.keySet()) {
			list.add(new Object[]{key, map.get(key)});
		}
		Collections.sort(list, new Comparator<Object[]> () {
			@Override
			public int compare(Object[] obj1, Object[] obj2) {
				Integer i1 = (Integer)obj1[1];
				Integer i2 = (Integer)obj2[1];
				return i2.compareTo(i1);
			}
		});
		for (int i = 0; i < list.size(); i ++) {
			keys[i] = (String) list.get(i)[0];
		}
		return keys;
	}
}
