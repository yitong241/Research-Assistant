package com.uncc.air.util;

import com.lhy.tool.util.Utils;

import java.util.Arrays;
/**
 * @author Huayu Li
 */
public class NgramTuple {

	private static final String STRING_CONNECT = ",";

	private String[] origNgramWords   = null;
	private String[] sortedNgramWords = null;

	public NgramTuple(String[] ngramWords) {
		this.origNgramWords   = ngramWords;
		this.sortedNgramWords = Arrays.copyOf(ngramWords, ngramWords.length);
		if (! checkAscentOrder(this.sortedNgramWords)) {
			Arrays.sort(this.sortedNgramWords);
		}
	}

	public NgramTuple(Object ... word) {
		Object[] array        = word;
		this.sortedNgramWords = new String[array.length];
		this.origNgramWords   = new String[array.length];
		for (int i = 0; i < this.sortedNgramWords.length; i ++) {
			this.sortedNgramWords[i] = (String)array[i];
			this.origNgramWords[i]   = (String)array[i];
		}

		if (! checkAscentOrder(this.sortedNgramWords)) {
			Arrays.sort(this.sortedNgramWords);
		}
	}

	@Override
	public boolean equals(Object object) {
		
		if (! (object instanceof NgramTuple) || object == null) return false;

		String[] another = ((NgramTuple) object).getSortedNgramWords();
		if (another.length != sortedNgramWords.length) return false;

		String[] sortedObject = Arrays.copyOf(another, another.length);

		Arrays.sort(sortedObject);

		for (int i = 0; i < sortedObject.length; i ++) {
			if (! sortedObject[i].equals(sortedNgramWords[i])) {
				return false;
			}
		}

		return true;
	}

	// allows user to operate the array ngramWords outside this object.
	public String[] getOrigNgramWords() {
		return this.origNgramWords;
	}

	public String[] getSortedNgramWords() {
		return this.sortedNgramWords;
	}
 
	@Override
	public int hashCode() {
		int hashCode = 1;
		for (String s : sortedNgramWords) {
			hashCode = 31 * hashCode + s.hashCode();
		}
		return hashCode;
	}

	@Override
	public String toString() {
		return toString(STRING_CONNECT);
	}

	public String toString(String spacer) {
		return Utils.convertToString(origNgramWords, spacer);
	}

	public String toSortedString(String spacer) {
		return Utils.convertToString(sortedNgramWords, spacer);
	}

	private boolean checkAscentOrder(String[] array) {
		for (int i = 1; i < array.length; i ++) {
			if (array[i].compareTo(array[i - 1]) < 0) return false;
		}

		return true;
	}
}
