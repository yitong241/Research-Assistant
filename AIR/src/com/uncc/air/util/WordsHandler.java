package com.uncc.air.util;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Comparator;
import java.util.StringTokenizer;
import java.util.Map;
import java.util.Set;

import org.apache.commons.math3.util.FastMath;

import com.lhy.tool.ToolException;
import com.lhy.tool.util.Utils;
import com.uncc.air.AIRConstants;
import com.lhy.tool.text.TextUtils;

/**
 * @author Huayu Li
 */
public class WordsHandler {

	/**
	 * Some invalid words which may have long tail, such as "so", it
	 * may have long tail, sooooo
	 */
	public static final String[] LONG_TAIL_WORDS    = {"so", "eh", "um", "eh"};
	/**
	 * The word maybe considered as the outlier when its some characters
	 * repeat a lot of times continuously.
	 */
	public static final int OUTLIER_CHAR_REPEAT_NUM = 4;
	/**
	 * The word maybe considered as the outlier word when
	 * it repeats a lot of times in one sentence.
	 * <code>OUTLIER_CHAR_REPEAT_NUM</code> is the repeat number.
	 */
	public static final int OUTLIER_WORD_REPEAT_NUM = 4;


	public static String handleWords(String reviewIndex,
			Map<String, String> stemWords,
				Set<String> stopWords,
				BufferedWriter stemWriter,
				BufferedWriter stopWriter,
					BufferedWriter upperWriter,
						BufferedWriter otherLanWriter,
							BufferedWriter outlierWriter,
							Map<String, String> stemMap,
								String text, char[] specChars){
		return handleWords(reviewIndex, stemWords, stopWords,
				stemWriter, stopWriter, upperWriter,
				otherLanWriter, outlierWriter, stemMap, text,
				specChars, new String[]{"'", "-"},
				OUTLIER_WORD_REPEAT_NUM, OUTLIER_CHAR_REPEAT_NUM,
				27);
		
	}
	/**
	 * Simply handles one text.
	 * 
	 * Handles words in the following steps:
	 *	Step 1: removes punctuation
	 * 	Step 2: converts to lower case letter
	 * 	Step 3: uses stemmer
	 *
	 * @param reviewIndex		the review index that is used to recognize each line
	 * @param stemWords		the stem words
	 * @param stopWords		the stop words
	 * @param stemWriter		the stem word output writer
	 * @param stopWriter		the stop word output writer
	 * @param upperWriter		the upper word output writer
	 * @param otherLanWriter	the other language writer
	 * @param outlierWriter		the outlier output writer
	 * @param stemMap		the stemMap
	 * @param text			the text that needs to handle
	 * @param specChars		some special chars that won't be spitted
	 * @param apostrophes		the apostrophe array
	 * @param outlierWordRepeatNum	the outlier word repeat number
	 * @param outlierCharRepeatNum  the char in outlier word repeat number
	 * @param maxWordLen 		the max word length
	 * @return			the handled text
	 */
	public static String handleWords(String reviewIndex,
			Map<String, String> stemWords,
				Set<String> stopWords,
				BufferedWriter stemWriter,
				BufferedWriter stopWriter,
				BufferedWriter upperWriter,
				BufferedWriter otherLanWriter,
				BufferedWriter outlierWriter,
				Map<String, String> stemMap,
				String text, char[] specChars,
				String[] apostrophes, int outlierWordRepeatNum,
				int outlierCharRepeatNum, int maxWordLen ) {
		if (Utils.isEmpty(text)) return text;

		// split string text into words
		if (outlierWriter != null) {
			text = removeOutlierWords(text, outlierWriter, reviewIndex,
					specChars, outlierWordRepeatNum,
					outlierCharRepeatNum, maxWordLen);
		} else {
			text = TextUtils.splitWords(text, specChars);
		}

		StringBuffer buffer             = new StringBuffer();
		StringBuffer capitalWordsBuffer = new StringBuffer();
		StringBuffer stemWordsBuffer    = new StringBuffer();
		StringBuffer stopWordsBuffer    = new StringBuffer();
		StringBuffer otherLanBuffer     = new StringBuffer();
		String corrS   = "=>";
		String splitS  = ", ";
		String[] words = text.split(" ");
		for (String word : words) {
			if ((word = word.trim()).length() <= 1) continue;
			
			if (! Utils.isEnglish(word)) {
				otherLanBuffer.append(word).append(" ");
			} else
			if (Utils.isEnglishNumeric(word)) {
				stopWordsBuffer.append(word).append(" ");
			} else {
				// convert to lower case letter
				String lowerCase = word.toLowerCase();
				// store the changes
				if (! lowerCase.equals(word)) {
					capitalWordsBuffer.append(word)
							.append(corrS)
							.append(lowerCase)
							.append(splitS);
				}
				if (stopWords.contains(lowerCase)) {
					// remove the stop word
					stopWordsBuffer.append(word).append(" ");
				} else {
					// stem
					String newword = null;
					if (stemWords != null) {
						newword = stemWords.get(lowerCase);
					}
					if (newword == null) {
						newword = new Stemmer(lowerCase).getStemResult();
					}
					// store the changes
					if (! newword.equals(lowerCase)) {
						stemWordsBuffer.append(lowerCase)
								.append(corrS)
								.append(newword)
								.append(splitS);
						if (stemMap != null) {
							if (stemMap.containsKey(lowerCase)) {
								if (! stemMap.get(lowerCase).equals(newword)) {
									Utils.err(String.format(
										"Stem word doesnot match.[word = %s][newword = %s][stored = %s]",
										lowerCase, newword, stemMap.get(lowerCase)));
								}
							} else {
								stemMap.put(lowerCase, newword);
							}
						}
					}
	
					newword = TextUtils.removeApostrophe(newword, apostrophes);
					if (newword.length() > 1 &&
							//! Utils.isNumeric(newword) &&
							! existNumber(newword) &&
							! checkNumberMeaning(newword)) {
						if (! stopWords.contains(newword) &&
							! isLongTailWord(LONG_TAIL_WORDS, newword)) {
							// remove stop word again
							buffer.append(newword).append(" ");
						} else {
							stopWordsBuffer.append(word).append(" ");
						}
					} // end if (newword.length() > 1)
				} // end else of if (stopWords.contains(word))
			} // end else of if (! Utils.isEnglish(word))
		}

		if (capitalWordsBuffer.length() > 0 && upperWriter != null) {
			try {
				upperWriter.write(reviewIndex +
						capitalWordsBuffer.substring(0,
							capitalWordsBuffer.length() -
								splitS.length()));
				upperWriter.newLine();
				upperWriter.flush();
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
		if (stemWordsBuffer.length() > 0 && stemWriter != null) {
			try {
				stemWriter.write(reviewIndex +
						stemWordsBuffer.substring(0,
							stemWordsBuffer.length() -
								splitS.length()));
				stemWriter.newLine();
				stemWriter.flush();
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
		if (stopWordsBuffer.length() > 0 && stopWriter != null) {
			try {
				stopWriter.write(reviewIndex + stopWordsBuffer.toString());
				stopWriter.newLine();
				stopWriter.flush();
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
		if (otherLanBuffer.length() > 0 && otherLanWriter != null) {
			try {
				otherLanWriter.write(reviewIndex + otherLanBuffer.toString());
				otherLanWriter.newLine();
				otherLanWriter.flush();
			} catch (IOException e) {
				e.printStackTrace();
			}
		}

		return buffer.toString();
	}

	
	/**
	 * Removes the outlier words. It will call <code>splitWords</code>
	 * method to split the word. If the word appear continuously equal
	 * or more than <code>OUTLIER_WORD_REPEAT_NUM</code>, or some
	 * characters in the word appears continuously equal or more than
	 * <code>OUTLIER_CHAR_REPEAT_NUM</code>, or the length of the word
	 * is more than 27, this word will be considered as the outlier.
	 * 
	 * @param content	the content that needs to remove outliter words
	 * @param outlierWriter	the outlier writer
	 * @param reviewIndex	the review index that will be written in outliterWriter
	 * @param specChars	the special chars that won't be splited
	 * @param outlierWordRepeatNum	the outlier word repeat number
	 * @param outlierCharRepeatNum  the char in outlier word repeat number
	 * @param maxWordLen 		the max word length
	 * @return		the new text that is removed outlier words
	 */
	public static String removeOutlierWords(String content,
			BufferedWriter outlierWriter,
				String reviewIndex, char[] specChars,
				int outlierWordRepeatNum,
				int outlierCharRepeatNum, int maxWordLen) {
		if (Utils.isEmpty(content)) return content;

		content              = TextUtils.splitWords(content, specChars);
		String[] array       = content.split(" ");
		StringBuffer buffer  = new StringBuffer();
		StringBuffer outlier = new StringBuffer();
		try {
			String prevElem = null;
			int count       = 0;
			String markers  = "";
			for (String element : array) {
				if (Utils.isEmpty(element)) continue;

				if (element.equals(prevElem)) {
					count ++;
					if ((count + 1) >= outlierWordRepeatNum) {
						outlier.append(element).append(" ");
						count   = 0;
						markers = "";
					} else {
						markers += element + " ";
					}
					continue;
				}
				if (count > 0) {
					buffer.append(markers);
					markers = "";
				} 
				prevElem = element;
				if (element.length() > maxWordLen || TextUtils
					.isOutlierWord(element, outlierCharRepeatNum)) {
					outlier.append(element).append(" ");
					continue;
				}
				buffer.append(element).append(" ");
			}

			if (outlier.length() > 0 && outlierWriter != null) {
				Utils.write(outlierWriter, reviewIndex + outlier.toString().trim(),
						true);
			} else
			if (outlier.length() > 0) {
				Utils.println(reviewIndex + outlier.toString().trim());
			}
		} catch (IOException e) {
			e.printStackTrace();
		}

		return buffer.toString().trim();
	}

	public static boolean removeLessTimeWords(File inputFile, File trainOutputFile,
			File testOutputFile, File validationOutputFile,
				HashMap<String, Integer> wordNumMap,
					int revNum, float testRatio,
					float validationRatio, int minWordFrequnce,
					int minWordNum, int maxWordNum,
					ArrayList<Integer>reviewLineIndexSet) {
		return removeLessTimeWords(inputFile, trainOutputFile,
				testOutputFile, validationOutputFile,
				wordNumMap, revNum, testRatio, validationRatio,
				minWordFrequnce, minWordNum, maxWordNum, null,
				reviewLineIndexSet);
	}

	/*
	 * If trainReviewLineIndexSet is null, it won't remove those
	 * reviews whose word number is less than minWordNum or larger
	 * than maxWordNum. In other words, the minWordNum and maxWordNum
	 * will be not used in this case.
	 */
	public static boolean removeLessTimeWords(File inputFile, File trainOutputFile,
			File testOutputFile, File validationOutputFile,
				HashMap<String, Integer> wordNumMap,
					int revNum, float testRatio,
					float validationRatio, int minWordFrequnce,
					int minWordNum, int maxWordNum,
					String[] excludingWords,
					ArrayList<Integer> trainReviewLineIndexSet) {
		if (inputFile == null || trainOutputFile == null) return false;

		int[] classifiFlag = new int[revNum];
		int testEnd        = (int)(testRatio * revNum);
		int validationEnd  = testEnd + ((int)(validationRatio * revNum));
		for (int i = 0; i < testEnd; i ++) {
			classifiFlag[i] = 1;
		}
		for (int i = testEnd; i < validationEnd; i ++) {
			classifiFlag[i] = 2;
		}
		Utils.shuffle(classifiFlag);

		Utils.println("removeLessFrequentWords :: Test Data Size       = " + testEnd);
		Utils.println("removeLessFrequentWords :: Validation Data Size = " + (validationEnd - testEnd));
		Utils.println("removeLessFrequentWords :: Train Data Size      = " + (revNum - validationEnd));

		BufferedReader reader      = null;
		BufferedWriter trainWriter = null;
		BufferedWriter testWriter  = null;
		BufferedWriter validWriter = null;
		try {
			reader              = Utils.createBufferedReader(inputFile);
			trainWriter         = Utils.createBufferedWriter(trainOutputFile);
			testWriter          = Utils.createBufferedWriter(testOutputFile);
			validWriter         = Utils.createBufferedWriter(validationOutputFile);
			String line         = null;
			int count           = 0;
			int validCount      = 0;
			int testCount       = 0;
			int validationCount = 0;
			int trainCount      = 0;
			while ((line = reader.readLine()) != null) {
				line = line.trim();

				StringBuffer buffer = new StringBuffer();
				String reviewIndex  = null;
				String rating       = null;
				String text         = "";
				int s       = line.indexOf(" ");
				reviewIndex = line.substring(0, s);
				line        = line.substring(s + 1);
				s           = line.indexOf(" ");
				if (s < 0) {
					rating = line;
				} else {
					rating      = line.substring(0, s);
					s           = line.indexOf(" ");
					text        = line.substring(s + 1);
				}
				if (Utils.isEmpty(text)) {
					Utils.err("Empty review text!");
					Utils.cleanup(reader);
					reader = null;
					return false;
				}

				buffer.append(reviewIndex)
					.append(" ")
					.append(rating)
					.append(" ");

				String[] words  = text.split(" ");
				boolean isEmpty = true;
				int wordCount   = 0;
				for (String word : words) {
					if (! Utils.isEmpty(excludingWords)) {
						boolean isFound = false;
						for (String excludingWord : excludingWords) {
							if (excludingWord.equals(word)) {
								isFound = true;
								break;
							}
						}
						if (isFound) {
							buffer.append(word).append(" ");
							continue;
						}
					}
					Integer num = wordNumMap.get(word);
					if (num != null && num >= minWordFrequnce) {
						buffer.append(word).append(" ");
						wordCount ++;
						isEmpty = false;
					}
				}
				if (! isEmpty) {
					/*
					 * If trainReviewLineIndexSet is not null,
					 * the wordCount must be >= minWordNum, &
					 * <= maxWordNum
					 */
					if (trainReviewLineIndexSet != null &&
							(wordCount < minWordNum ||
								wordCount > maxWordNum)) {
						count ++;
						continue;
					}
					if (testWriter != null && classifiFlag[count] == 1) {
						Utils.write(testWriter, buffer.toString(), true);
						testCount ++;
					} else
					if (validWriter != null && classifiFlag[count] == 2){
						Utils.write(validWriter, buffer.toString(), true);
						validationCount ++;
						
					} else
					if (trainWriter != null) {
						Utils.write(trainWriter, buffer.toString(), true);
						trainCount ++;
						if (trainReviewLineIndexSet != null) {
							trainReviewLineIndexSet.add(count);
						}
					}
					validCount ++;
				}
				count ++;
			}

			Utils.println("removeLessTimeWords :: ReviewCount      = " + count);
			Utils.println("removeLessTimeWords :: ValidReviewCount = " + validCount);
			Utils.println("removeLessTimeWords :: TrainCount       = " + trainCount);
			Utils.println("removeLessTimeWords :: ValidationCount  = " + validationCount);
			Utils.println("removeLessTimeWords :: TestCount        = " + testCount);
			Utils.println("removeLessTimeWords :: validationRatio  = " + validationRatio);
			Utils.println("removeLessTimeWords :: testRatio        = " + testRatio);

			return true;
		} catch (IOException e) {
			e.printStackTrace();
		} finally {
			Utils.cleanup(reader);
			Utils.cleanup(trainWriter);
			Utils.cleanup(testWriter);
			Utils.cleanup(validWriter);
		}

		return false;
	}

	/**
	 * Check the string if the number order, such as 1st.
	 * 
	 * @param s	the string need to check
	 * @return	<code>true</code> if it is number order;
	 * 		<code>false</code> otherwise
	 */
	private static boolean checkNumberMeaning(String s) {
		if (Utils.isEmpty(s)) return false;

		s = s.trim();
		if (s.equals("1st") || s.equals("2nd") || s.equals("3rd")) {
			return true;
		} else
		if (s.length() > 2 && s.lastIndexOf("th") == s.length() - 2 &&
				Utils.isEnglishNumeric(s.substring(0, s.length() - 2))) {
			return true;
		}

		return false;
	}

	/**
	 * Check if a string contains number
	 * @param s	a string
	 * @return	<code>true</code> if the string contains number;
	 * 		<code>false</code> otherwise
	 */
	public static boolean existNumber(String s) {
		if (! Utils.isEmpty(s)) {
			for (int i = 0; i < s.length(); i ++) {
				if (Character.isDigit(s.charAt(i))) return true;
			}
		}
		return false;
	}

	public static String removeTitleWords(String text, String title,
				Map<String, String> stemWords,
					char[] otherSplitChars) {
		if (Utils.isEmpty(text) || Utils.isEmpty(title)) return text;


		text  = text.trim().toLowerCase();
		title = title.trim().toLowerCase(); 

		StringBuffer titleBuffer = new StringBuffer();
		String[] titleArray      = title.trim().split(" ");
		for (String word : titleArray) {
			if (word.length() <= 1) continue;

			titleBuffer.append(word).append(" ");
			if (otherSplitChars != null) {
				for (char ch : otherSplitChars) {
					int index = word.indexOf(ch);
					if (index != -1) {
						if (index != 0) {
							titleBuffer.append(word.replace(ch, ' ')).append(" ");
						}
						titleBuffer.append(word.replace(String.valueOf(ch), "")).append(" ");
					}
				}
			}
		}

		HashSet<String> titleWords = handleStemWords(titleBuffer.toString(), stemWords);
		
		String[] words      = text.split(" ");
		StringBuffer buffer = new StringBuffer();
		for (String word : words) {
			if (! titleWords.contains(word)) {
				buffer.append(word).append(" ");
			}
		}
		
		return buffer.toString().trim();
	}

	public static HashSet<String> handleStemWords(String text,
					Map<String, String> stemWords) {
		if (Utils.isEmpty(text)) return null;
		
		HashSet<String> set       = new HashSet<String>();
		StringTokenizer tokenizer = new StringTokenizer(text);
		while (tokenizer.hasMoreElements()) {
			String token = tokenizer.nextToken();
		
			token = token.toLowerCase();
			if (stemWords != null && stemWords.containsKey(token)) {
				token = stemWords.get(token);
			} else {
				token = new Stemmer(token).getStemResult();
			}
			set.add(token);
		}
		
		return set;
	}

	public static void pickValidationFromTrainData(File revInputFile,
			File revValidOutputFile, int reviewNum,
			float validationRatio, File validRandomIndexOutputFile,
			boolean validIndexReadFromFile)
					throws ToolException {
		BufferedWriter revValiWriter = null;
		BufferedReader revReader     = null;
		try {
			int validationNum = (int)(reviewNum * validationRatio);
			int[] validArr    = new int[validationNum];
			String line       = null;
			int revIndex      = 0;
			int validIndex    = 0;
			if (validIndexReadFromFile) {
				validArr = Utils.load1IntArray(validRandomIndexOutputFile, " ");
				if (validArr == null || validArr.length != validationNum) {
					throw new ToolException("Validation Index format is not correct..");
				}
			} else {
				int[] origRevArr  = new int[reviewNum];
				for (int i = 0; i < origRevArr.length;  i ++) {
					origRevArr[i] = i;
				}
				Utils.shuffle(origRevArr);
				for (int i = 0; i < validArr.length; i ++) {
					validArr[i] = origRevArr[i];
				}
				Arrays.sort(validArr);
			}
			revValiWriter = Utils.createBufferedWriter(revValidOutputFile);
			revReader     = Utils.createBufferedReader(revInputFile);
			while ((line = revReader.readLine()) != null &&
						validIndex < validArr.length) {
				if (Utils.isEmpty(line.trim())) continue;

				if (revIndex == validArr[validIndex]) {
					Utils.write(revValiWriter, line, true);
					validIndex ++;
				}
				revIndex ++;
			}
			if (! validIndexReadFromFile &&
					validRandomIndexOutputFile != null) {
				Utils.save(validArr, " ",
						validRandomIndexOutputFile,
						false, true);
			}

			Utils.println("Total Review number = " + reviewNum);
			Utils.println("Validation number   = " + validationNum);
			Utils.println("Validation Ratio    = " + validationRatio);
			
		} catch (IOException e) {
			Utils.cleanup(revReader);
			Utils.cleanup(revValiWriter);
		}
	
	}

	public static void doStatistics(File inputFile) {
		if (! Utils.exists(inputFile)) {
			Utils.err(String.format("File doesn't exist. [%s]",
					inputFile == null ? "null" : inputFile.getAbsolutePath()));
			return;
		}
		BufferedReader reader = null;
		try {
			reader           = Utils.createBufferedReader(inputFile);
			String line      = null;
			int reviewNum    = 0;
			int avgTermNum   = 0;
			int minTermNum   = Integer.MAX_VALUE;
			int maxTermNum   = 0;
			double avgRating = 0.0;
			while ((line = reader.readLine()) != null) {
				if ((line = line.trim()).equals("")) {
					Utils.err("There is empty line.");
					continue;
				}

				String text     = "";
				int rating      = 0;
				int s    = line.indexOf(" ");
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
					e.printStackTrace();
				}
	
				int tokenNum = text.trim().split(AIRConstants.SPACER_SPACE).length;
				reviewNum ++;
				avgTermNum += tokenNum;
				avgRating  += rating;
				if (tokenNum > maxTermNum) {
					maxTermNum = tokenNum;
				}
				if (tokenNum < minTermNum) {
					minTermNum = tokenNum;
				}
			}
			Utils.println("============ Data Info ===============");
			Utils.println("ReviewNum      = " + reviewNum);
			Utils.println("AverageTermNum = " + (avgTermNum + 0.0) / reviewNum);
			Utils.println("MaxTermNum     = " + maxTermNum);
			Utils.println("MinTermNum     = " + minTermNum);
			Utils.println("AverageRating  = " + (avgRating / (reviewNum + 0.0)));
			Utils.println("======================================");
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	public static ArrayList<String> getReviewIds(File reviewInputFile,
						boolean addRepeatReviewId) {
		if (! Utils.exists(reviewInputFile)) {
			Utils.err(String.format("ReviewInputFile doesn't exisit.[%s]",
					reviewInputFile == null ?
						null : reviewInputFile.getAbsolutePath()));
			return null;
		}

		BufferedReader reader = null;
		try {
			reader          = Utils.createBufferedReader(reviewInputFile);
			String line     = null;
			int reviewCount = 0;
			ArrayList<String> reviewIds = new ArrayList<String>();
			while ((line = reader.readLine()) != null) {
				if (Utils.isEmpty((line = line.trim()))) continue;

				String reviewId = line.split(" ")[0];
				if (! reviewIds.contains(reviewId)) {
					reviewIds.add(reviewId);
				} else {
					if (addRepeatReviewId) reviewIds.add(reviewId);
					Utils.err(String.format(
							"Reivew [%s] has existed.",
							reviewId));
				}
				reviewCount ++;
			}

			Utils.println("--------- Loading Review Ids ---------");
			Utils.println("ReviewCount    = " + reviewCount + "; " + reviewInputFile.getAbsolutePath());
			Utils.println("CollectedCount = " + reviewIds.size() + "; " + reviewInputFile.getAbsolutePath());
			Utils.println("--------------------------------------");

			return reviewIds;
		} catch (IOException e) {
			e.printStackTrace();
		} finally {
			Utils.cleanup(reader);
		}

		return null;
	}

	public static void splitData(File inputFile, File trainOutputFile,
			File validationOutputFile, File testOutputFile,
			float validationRatio, float testRatio,
			File groundtruthScoreFile, File trainGroundtruthScoreOutputFile,
			File validationGroundtruthScoreOutputFile,
			File testGroundtruthScoreOutputFile,
			boolean readSplitIndexFromFile, File splitIndexFile)
					throws ToolException {
		if (!Utils.exists(inputFile) || (validationRatio > 0 &&
				validationOutputFile == null) ||
				(testRatio > 0 && testOutputFile == null)) {
			Utils.err("Invalid Input in splitData method.");
			return;
		}

		BufferedReader inputReader           = null;
		BufferedReader scoreReader           = null;
		BufferedWriter trainWriter           = null;
		BufferedWriter validationWriter      = null;
		BufferedWriter testWriter            = null;
		BufferedWriter scoreTrainWriter      = null;
		BufferedWriter scoreValidationWriter = null;
		BufferedWriter scoreTestWriter       = null;
		try {
			int reviewNum = 0;
			inputReader   = Utils.createBufferedReader(inputFile);
			String line   = null;
			// calculate review number
			while ((line = inputReader.readLine()) != null) {
				if (Utils.isEmpty((line = line.trim()))) continue;
				reviewNum ++;
			}
			Utils.cleanup(inputReader);
			if (reviewNum == 0) return;

			// prepare to split data
			int validationNum          = Math.round(reviewNum * validationRatio);
			int testNum                = Math.round(reviewNum * testRatio);
			int trainNum               = reviewNum - validationNum - testNum;
			final int validationMarker = 1;
			final int testMarker       = 2;
			int[] markers        = new int[reviewNum];
			if (readSplitIndexFromFile) {
				Utils.println("Load Split Index...");
				markers = Utils.load1IntArray(splitIndexFile,
						AIRConstants.SPACER_SPACE);
				if (Utils.isEmpty(markers))
					throw new ToolException("Split Index File is empty.");
				if (reviewNum != markers.length)
					throw new ToolException(String.format(
							"Review num in Split Index File doesn't match with input file.[#InputFile = %s, #SplitIndexFile = %s]",
							reviewNum, markers.length));
				validationNum = 0;
				testNum       = 0;
				trainNum      = 0;
				for (int i = 0; i < markers.length; i ++) {
					switch (markers[i]) {
						case validationMarker:
							validationNum ++;
							break;
						case testMarker:
							testNum ++;
							break;
						default:
							trainNum ++;
					}
				}
				validationRatio = (validationNum + 0.0f) /
							(reviewNum + 0.0f);
				testRatio       = (testNum + 0.0f) /
							(reviewNum + 0.0f);
			} else {
				for (int i = 0; i < validationNum; i ++) {
					markers[i] = validationMarker; // validation
				}
				for (int i = validationNum; i < validationNum + testNum; i ++) {
					markers[i] = testMarker; // test
				}
				Utils.println("Shuffling Split Index...");
				Utils.shuffle(markers);
				Utils.save(markers, AIRConstants.SPACER_SPACE,
						splitIndexFile , false, true);
			}

			// load scores
			String[] scores = new String[reviewNum];
			int reviewIndex = 0;
			if (groundtruthScoreFile != null) {
				scoreReader = Utils.createBufferedReader(groundtruthScoreFile);
				line        = null;
				while ((line = scoreReader.readLine()) != null) {
					if (Utils.isEmpty(line = line.trim())) continue;

					if (reviewIndex >= reviewNum) {
						Utils.cleanup(scoreReader);
						scoreReader = null;
						throw new ToolException(String.format(
								"Review number doesn't match.[#InReview=%s]",
								reviewNum));
					}
					scores[reviewIndex] = line;
					reviewIndex ++;
				}
				Utils.cleanup(scoreReader);
			}
			// prepare to store the splited data
			inputReader       = Utils.createBufferedReader(inputFile);
			trainWriter       = Utils.createBufferedWriter(trainOutputFile);
			validationWriter  = validationRatio > 0.0 ?
						Utils.createBufferedWriter(validationOutputFile) :
						null;
			testWriter        = testRatio > 0.0 ?
						Utils.createBufferedWriter(testOutputFile) :
						null;
			scoreTrainWriter  = (Utils.isEmpty(scores) || trainWriter == null ||
						trainGroundtruthScoreOutputFile == null) ?
						null : Utils.createBufferedWriter(trainGroundtruthScoreOutputFile);
			scoreValidationWriter = (Utils.isEmpty(scores) || validationWriter == null ||
						validationGroundtruthScoreOutputFile == null) ?
						null : Utils.createBufferedWriter(validationGroundtruthScoreOutputFile);
			scoreTestWriter       = (Utils.isEmpty(scores) || testWriter == null ||
						testGroundtruthScoreOutputFile == null) ?
						null : Utils.createBufferedWriter(testGroundtruthScoreOutputFile);
			line                  = null;
			reviewIndex           = 0;

			while ((line = inputReader.readLine()) != null) {
				if (Utils.isEmpty(line = line.trim())) continue;

				switch(markers[reviewIndex]) {
					case validationMarker:
						Utils.write(validationWriter, line, true);
						if (scoreValidationWriter != null) {
							Utils.write(scoreValidationWriter, scores[reviewIndex], true);
						}
						break;
					case testMarker:
						Utils.write(testWriter, line, true);
						if (scoreTestWriter != null) {
							Utils.write(scoreTestWriter, scores[reviewIndex], true);
						}
						break;
					default:
						Utils.write(trainWriter, line, true);
						if (scoreTrainWriter != null) {
							Utils.write(scoreTrainWriter, scores[reviewIndex], true);
						}
				}
				reviewIndex ++;
			}
			Utils.println("--------------------------------------");
			Utils.println("ReviewNum       = " + reviewNum);
			Utils.println("TrainNum        = " + trainNum);
			Utils.println("ValidationNum   = " + validationNum);
			Utils.println("TestNum         = " + testNum);
			Utils.println("ValidationRatio = " + validationRatio);
			Utils.println("TestRatio       = " + testRatio);
			Utils.println("--------------------------------------");
		} catch (IOException e) {
			e.printStackTrace();
		} finally {
			Utils.cleanup(inputReader);
			Utils.cleanup(scoreReader);
			Utils.cleanup(trainWriter);
			Utils.cleanup(validationWriter);
			Utils.cleanup(testWriter);
			Utils.cleanup(scoreTrainWriter);
			Utils.cleanup(scoreValidationWriter);
			Utils.cleanup(scoreTestWriter);
		}
		
	}

	/**
	 * Output the merged file. Appends the <code>inputFile2</code> to
	 * the end of <code>inpputFile1</code>.
	 *
	 * Note removes those empty lines.
	 *
	 * @param inputFile1		the input file that needs to merge
	 * @param inputFile2		the input file that appends to the
	 * 				<code>intputFile2</code>
	 * @param mergedOutputFile	the output file
	 */
	public static void mergeFiles(File inputFile1, File inputFile2,
						File mergedOutputFile) {
		if (! Utils.exists(inputFile1) || ! Utils.exists(inputFile2)) {
			Utils.err(String.format("File does not exisit.[InputFile1 = %s][InputFile2 = %s]",
					inputFile1 == null ? null : inputFile1.getAbsolutePath(),
					inputFile2 == null ? null : inputFile2.getAbsolutePath()));
			return;
		}
		BufferedReader reader1      = null;
		BufferedReader reader2      = null;
		BufferedWriter mergedWriter = null;
		try {
			Utils.println("------------- Merge Info -------------");

			reader1               = Utils.createBufferedReader(inputFile1);
			reader2               = Utils.createBufferedReader(inputFile2);
			mergedWriter          = Utils.createBufferedWriter(mergedOutputFile);
			int lineNum           = 0;
			int validLineNum      = 0;
			int totalValidLineNum = 0;
			String line           = null;
	
			while ((line = reader1.readLine()) != null) {
				lineNum ++;
				if (! Utils.isEmpty(line = line.trim())) {
					validLineNum ++;
					totalValidLineNum ++;
					Utils.write(mergedWriter, line, true);
				}
			}

			Utils.println("File 1 : Total Line Num      = " + lineNum);
			Utils.println("File 1 : Valid Line Num      = " + validLineNum);
			
			lineNum      = 0;
			validLineNum = 0;
			while ((line = reader2.readLine()) != null) {
				lineNum ++;
				if (! Utils.isEmpty(line)) {
					validLineNum ++;
					totalValidLineNum ++;
					Utils.write(mergedWriter, line, true);
				}
			}

			Utils.println("File 2 : Total Line Num      = " + lineNum);
			Utils.println("File 2 : Valid Line Num      = " + validLineNum);
			Utils.println("Merged File : Valid Line Num = " + totalValidLineNum );

		} catch (IOException e) {
			e.printStackTrace();
		} finally {
			Utils.cleanup(reader1);
			Utils.cleanup(reader2);
			Utils.cleanup(mergedWriter);
			Utils.println("--------------------------------------");
		}
	}

	public static boolean containLetter(String s, int minLetterNum) {
		if (Utils.isEmpty(s)) return false;
		
		int count = 0;
		for (int i = 0; i < s.length(); i ++) {
			char ch = s.charAt(i);
			if ((ch >= 'a' && ch <= 'z') ||
					(ch >= 'A' && ch <= 'z')) {
				count ++;
			}
		}
		if (count >= minLetterNum) return true;
		return false;
	}

	public static boolean isValidText(String text, char[] specChars,
						int repeatCharNum) {
		if (Utils.isEmpty(text)) return false;

		String[] words = TextUtils.splitWords((text = text.trim()),
					specChars).split(" ");
		for (String word : words) {
			if (! Utils.isEnglish(word) || TextUtils
					.isOutlierWord(word, repeatCharNum)) {
				return false;
			}
		}

		return true;
	}

	/*
	 * The following methods is to handle bigram case.
	 */

	public static void handleBigramWords(File inputFile, File outputFile,
					File biWordsPmiStatOutputFile,
					int pruneWordNumThreshold,
					double pmiThreshold,
					boolean isKeepUnigram,
					ArrayList<String>[] keywords) {
		if (! Utils.exists(inputFile)) {
			Utils.err("Empty inputFile.");
			return;
		}

		int[] sentenceCount                           = new int[1];
		HashMap<String, Integer> unitWordSenNumMap    = new HashMap<String, Integer>();
		HashMap<NgramTuple, Integer> biWordSenNumMap  = new HashMap<NgramTuple, Integer>();
		HashMap<NgramTuple, Double> biWordsPmiStatMap = new HashMap<NgramTuple, Double>(); // just for debug

		calculateBigramStats(inputFile, sentenceCount,
				unitWordSenNumMap, biWordSenNumMap);

		Utils.println("Sentence Num  = " + sentenceCount[0]);
		Utils.println("----- Before Pruning ----");
		Utils.println("Unit Word Num = " + unitWordSenNumMap.size());
		Utils.println("Biwords   Num = " + biWordSenNumMap.size());
		Utils.println("-------------------------");

		String[] unitWords = unitWordSenNumMap.keySet().toArray(new String[0]);
		for (String word : unitWords) {
			if (unitWordSenNumMap.get(word) <= pruneWordNumThreshold) {
				unitWordSenNumMap.remove(word);
			}
		}
		NgramTuple[] biwords = biWordSenNumMap.keySet().toArray(new NgramTuple[0]);
		for (NgramTuple tuple : biwords) {
			if (biWordSenNumMap.get(tuple) <= pruneWordNumThreshold) {
				biWordSenNumMap.remove(tuple);
			}
		}

		Utils.println("------ After Pruning ----");
		Utils.println("Unit Word Num = " + unitWordSenNumMap.size());
		Utils.println("Biwords   Num = " + biWordSenNumMap.size());
		Utils.println("-------------------------");

		BufferedReader reader           = null;
		BufferedWriter biWordsPmiWriter = null;
		BufferedWriter outputWriter     = null;
		try {
			reader           = Utils.createBufferedReader(inputFile);
			outputWriter     = Utils.createBufferedWriter(outputFile);
			String line      = null;
			while ((line = reader.readLine()) != null) {
				if (Utils.isEmpty((line = line.trim()))) continue;

				String revIndex = null;
				String text     = "";
				int rating      = 0;
				int s           = line.indexOf(" ");
				revIndex        = line.substring(0, s);
				line            = line.substring(s + 1);
				s               = line.indexOf(" ");
				try {
					if (s < 0) {
						rating = Integer.parseInt(line);
					} else {
						rating      = Integer.parseInt(line.substring(0, s));
						s           = line.indexOf(" ");
						text        = line.substring(s + 1);
					}
				} catch (NumberFormatException e) {
					e.printStackTrace();
				}

				StringTokenizer sentenceTokenizer = new StringTokenizer(text, ".");
				StringBuffer    textBuffer        = new StringBuffer();
				textBuffer.append(revIndex).append(" ")
						.append(rating).append(" ");
				while (sentenceTokenizer.hasMoreElements()) {
					String sentence = sentenceTokenizer.nextToken().trim();
					sentence        = handleBigramWordsInSentence(
								sentence,
								sentenceCount[0],
								unitWordSenNumMap,
								biWordSenNumMap,
								biWordsPmiStatMap,
								isKeepUnigram,
								keywords,
								pmiThreshold);
					//Utils.println(sentence);
					textBuffer.append(sentence.trim()).append(" ");
				}
				Utils.write(outputWriter, textBuffer.toString().trim(), true);
			}

			// biWordsPmiStatMap
			ArrayList<Object[]> biWordsPmiList = new ArrayList<Object[]>();
			for (NgramTuple tuple : biWordsPmiStatMap.keySet()) {
				biWordsPmiList.add(new Object[] {tuple, biWordsPmiStatMap.get(tuple)});
			}
			Collections.sort(biWordsPmiList, new Comparator<Object[]> () {
				@Override
				public int compare(Object[] obj1, Object[] obj2) {
					Double d1 = (Double) obj1[1];
					Double d2 = (Double) obj2[1];
					return d2.compareTo(d1);
				}
			});

			biWordsPmiWriter = Utils.createBufferedWriter(biWordsPmiStatOutputFile);
			for (Object[] object : biWordsPmiList) {
				NgramTuple tuple = (NgramTuple) object[0];
				Utils.write(biWordsPmiWriter, tuple.toSortedString("_") + "\t" + (Double)object[1], true);
			}
		} catch (IOException e) {
			e.printStackTrace();
		} finally {
			Utils.cleanup(reader);
			Utils.cleanup(outputWriter);
		}
	}

	/*
	 * The spacer between words in the sentence is " " (space).
	 */
	private static String handleBigramWordsInSentence(String sentence,
			int sentenceCount, HashMap<String, Integer> unitWordSenNumMap,
			HashMap<NgramTuple, Integer> biWordSenNumMap,
			HashMap<NgramTuple, Double> biWordsPmiStatMap,
			boolean isKeepUnigram,
			ArrayList<String>[] keywords,
					double pmiThreshold) {
		String[] words = null;
		if (Utils.isEmpty((sentence = sentence.trim())) ||
				((words = sentence.split(" ")).length <= 1)) return sentence;

		ArrayList<Object[]> biwordsList = getAllBiwords(words,
					sentenceCount, unitWordSenNumMap,
					biWordSenNumMap, pmiThreshold);
		if (Utils.isEmpty(biwordsList)) return sentence;

		// sort according pmi value
		Collections.sort(biwordsList, new Comparator<Object[]>() {
			@Override
			public int compare(Object[] obj1, Object[] obj2) {
				Double pmi1 = (Double) obj1[1];
				Double pmi2 = (Double) obj2[1];
				return pmi2.compareTo(pmi1);
			}
		});

		int len = biwordsList.size();
		for (int i = 0; i < len; i ++) {
			Object[] object = biwordsList.get(i);
			int word1Index  = (Integer)object[2];
			int word2Index  = (Integer)object[3];

			int restElemLen = biwordsList.size();
			for (int j = i + 1; j < restElemLen; j ++) {
				Object[] subElemObject = biwordsList.get(j);
				int subElem1Index      = (Integer)subElemObject[2];
				int subElem2Index      = (Integer)subElemObject[3];
				if (subElem1Index == word1Index ||
						subElem1Index == word2Index ||
						subElem2Index == word1Index ||
						subElem2Index == word2Index) {
					biwordsList.remove(j);
					j --;
					restElemLen --;
					len --;
				}
			}
		}
		if (biwordsList.isEmpty()) return sentence;

		// store biWordsPmiStatMap
		for (Object[] objects : biwordsList) {
			NgramTuple tuple = (NgramTuple)objects[0];
			if (! biWordsPmiStatMap.containsKey(tuple)) {
				biWordsPmiStatMap.put(tuple, (Double)objects[1]);
			}
		}

		// sorting according index 
		Collections.sort(biwordsList, new Comparator<Object[]>() {
			@Override
			public int compare(Object[] obj1, Object[] obj2) {
				Integer index1 = (Integer) obj1[2];
				Integer index2 = (Integer) obj2[2];
				return index1.compareTo(index2);
			}
		});

		boolean[] status     = new boolean[words.length];
		StringBuffer buffer  = new StringBuffer();
		int biwordsListIndex = 0;
		for (int i = 0; i < words.length; i ++) {
			if (status[i]) continue;
			if (biwordsListIndex < biwordsList.size()) {
				Object[] biwordsObject = biwordsList.get(biwordsListIndex);
				NgramTuple ngramTuple  = (NgramTuple) biwordsObject[0]; 
				int biwordsIndex1      = (Integer)biwordsObject[2];
				int biwordsIndex2      = (Integer)biwordsObject[3];
				if (biwordsIndex1 == i && ! isNgramTupleInDifferentTopic(
						ngramTuple, keywords)) {
					buffer.append(((NgramTuple)biwordsObject[0]).toSortedString("_"))
						.append(" ");
					biwordsListIndex ++;
					if (! isKeepUnigram) {
						status[biwordsIndex1] = true;
						status[biwordsIndex2] = true;
						continue;
					}
				}
			}
			buffer.append(words[i]).append(" ");
		}

		return buffer.toString().trim();
	}

	private static void calculateBigramStats(File inputFile,
			int[] sentenceCount,
			HashMap<String, Integer> unitWordSenNumMap,
			HashMap<NgramTuple, Integer> biWordSenNumMap) {
		if (sentenceCount == null || sentenceCount.length != 1 ||
				unitWordSenNumMap == null ||
					biWordSenNumMap == null) {
			Utils.err("Error arguments in calculateBigramStats method.");
			return;
		}

		BufferedReader reader = null;
		try {
			reader           = Utils.createBufferedReader(inputFile);
			String line      = null;
			sentenceCount[0] = 0;
			while ((line = reader.readLine()) != null) {
				if (Utils.isEmpty((line = line.trim()))) continue;

				String text     = "";
				int rating      = 0;
				int s           = line.indexOf(" ");
				line            = line.substring(s + 1);
				s               = line.indexOf(" ");
				try {
					if (s < 0) {
						rating = Integer.parseInt(line);
					} else {
						rating      = Integer.parseInt(line.substring(0, s));
						s           = line.indexOf(" ");
						text        = line.substring(s + 1);
					}
				} catch (NumberFormatException e) {
					e.printStackTrace();
				}

				StringTokenizer sentenceTokenizer = new StringTokenizer(text, ".");
				while (sentenceTokenizer.hasMoreElements()) {
					String sentence = sentenceTokenizer.nextToken().trim();
					if (Utils.isEmpty(sentence)) continue;

					sentenceCount[0] ++;
					calculateBigramStatsInSentence(sentence,
							unitWordSenNumMap,
							biWordSenNumMap);
				}
			}
		} catch (IOException e) {
			e.printStackTrace();
		} finally {
			Utils.cleanup(reader);
		}
	}

	/*
	 * The spacer between words in the sentence is " " (space).
	 */
	private static void calculateBigramStatsInSentence(String sentence,
			HashMap<String, Integer> unitWordSenNumMap,
			HashMap<NgramTuple, Integer> biWordSenNumMap) {
		if (Utils.isEmpty((sentence = sentence.trim()))) return;

		StringTokenizer tokenizer = new StringTokenizer(sentence, " ");
		HashSet<String> wordSet   = new HashSet<String>();

		while (tokenizer.hasMoreElements()) {
			String token = tokenizer.nextToken();

			wordSet.add(token);
		}

		for (String word : wordSet) {
			if (unitWordSenNumMap.containsKey(word)) {
				unitWordSenNumMap.put(word, unitWordSenNumMap.get(word) + 1);
			} else {
				unitWordSenNumMap.put(word, 1);
			}
		}

		NgramTuple[] biwordsArray = getNgramTuples(wordSet);
		if (biwordsArray != null) {
			for (NgramTuple biwords : biwordsArray) {
				if (biWordSenNumMap.containsKey(biwords)) {
					biWordSenNumMap.put(biwords, biWordSenNumMap.get(biwords) + 1);
				} else {
					biWordSenNumMap.put(biwords, 1);
				}
			}
		}
	}

	public static NgramTuple[] getNgramTuples(HashSet<String> wordSet) {
		if (wordSet == null || wordSet.isEmpty() ||
				wordSet.size() == 1) return null;

		int combinationCount = wordSet.size() * ( wordSet.size() - 1 ) / 2;
		NgramTuple[] results = new NgramTuple[combinationCount];
		String[] words       = wordSet.toArray(new String[0]);
		int index            = 0;

		Arrays.sort(words);

		for (int i = 0; i < words.length; i ++) {
			for (int j = i + 1; j < words.length; j ++) {
				results[index] = new NgramTuple(words[i], words[j]);
				index ++;
			}
		}

		return results;
	}

	/*
	 * Return all possible combinations that bigger or equal than
	 * pmiThreshold. The element in the returned list is an array,
	 * which is as the format {NgramTuple, PMI, word1_index, word2_index},
	 * where word1_index < word2_index.
	 */
	private static ArrayList<Object[]> getAllBiwords(String[] words,
				int sentenceCount,
				HashMap<String, Integer> unitWordSenNumMap,
				HashMap<NgramTuple, Integer> biWordSenNumMap,
				double pmiThreshold) {
		if (words.length <= 1) return null;

		ArrayList<Object[]> results             = new ArrayList<Object[]>();
		HashMap<NgramTuple, Double> tuplePmiMap = new HashMap<NgramTuple, Double>();
		for (int i = 0; i < words.length; i ++) {
			for (int j = i + 1; j < words.length; j ++) {
				NgramTuple tuple = new NgramTuple(words[i], words[j]);
				double pmiHat    = -1.0 * Double.MAX_VALUE;
				if (tuplePmiMap.containsKey(tuple)) {
					pmiHat = tuplePmiMap.get(tuple);
				} else {
					pmiHat = calculatePMI(words[i], words[j],
						tuple, sentenceCount,
						unitWordSenNumMap, biWordSenNumMap);
					tuplePmiMap.put(tuple, pmiHat);
				}
				if (pmiHat < pmiThreshold) continue;

				Object[] arr = new Object[]{tuple, pmiHat, i, j};
				results.add(arr);
			}
		}

		return results;
	}

	/*                                  P(word1 ^ word2)
	 * PMI = P(word1 ^ word2) * log(----------------------)
	 *                                 P(word1) x P(word2)
	 */
	private static double calculatePMI(String word1, String word2,
				NgramTuple tuple,
				int sentenceCount,
				HashMap<String, Integer> unitWordSenNumMap,
				HashMap<NgramTuple, Integer> biWordSenNumMap) {
		if (! unitWordSenNumMap.containsKey(word1) ||
				! unitWordSenNumMap.containsKey(word2) ||
				! biWordSenNumMap.containsKey(tuple)) return -1.0 * Double.MAX_VALUE;

		int tupleCount = biWordSenNumMap.get(tuple);
		double word1Count = unitWordSenNumMap.get(word1);
		double word2Count = unitWordSenNumMap.get(word2);
		if (word1Count < word2Count) {
			word1Count = unitWordSenNumMap.get(word2);
			word2Count = unitWordSenNumMap.get(word1);
		}
		double d1      = (sentenceCount * 1.0) / (word1Count * 1.0);
		double senDivW = tupleCount * d1 / (word2Count * 1.0);
		double tPert   = tupleCount * 1.0 / (sentenceCount * 1.0);
		double pmi     = FastMath.log(senDivW) * tPert;

		if (Double.isNaN(pmi) || Double.isInfinite(pmi)) {
			Utils.err("tupleCount = " + tupleCount +
					" ; word1_count = " + unitWordSenNumMap.get(word1) +
					" ; word2_count = " + unitWordSenNumMap.get(word2) +
					" ; sentenceCount = " + sentenceCount);
		}

		return pmi;
	}

	/*
	 * If elements in the NgramTuple are in different topic, then
	 * we don't consider it as ngram.
	 * We consider bigram case here. 
	 */
	public static boolean isNgramTupleInDifferentTopic(NgramTuple ngramTuple,
					ArrayList<String>[] keywords) {
		if (Utils.isEmpty(keywords)) return false;

		HashSet<Integer> word1Index = new HashSet<Integer>();
		HashSet<Integer> word2Index = new HashSet<Integer>();
		String word1                = ngramTuple.getSortedNgramWords()[0];
		String word2                = ngramTuple.getSortedNgramWords()[1];

		for (int i = 0; i < keywords.length; i ++) {
			for (String keyword : keywords[i]) {
				if (word1.equals(keyword)) {
					word1Index.add(i);
					if (! word2Index.isEmpty()) {
						for (Integer index : word2Index) {
							if (i != index) return true;
						}
					}
				} else
				if (word2.equals(keyword)) {
					word2Index.add(i);
					if (! word1Index.isEmpty()) {
						for (Integer index : word1Index) {
							if (i != index) return true;
						}
					}
				}
			}
		}

		return false;
	}

	public static boolean isLongTailWord(String[] referWords, String s) {
		if (Utils.isEmpty(referWords) || Utils.isEmpty(s)) return false;

		for (String referWord : referWords) {
			if (isLongTailWord(referWord,
				referWord.charAt(referWord.length() - 1),s))
				return true;
		}
		return false;
	}

	public static boolean isLongTailWord(String referWord, char tail, String s) {
		if (Utils.isEmpty(s) || Utils.isEmpty(referWord) ||
				s.length() < referWord.length()) return false;

		int index = 0;
		for (; index < referWord.length(); index ++) {
			if (s.charAt(index) != referWord.charAt(index)) return false;
		}

		for (; index < s.length(); index ++) {
			if (s.charAt(index) != tail) return false;
		}

		return true;
	}
}
