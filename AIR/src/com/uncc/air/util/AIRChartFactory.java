package com.uncc.air.util;

import java.awt.Color;
import java.awt.Font;
import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.HashMap;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.ChartUtilities;
import org.jfree.chart.plot.CategoryPlot;
import org.jfree.chart.plot.XYPlot;
import org.jfree.chart.axis.CategoryAxis;
import org.jfree.chart.axis.ValueAxis;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.chart.renderer.category.BarRenderer;
import org.jfree.data.category.CategoryDataset;
import org.jfree.data.category.DefaultCategoryDataset;
import org.jfree.data.xy.XYDataset;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;
import org.jfree.chart.axis.CategoryLabelPositions;

import com.lhy.tool.util.Utils;

/**
 * @author Huayu Li
 */
public class AIRChartFactory {
	private static final Font PLAIN_12 = new Font("Calibri", Font.PLAIN, 12);
	private static final Font BOLD_14 = new Font("Calibri", Font.BOLD, 14);

	public static boolean saveHistogram(String title, String categoryAxisLabel,
					String valueAxisLabel,
					HashMap<String, Integer> domainValueMap,
					File imageFile, int width, int height) {

		return saveHistogram(title, categoryAxisLabel,
					valueAxisLabel,
					createDataset(domainValueMap),
					imageFile, width, height);

	}

	public static boolean saveHistogram(String title, String categoryAxisLabel,
				String valueAxisLabel, String[] domains,
				Integer[] values, File imageFile, int width, int height) {

		return saveHistogram(title, categoryAxisLabel,
				valueAxisLabel,
				createDataset(domains, values),
				imageFile, width, height);

	}

	public static boolean saveXYLineChart(String title,
				String categoryAxisLabel, String valueAxisLabel,
				double[][] data,
				File imageFile, int width, int height) {
		JFreeChart chart = createXYLineChart(title, categoryAxisLabel,
					valueAxisLabel, data);
		if (chart != null) {
			try {
				if (! imageFile.getParentFile().exists()) {
					imageFile.mkdirs();
				}
				ChartUtilities.saveChartAsJPEG(imageFile,
						chart, width, height);
			} catch (IOException e) {
				e.printStackTrace();
			}
		}

		return false;
	}

	private static boolean saveHistogram(String title, String categoryAxisLabel,
				String valueAxisLabel,
				CategoryDataset categoryDataset,
				File imageFile, int width, int height) {
		if (categoryDataset != null) {
			JFreeChart chart = createHistogramChart(title,
					categoryAxisLabel,
					valueAxisLabel, categoryDataset);
			try {
				if (! imageFile.getParentFile().exists()) {
					imageFile.getParentFile().mkdirs();
				}
				ChartUtilities.saveChartAsJPEG(imageFile,
					chart, width, height);
				return true;
			} catch (IOException e) {
				e.printStackTrace();
			}

		} else {
			Utils.err("Failed to draw histogram because of empty dataset.");
		}

		return false;
	}

	private static JFreeChart createHistogramChart(String title,
				String categoryAxisLabel,
				String valueAxisLabel,
				CategoryDataset categoryDataset) {

		if (categoryDataset == null) {
			return null;
		}

		JFreeChart chart = ChartFactory.createBarChart(title,
					categoryAxisLabel, valueAxisLabel,
					categoryDataset,
					PlotOrientation.VERTICAL,
					false, false, false);

		chart.setBackgroundPaint(Color.WHITE);
		CategoryPlot categoryPlot = chart.getCategoryPlot();
		categoryPlot.setBackgroundPaint(Color.WHITE);
		categoryPlot.setRangeGridlinePaint(Color.black);
		categoryPlot.setRangeGridlinesVisible(true);
		//categoryPlot.setDomainGridlinesVisible(true);

		CategoryAxis domainAxis = categoryPlot.getDomainAxis();
		domainAxis.setLabelFont(BOLD_14);
		domainAxis.setTickLabelFont(PLAIN_12);
		domainAxis.setCategoryLabelPositions(CategoryLabelPositions.UP_45);
		ValueAxis valueAxis = categoryPlot.getRangeAxis();
		valueAxis.setLabelFont(BOLD_14);
		valueAxis.setTickLabelFont(PLAIN_12);

		BarRenderer render = (BarRenderer)categoryPlot.getRenderer();
		render.setDrawBarOutline(false);
		render.setSeriesPaint(0, new Color(79, 129, 189));

		return chart;
	}

	private static JFreeChart createXYLineChart(String title,
				String categoryAxisLabel,
				String valueAxisLabel, double[][] data) {

		JFreeChart chart = ChartFactory.createXYLineChart(title,
					categoryAxisLabel, valueAxisLabel,
					createXYDataset(data),
					PlotOrientation.VERTICAL,
					false, false, false);

		double minY  = min(data, 1);
		double maxY  = max(data, 1);
		double iterY = (maxY - minY) / data.length;
		//double minX  = min(data, 0);
		//double maxX  = max(data, 0);
		//double iterX = (maxX - minX) / data.length;

		XYPlot XYPlot = chart.getXYPlot();
		ValueAxis domainAxis = XYPlot.getDomainAxis();
		ValueAxis valueAxis = XYPlot.getRangeAxis();
		//domainAxis.setRange(minX - 10 * iterX, maxX + 10 * iterX);
		valueAxis.setRange(minY - 10 * iterY, maxY + 10 * iterY);

		chart.setBackgroundPaint(Color.WHITE);
		
		XYPlot.setBackgroundPaint(Color.LIGHT_GRAY);

		
		domainAxis.setLabelFont(BOLD_14);
		domainAxis.setTickLabelFont(PLAIN_12);

		valueAxis.setLabelFont(BOLD_14);
		valueAxis.setTickLabelFont(PLAIN_12);

		return chart;
	}

	private static XYDataset createXYDataset(double[][] data) {
		if (Utils.isEmpty(data) || data[0].length < 2) {
			return null;
		}

		XYSeries xySeries = new XYSeries("");
		for (int i = 0; i < data.length; i ++) {
			xySeries.add(data[i][0], data[i][1]);
		}

		XYSeriesCollection xySC = new XYSeriesCollection();
		xySC.addSeries(xySeries);

		return xySC;
	}

	private static CategoryDataset createDataset(HashMap<String, Integer> domainValueMap) {
		DefaultCategoryDataset dataset = null;

		if (! Utils.isEmpty(domainValueMap)) {
			dataset       = new DefaultCategoryDataset();
			String[] keys = domainValueMap.keySet().toArray(new String[0]);
			Arrays.sort(keys);
			for (String key : keys) {
				dataset.addValue(domainValueMap.get(key), "Topic", key);
			}

		}

		return dataset;
	}

	private static CategoryDataset createDataset(String[] domains,
							Integer[] values) {
		DefaultCategoryDataset dataset = null;

		if (! Utils.isEmpty(domains) && ! Utils.isEmpty(values) &&
				domains.length == values.length) {
			dataset = new DefaultCategoryDataset();
			for (int i = 0; i < domains.length; i ++) {
				dataset.addValue(values[i], "Topic", domains[i]);
			}
		}

		return dataset;
	}

	private static double min(double[][] data, int columnIndex) {
		if (! Utils.isEmpty(data)) {
			double min = data[0][columnIndex];
			for (int i = 1; i < data.length; i ++) {
				if (data[i][1] < min) {
					min = data[i][columnIndex];
				}
			}
			return min;
		}
		return Integer.MIN_VALUE;
	}

	private static double max(double[][] data, int columnIndex) {
		if (! Utils.isEmpty(data)) {
			double max = data[0][columnIndex];
			for (int i = 1; i < data.length; i ++) {
				if (data[i][1] > max) {
					max = data[i][columnIndex];
				}
			}
			return max;
		}
		return Integer.MAX_VALUE;
	}
}
