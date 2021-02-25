#pragma once
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

/**
 * @brief thresholds (by making sure all color values are greater than the threshold) src and returns the result in dst. The background will be made black.
 * @tparam T has to be of the type cv::Vec with size 3
 * @param src the matrix to be threshold-ed (the matrix is of the type T)
 * @param dst the matrix which will contain the result (this matrix should always be of the type uchar)
 * @param t_up upper threshold
 * @param t_low lower threshold
*/
template <typename T>
void threshold(cv::Mat& src, cv::Mat& dst, const double t_up, const double t_low = 0) {
	for (auto i = 0; i < src.rows; ++i) {
		for (auto j = 0; j < src.cols; ++j) {
			if ((src.at<T>(i, j)[0] > t_up && src.at<T>(i, j)[1] > t_up && src.at<T>(i, j)[2] > t_up) ||
				(src.at<T>(i, j)[0] < t_low && src.at<T>(i, j)[1] < t_low && src.at<T>(i, j)[2] < t_low)) {
				dst.at<uchar>(i, j) = 0;
			}
			else { dst.at<uchar>(i, j) = 255; }
		}
	}
}

// ref: https://docs.opencv.org/3.4/de/d01/samples_2cpp_2connected_components_8cpp-example.html
void segmentAndColorRegions(cv::Mat& src, cv::Mat& dst, int connectivity = 4) {
	// TODO implement own connected components algorithm
	cv::Mat labels;
	cv::Mat stats;
	cv::Mat centroids;
	int numLabels = cv::connectedComponentsWithStats(src, labels, stats, centroids);
	std::vector<cv::Vec3b> colors(numLabels);
	colors[0] = cv::Vec3b(0, 0, 0); // black for the bkg
	for (int i = 1; i < numLabels; ++i) {
		// TODO instead select top N
		if (stats.at<int>(i, cv::CC_STAT_AREA) < 2000) { colors[i] = cv::Vec3b(0, 0, 0); }
		else { colors[i] = cv::Vec3b((rand() & 255), (rand() & 255), (rand() & 255)); }
	}

	for (int i = 0; i < src.rows; ++i) {
		for (int j = 0; j < src.cols; ++j) { dst.at<cv::Vec3b>(i, j) = colors[labels.at<int>(i, j)]; }
	}
}
