#pragma once
#include <opencv2/core.hpp>

/**
 * @brief thresholds (by making sure all color values are greater than the threshold) src and returns the result in dst.
 * @tparam T has to be of the type cv::Vec with size 3
 * @param src the matrix to be threshold-ed
 * @param dst the matrix which will contain the result
 * @param thresh the value at which to threshold
*/
template<typename T>
void threshold(cv::Mat& src, cv::Mat& dst, const double thresh) {
	for (auto i = 0; i < src.rows; ++i) {
		for (auto j = 0; j < src.cols; ++j) {
			if (src.at<T>(i, j)[0] > thresh && src.at<T>(i, j)[1] > thresh && src.at<T>(i, j)[2] > thresh) {
				dst.at<cv::Vec3b>(i, j) = cv::Vec3b(255, 255, 255);
			}
			else { dst.at<cv::Vec3b>(i, j) = cv::Vec3b(0, 0, 0); }
		}
	}
}