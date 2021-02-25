#pragma once
#include <opencv2/core.hpp>

/**
 * @brief A helper which can perform both (erosion and dilation) depending on the value of the integer c (0 or 255).
 * @param src source matrix (type uchar)
 * @param dst destination matrix (type uchar)
 * @param connectivity the type of connectivity to use (should only have values 4 or 8)
 * @param c 0 for erosion and 255 for dilation
*/
static void helper(const cv::Mat& src, cv::Mat& dst, const int connectivity, const uchar c) {
	for (auto i = 0; i < src.rows; ++i) {
		for (int j = 0; j < src.cols; ++j) {
			bool doIt = false;
			if (src.at<uchar>(i, j) == c) { dst.at<uchar>(i, j) = src.at<uchar>(i, j); }
			// left side
			if (i > 0) {
				if (j > 0 && connectivity == 8 && src.at<uchar>(i - 1, j - 1) == c) { doIt = true; }
				if (src.at<uchar>(i - 1, j) == c) { doIt = true; }
				if (j < src.cols - 1 && connectivity == 8 && src.at<uchar>(i - 1, j + 1) == c) { doIt = true; }
			}
			// right side
			if (i < src.rows - 1) {
				if (j > 0 && connectivity == 8 && src.at<uchar>(i + 1, j - 1) == c) { doIt = true; }
				if (src.at<uchar>(i + 1, j) == c) { doIt = true; }
				if (j < src.cols - 1 && connectivity == 8 && src.at<uchar>(i + 1, j + 1) == c) { doIt = true; }
			}
			// top middle
			if (j > 0 && src.at<uchar>(i, j - 1) == c) { doIt = true; }
			// bottom middle
			if (j < src.cols - 1 && src.at<uchar>(i, j + 1) == c) { doIt = true; }
			// now erode/dilate if needed
			if (doIt) { dst.at<uchar>(i, j) = c; }
			else { dst.at<uchar>(i, j) = src.at<uchar>(i,j); }
		}
	}
}

/**
 * @brief Erodes the threshold-ed matrix
 * @param src source matrix (type uchar)
 * @param dst destination matrix (type uchar)
 * @param connectivity the type of connectivity to use (should only have values 4 or 8)
*/
static void erode(const cv::Mat& src, cv::Mat& dst, const int connectivity) { helper(src, dst, connectivity, 0); }

/**
 * @brief Dilates the threshold-ed matrix
 * @param src source matrix (type uchar)
 * @param dst destination matrix (type uchar)
 * @param connectivity the type of connectivity to use (should only have values 4 or 8)
*/
static void dilate(const cv::Mat& src, cv::Mat& dst, const int connectivity) { helper(src, dst, connectivity, 255); }

/**
 * @brief Opening clean up of a threshold-ed matrix by eroding first and then dilating.
 * @param src source matrix (type uchar)
 * @param dst destination matrix (type uchar)
 * @param erosionConn connectivity to use while eroding
 * @param dilationConn connectivity to use while dilating
*/
void opening(cv::Mat& src, cv::Mat& dst, int erosionConn, int dilationConn) {
	cv::Mat temp(src.size(), CV_8UC1);
	erode(src, temp, erosionConn);
	dilate(temp, dst, dilationConn);
}

/**
 * @brief Opening clean up of a threshold-ed matrix by dilating first and then eroding.
 * @param src source matrix (type uchar)
 * @param dst destination matrix (type uchar)
 * @param erosionConn connectivity to use while eroding
 * @param dilationConn connectivity to use while dilating
*/
void closing(cv::Mat& src, cv::Mat& dst, int dilationConn, int erosionConn) {
	cv::Mat temp(src.size(), CV_8UC1);
	dilate(src, temp, dilationConn);
	erode(temp, dst, erosionConn);
}


