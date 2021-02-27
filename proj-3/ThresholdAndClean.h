#pragma once
#include <opencv2/core.hpp>

// bkg is black (0) and foreground is white (255)

/**
 * @brief thresholds (by making sure all color values are greater than the threshold) src and returns the result in dst. The background will be made black.
 * @tparam T has to be the type of src (of the type cv::Vec with size 3)
 * @param src the matrix to be threshold-ed (the matrix is of the type T)
 * @param dst the matrix which will contain the result (this matrix should always be of the type uchar)
 * @param t_up upper threshold
 * @param t_low lower threshold
*/
template <typename T>
static void threshold(cv::Mat& src, cv::Mat& dst, const double t_up, const double t_low = 0) {
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


// Code for cleaning

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
			else { dst.at<uchar>(i, j) = src.at<uchar>(i, j); }
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
static void opening(cv::Mat& src, cv::Mat& dst, int erosionConn, int dilationConn) {
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
static void closing(cv::Mat& src, cv::Mat& dst, int dilationConn, int erosionConn) {
	cv::Mat temp(src.size(), CV_8UC1);
	dilate(src, temp, dilationConn);
	erode(temp, dst, erosionConn);
}


// Grassfire transform - extn

static void grassfireTransform(cv::Mat& src, cv::Mat& dst, int c, int connectivity) {
	// pass one
	for (int i = 0; i < src.rows; ++i) {
		for (int j = 0; j < src.cols; ++j) {
			if (src.at<uchar>(i, j) == c) { dst.at<int>(i, j) = 0; }
			else {
				int m = INT32_MAX;
				if (i > 0) { m = std::min(m, dst.at<int>(i - 1, j) + 1); }
				else { m = std::min(m, c + 1); }
				if (j > 0) { m = std::min(m, dst.at<int>(i, j - 1) + 1); }
				else { m = std::min(m, c + 1); }

				if (connectivity == 8 && i > 0) {
					if (j > 0) { m = std::min(m, dst.at<int>(i - 1, j - 1) + 1); }
					if (j < src.cols - 1) { m = std::min(m, dst.at<int>(i - 1, j + 1) + 1); }
				}
				dst.at<int>(i, j) = m;
			}
		}
	}

	// pass 2
	for (int i = src.rows - 1; i > -1; --i) {
		for (int j = src.cols - 1; j > -1; --j) {
			if (src.at<uchar>(i, j) == c) { dst.at<int>(i, j) = 0; }
			else {
				int m = dst.at<int>(i, j);
				if (i < src.rows - 1) { m = std::min(m, dst.at<int>(i + 1, j) + 1); }
				else { m = std::min(m, c + 1); }
				if (j < src.cols - 1) { m = std::min(m, dst.at<int>(i, j + 1) + 1); }
				else { m = std::min(m, c + 1); }

				if (connectivity == 8 && i < src.rows - 1) {
					if (j < src.cols - 1) { m = std::min(m, dst.at<int>(i + 1, j + 1) + 1); }
					if (j > 0) { m = std::min(m, dst.at<int>(i + 1, j - 1) + 1); }
				}
				dst.at<int>(i, j) = m;
			}
		}
	}
}

static void grassfireOpen(cv::Mat& src, cv::Mat& dst, const int distToRemove, const int erosionConn = 4,
                          const int dilationConn = 8) {
	// erosion
	cv::Mat op(src.size(), CV_32SC1);
	cv::Mat temp(src.size(), CV_8UC1);
	grassfireTransform(src, op, 0, erosionConn);
	for (int i = 0; i < src.rows; ++i) {
		for (int j = 0; j < src.cols; ++j) {
			if (op.at<int>(i, j) <= distToRemove) { temp.at<uchar>(i, j) = 0; }
			else { temp.at<uchar>(i, j) = src.at<uchar>(i, j); }
		}
	}

	// dilation
	cv::Mat cl(src.size(), CV_32SC1);
	grassfireTransform(temp, cl, 255, dilationConn);
	for (int i = 0; i < src.rows; ++i) {
		for (int j = 0; j < src.cols; ++j) {
			if (cl.at<int>(i, j) <= distToRemove) { dst.at<uchar>(i, j) = 255; }
			else { dst.at<uchar>(i, j) = src.at<uchar>(i, j); }
		}
	}
}

static void grassfireClose(cv::Mat& src, cv::Mat& dst, const int distToRemove, const int dilationConn = 4,
                           const int erosionConn = 8) {
	// dilation
	cv::Mat cl(src.size(), CV_32SC1);
	cv::Mat temp(src.size(), CV_8UC1);
	grassfireTransform(src, cl, 255, dilationConn);
	for (int i = 0; i < src.rows; ++i) {
		for (int j = 0; j < src.cols; ++j) {
			if (cl.at<int>(i, j) <= distToRemove) { temp.at<uchar>(i, j) = 255; }
			else { temp.at<uchar>(i, j) = src.at<uchar>(i, j); }
		}
	}

	// erosion
	cv::Mat op(src.size(), CV_32SC1);
	grassfireTransform(src, op, 0, erosionConn);
	for (int i = 0; i < src.rows; ++i) {
		for (int j = 0; j < src.cols; ++j) {
			if (op.at<int>(i, j) <= distToRemove) { dst.at<uchar>(i, j) = 0; }
			else { dst.at<uchar>(i, j) = src.at<uchar>(i, j); }
		}
	}
}

/**
 * @brief Cleans the image by running one pass of grassfire transform for closing and opening. Treats outside the image as the background.
 * @param src the image to be cleaned
 * @param dst the cleaned image
 * @param distToRemove the distance value <= which would be eroded or dilated to clean
*/
static void grassfireClean(cv::Mat& src, cv::Mat& dst, const int distToRemove) {
	cv::Mat temp(src.size(), CV_8UC1);
	grassfireOpen(src, temp, distToRemove);
	grassfireClose(temp, dst, distToRemove);
}
