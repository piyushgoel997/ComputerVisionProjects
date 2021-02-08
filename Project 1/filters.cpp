#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

#include <iostream>
#include <opencv2/imgproc.hpp>

#include "filters.h"

// int convolve(cv::Mat& a, cv::Mat& b, cv::Mat& result) {
// 	// convolution including the division by magnitude.
// 	for (int i = 0; i < a.rows; ++i) {
// 		for (int j = 0; j < a.cols; ++j) {
// 			if (i < b.rows / 2 || j < b.cols / 2 || i >= a.rows - b.rows / 2 || j >= a.cols - b.cols / 2) {
// 				result.at<cv::Vec3b>(i, j) = a.at<cv::Vec3b>(i, j);
// 				continue;
// 			}
//
// 			ushort g[] = {0, 0, 0};
// 			ushort den = 0;
// 			for (int k = -b.rows / 2; k <= b.rows / 2; ++k) {
// 				for (int l = -b.cols / 2; l <= b.cols / 2; ++l) {
// 					ushort h = b.at<uchar>(b.rows / 2 + k, b.cols / 2 + l);
// 					cv::Vec3b f = a.at<cv::Vec3b>(i - k, j - l);
// 					for (int t = 0; t < 3; ++t) { g[t] += f[t] * h; }
// 					den += h;
// 				}
// 			}
// 			cv::Vec3b g_;
// 			for (int t = 0; t < 3; ++t) { g_[t] = g[t] / den; }
// 			result.at<cv::Vec3b>(i, j) = g_;
// 		}
// 	}
// 	return 0;
// }

// int blur5x5(cv::Mat& src, cv::Mat& dst) {
// 	cv::Mat onedfilter(5, 1, CV_8UC1, 1);
// 	onedfilter.at<uchar>(1, 0) = 2;
// 	onedfilter.at<uchar>(2, 0) = 4;
// 	onedfilter.at<uchar>(3, 0) = 2;
// 	cv::Mat temp(src.rows, src.cols, CV_8UC3);
// 	convolve(src, onedfilter, temp);
// 	onedfilter = onedfilter.t();
// 	convolve(temp, onedfilter, dst);
// 	return 0;
// }

// int convolve(cv::Mat& a, cv::Mat& b, cv::Mat& result) {
// 	// convolution including the division by magnitude.
// 	for (auto i = 0; i < a.rows; ++i) {
// 		for (auto j = 0; j < a.cols; ++j) {
// 			if (i < b.rows / 2 || j < b.cols / 2 || i >= a.rows - b.rows / 2 || j >= a.cols - b.cols / 2) {
// 				result.at<cv::Vec3s>(i, j) = a.at<cv::Vec3s>(i, j);
// 				continue;
// 			}
//
// 			short g[] = {0, 0, 0};
// 			short den = 0;
// 			for (auto k = -b.rows / 2; k <= b.rows / 2; ++k) {
// 				for (auto l = -b.cols / 2; l <= b.cols / 2; ++l) {
// 					const auto h = b.at<short>(b.rows / 2 + k, b.cols / 2 + l);
// 					auto f = a.at<cv::Vec3s>(i - k, j - l);
// 					for (auto t = 0; t < 3; ++t) { g[t] += f[t] * h; }
// 					den += h;
// 				}
// 			}
// 			cv::Vec3s g_;
// 			for (auto t = 0; t < 3; ++t) {
// 				if (den != 0) { g_[t] = g[t] / den; }
// 				else { g_[t] = g[t]; }
// 			}
// 			result.at<cv::Vec3s>(i, j) = g_;
// 		}
// 	}
// 	return 0;
// }

int convolve(cv::Mat& a, cv::Mat& b, cv::Mat& result) {
	// convolution including the division by magnitude.
	for (auto i = 0; i < a.rows; ++i) {
		for (auto j = 0; j < a.cols; ++j) {
			short g[] = {0, 0, 0};
			short den = 0;
			for (auto k = -b.rows / 2; k <= b.rows / 2; ++k) {
				for (auto l = -b.cols / 2; l <= b.cols / 2; ++l) {
					const auto h = b.at<short>(b.rows / 2 + k, b.cols / 2 + l);
					auto f = a.at<cv::Vec3s>((i - k + a.rows) % a.rows, (j - l + a.cols) % a.cols);
					for (auto t = 0; t < 3; ++t) { g[t] += f[t] * h; }
					den += h;
				}
			}
			cv::Vec3s g_;
			for (auto t = 0; t < 3; ++t) {
				if (den != 0) { g_[t] = g[t] / den; }
				else { g_[t] = g[t]; }
			}
			result.at<cv::Vec3s>(i, j) = g_;
		}
	}
	return 0;
}

int blur5x5(cv::Mat& src, cv::Mat& dst) {
	cv::Mat onedfilter(5, 1, CV_16SC1, 1);
	onedfilter.at<short>(1, 0) = 2;
	onedfilter.at<short>(2, 0) = 4;
	onedfilter.at<short>(3, 0) = 2;
	cv::Mat temp(src.rows, src.cols, CV_16SC3);
	cv::Mat s;
	src.convertTo(s, CV_16SC3);
	convolve(s, onedfilter, temp);
	onedfilter = onedfilter.t();
	cv::Mat temp2(src.rows, src.cols, CV_16SC3);
	convolve(temp, onedfilter, temp2);
	temp2.convertTo(dst, CV_8UC3);
	return 0;
}

int sobolX3x3(cv::Mat& src, cv::Mat& dst) {
	cv::Mat a(3, 1, CV_16SC1, 1);
	a.at<short>(1, 0) = 2;
	cv::Mat b(1, 3, CV_16SC1, 1);
	b.at<short>(0, 0) = -1;
	b.at<short>(0, 1) = 0;
	cv::Mat temp(src.rows, src.cols, CV_16SC3);
	cv::Mat s;
	src.convertTo(s, CV_16SC3);
	convolve(s, a, temp);
	convolve(temp, b, dst);
	return 0;
}

int sobolY3x3(cv::Mat& src, cv::Mat& dst) {
	cv::Mat a(1, 3, CV_16SC1, 1);
	a.at<short>(0, 1) = 2;
	cv::Mat b(3, 1, CV_16SC1, 1);
	b.at<short>(2, 0) = -1;
	b.at<short>(1, 0) = 0;
	cv::Mat temp(src.rows, src.cols, CV_16SC3);
	cv::Mat s;
	src.convertTo(s, CV_16SC3);
	convolve(s, a, temp);
	convolve(temp, b, dst);
	return 0;
}

int magnitude(cv::Mat& sx, cv::Mat& sy, cv::Mat& dst) {
	sx.convertTo(sx, CV_64FC3);
	sy.convertTo(sy, CV_64FC3);
	cv::Mat prod = sx.mul(sx) + sy.mul(sy);
	cv::Mat temp2(sx.rows, sx.cols, CV_64FC3);
	cv::pow(prod, 0.5, temp2);
	temp2.convertTo(dst, CV_8UC3);
	return 0;
}

int quantize(cv::Mat& src, cv::Mat& dst, int levels) {
	uchar b = 255 / levels;
	for (int i = 0; i < src.rows; ++i) {
		for (int j = 0; j < src.cols; ++j) {
			cv::Vec3b s = src.at<cv::Vec3b>(i, j);
			cv::Vec3b d;
			for (int t = 0; t < 3; ++t) { d[t] = (s[t] / b) * b; }
			dst.at<cv::Vec3b>(i, j) = d;
		}
	}
	return 0;
}

int blurQuantize(cv::Mat& src, cv::Mat& dst, int levels) {
	cv::Mat blurred(src.rows, src.cols, CV_16SC3);
	blur5x5(src, blurred);
	// now quantize every channel
	quantize(blurred, dst, levels);
	return 0;
}

int threshold(cv::Mat& src, cv::Mat& sg, cv::Mat& dst, int magThreshold) {
	for (int i = 0; i < src.rows; ++i) {
		for (int j = 0; j < src.cols; ++j) {
			cv::Vec3b s = sg.at<cv::Vec3b>(i, j);
			cv::Vec3b d;
			int mag = 0;
			for (int t = 0; t < 3; ++t) { mag += s[t]; }
			if (mag / 3 > magThreshold) { d = cv::Vec3b(0, 0, 0); }
			else { d = src.at<cv::Vec3b>(i, j); }
			dst.at<cv::Vec3b>(i, j) = d;
		}
	}
	return 0;
}

int cartoon(cv::Mat& src, cv::Mat& dst, int levels, int magThreshold) {
	cv::Mat sx(src.rows, src.cols, CV_16SC3);
	cv::Mat sy(src.rows, src.cols, CV_16SC3);
	sobolX3x3(src, sx);
	sobolY3x3(src, sy);
	cv::Mat sg(src.rows, src.cols, CV_8UC3);
	magnitude(sx, sy, sg);
	cv::Mat temp(src.rows, src.cols, CV_8UC3);
	blurQuantize(src, temp, levels);
	threshold(temp, sg, dst, magThreshold);
	return 0;
}

int negative(cv::Mat& src, cv::Mat& dst, bool mask[]) {
	for (int i = 0; i < src.rows; ++i) {
		for (int j = 0; j < src.cols; ++j) {
			cv::Vec3b s = src.at<cv::Vec3b>(i, j);
			cv::Vec3b d;
			for (int t = 0; t < 3; ++t) {
				if (mask[t]) { d[t] = 255 - s[t]; }
				else { d[t] = s[t]; }
			}
			dst.at<cv::Vec3b>(i, j) = d;
		}
	}
	return 0;
}

int adjustBrightnessContrast(cv::Mat& src, cv::Mat& dst, double contrast, double brightness) {
	for (int i = 0; i < src.rows; ++i) {
		for (int j = 0; j < src.cols; ++j) {
			cv::Vec3b s = src.at<cv::Vec3b>(i, j);
			cv::Vec3b d;
			for (int t = 0; t < 3; ++t) { d[t] = cv::saturate_cast<uchar>(contrast * s[t] + brightness); }
			dst.at<cv::Vec3b>(i, j) = d;
		}
	}
	return 0;
}

int laplacian(cv::Mat& src, cv::Mat& dst) {
	cv::Mat filter(3, 3, CV_16SC1, -1);
	filter.at<short>(0, 0) = 0;
	filter.at<short>(0, 2) = 0;
	filter.at<short>(2, 0) = 0;
	filter.at<short>(2, 2) = 0;
	filter.at<short>(1, 1) = 4;
	cv::Mat s;
	src.convertTo(s, CV_16SC3);
	cv::Mat temp2(src.rows, src.cols, CV_16SC3);
	convolve(s, filter, temp2);
	temp2.convertTo(dst, CV_8UC3);
	return 0;
}

int combine(cv::Mat& src, cv::Mat& other, cv::Mat& dst, double ratio) {
	// ratio of the other image to mix
	assert(ratio >=0);
	assert(ratio <=1);
	for (int i = 0; i < src.rows; ++i) {
		for (int j = 0; j < src.cols; ++j) {
			cv::Vec3b s = src.at<cv::Vec3b>(i, j);
			cv::Vec3b o = other.at<cv::Vec3b>(i, j);
			cv::Vec3b d;
			for (int t = 0; t < 3; ++t) { d[t] = (1 - ratio) * s[t] + ratio * o[t]; }
			dst.at<cv::Vec3b>(i, j) = d;
		}
	}
	return 0;
}

int meanBlur(cv::Mat& src, cv::Mat& dst, int blurLevel) {
	assert(blurLevel >= 0);
	int l = 2 * blurLevel + 1;
	cv::Mat filter(l, l, CV_16SC1, 1);
	cv::Mat s;
	src.convertTo(s, CV_16SC3);
	cv::Mat temp2(src.rows, src.cols, CV_16SC3);
	convolve(s, filter, temp2);
	temp2.convertTo(dst, CV_8UC3);
	return 0;
}

int sepia(cv::Mat& src, cv::Mat& dst) {
	for (int i = 0; i < src.rows; ++i) {
		for (int j = 0; j < src.cols; ++j) {
			cv::Vec3b s = src.at<cv::Vec3b>(i, j);
			cv::Vec3b d;
			d[0] = cv::saturate_cast<uchar>((s[0] * .131) + (s[2] * .272) + (s[1] * .534));
			d[2] = cv::saturate_cast<uchar>((s[0] * .189) + (s[2] * .393) + (s[1] * .769));
			d[1] = cv::saturate_cast<uchar>((s[0] * .168) + (s[2] * .349) + (s[1] * .686));
			dst.at<cv::Vec3b>(i, j) = d;
		}
	}
	return 0;
}

int greyscale(cv::Mat& src, cv::Mat& dst) {
	for (int i = 0; i < src.rows; ++i) {
		for (int j = 0; j < src.cols; ++j) {
			cv::Vec3b s = src.at<cv::Vec3b>(i, j);
			cv::Vec3b d;
			short avg = (s[0] + s[1] + s[2]) / 3;
			d[0] = cv::saturate_cast<uchar>(avg);
			d[1] = cv::saturate_cast<uchar>(avg);
			d[2] = cv::saturate_cast<uchar>(avg);
			dst.at<cv::Vec3b>(i, j) = d;
		}
	}
	return 0;
}

int rotateACW(cv::Mat& src, cv::Mat& dst) {
	for (int i = 0; i < src.rows; ++i) {
		for (int j = 0; j < src.cols; ++j) { dst.at<cv::Vec3b>(src.cols - 1 - j, i) = src.at<cv::Vec3b>(i, j); }
	}
	return 0;
}

int rotateCW(cv::Mat& src, cv::Mat& dst) {
	for (int i = 0; i < src.rows; ++i) {
		for (int j = 0; j < src.cols; ++j) { dst.at<cv::Vec3b>(j, src.rows - 1 - i) = src.at<cv::Vec3b>(i, j); }
	}
	return 0;
}

int upsideDown(cv::Mat& src, cv::Mat& dst) {
	for (int i = 0; i < src.rows; ++i) {
		for (int j = 0; j < src.cols; ++j) { dst.at<cv::Vec3b>(src.rows - 1 - i, j) = src.at<cv::Vec3b>(i, j); }
	}
	return 0;
}

int mirror(cv::Mat& src, cv::Mat& dst) {
	for (int i = 0; i < src.rows; ++i) {
		for (int j = 0; j < src.cols; ++j) { dst.at<cv::Vec3b>(i, src.cols - 1 - j) = src.at<cv::Vec3b>(i, j); }
	}
	return 0;
}

// int main() {
//
// 	// const std::string image_path = cv::samples::findFile("starry_night.jpg");
// 	const std::string image_path = "my_image.jpg";
// 	cv::Mat img = cv::imread(image_path, cv::IMREAD_COLOR);
//
// 	if (img.empty()) {
// 		std::cout << "Couldn't read the image" << image_path << std::endl;
// 		return 1;
// 	}
//
//
// 	// cv::resize(img, img, cv::Size(10, 10));
// 	// cv::Mat modified(img.rows, img.cols, CV_16SC3);
//
// 	cv::Mat modified(img.rows, img.cols, CV_8UC3);
//
// 	// blur5x5(img, modified);
//
// 	// cv::Mat sx(img.rows, img.cols, CV_16SC3);
// 	// cv::Mat sy(img.rows, img.cols, CV_16SC3);
// 	// sobolX3x3(img, sx);
// 	// sobolY3x3(img, sy);
// 	// magnitude(sx, sy, modified);
// 	// modified.convertTo(modified, CV_8UC3);
//
// 	// blurQuantize(img, modified, 15);
//
// 	// cartoon(img, modified, 15, 20);
//
// 	// bool mask[] = {true, true, true};
// 	// negative(img, modified, mask);
//
// 	// adjustBrightnessContrast(img, modified, 3.0, -100);
//
// 	// laplacian(img, modified);
//
// 	// greyscale(img, modified);
//
// 	// mirror(img, modified);
//
// 	meanBlur(img, modified, 3);
//
// 	cv::imshow("modified", modified);
//
// 	while (true) {
// 		auto k = cv::waitKey(0);
// 		if (k == 'q') { return 0; }
// 		if (k == 's') { cv::imwrite("meanBlur3.jpg", modified); }
// 	}
// }
