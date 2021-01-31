#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

#include <iostream>
#include <opencv2/imgproc.hpp>

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

int convolve(cv::Mat& a, cv::Mat& b, cv::Mat& result) {
	// convolution including the division by magnitude.
	for (auto i = 0; i < a.rows; ++i) {
		for (auto j = 0; j < a.cols; ++j) {
			if (i < b.rows / 2 || j < b.cols / 2 || i >= a.rows - b.rows / 2 || j >= a.cols - b.cols / 2) {
				result.at<cv::Vec3s>(i, j) = a.at<cv::Vec3s>(i, j);
				continue;
			}

			short g[] = {0, 0, 0};
			short den = 0;
			for (auto k = -b.rows / 2; k <= b.rows / 2; ++k) {
				for (auto l = -b.cols / 2; l <= b.cols / 2; ++l) {
					const auto h = b.at<short>(b.rows / 2 + k, b.cols / 2 + l);
					auto f = a.at<cv::Vec3s>(i - k, j - l);
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
	convolve(temp, onedfilter, dst);
	dst.convertTo(dst, CV_8UC3);
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
	cv::pow(prod, 0.5, dst);
	dst.convertTo(dst, CV_8UC3);
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
			for (int t = 0; t < 3; ++t) {
				d[t] = cv::saturate_cast<uchar>(contrast * s[t] + brightness);
			}
			dst.at<cv::Vec3b>(i, j) = d;
		}
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
// 	// cv::Mat sx(img.rows, img.cols, CV_16SC3);
// 	// cv::Mat sy(img.rows, img.cols, CV_16SC3);
// 	// sobolX3x3(img, sx);
// 	// sobolY3x3(img, sy);
// 	// magnitude(sx, sy, modified);
// 	// cv::abs(modified);
// 	// modified.convertTo(modified, CV_8UC3);
//
// 	// blurQuantize(img, modified, 15);
//
// 	// cartoon(img, modified, 15, 20);
//
// 	// bool mask[] = {true, true, true};
// 	// negative(img, modified, mask);
//
// 	adjustBrightnessContrast(img, modified, 3.0, -100);
// 	
//
// 	cv::imshow("modified", modified);
//
// 	while (true) {
// 		auto k = cv::waitKey(0);
// 		if (k == 'q') { return 0; }
// 		if (k == 's') { cv::imwrite("contrast3brightness-100.jpg", modified); }
// 	}
// }
