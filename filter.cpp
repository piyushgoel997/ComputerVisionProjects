#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

#include <iostream>
#include <opencv2/imgproc.hpp>

int convolve(cv::Mat& a, cv::Mat& b, cv::Mat& result) {
	// convolution including the division by magnitude.
	for (int i = 0; i < a.rows; ++i) {
		for (int j = 0; j < a.cols; ++j) {
			if (i < b.rows / 2 || j < b.cols / 2 || i >= a.rows - b.rows / 2 || j >= a.cols - b.cols / 2) {
				result.at<cv::Vec3b>(i, j) = a.at<cv::Vec3b>(i, j);
				continue;
			}

			ushort g[] = {0, 0, 0};
			ushort den = 0;
			for (int k = -b.rows / 2; k <= b.rows / 2; ++k) {
				for (int l = -b.cols / 2; l <= b.cols / 2; ++l) {
					ushort h = b.at<uchar>(b.rows / 2 + k, b.cols / 2 + l);
					cv::Vec3b f = a.at<cv::Vec3b>(i - k, j - l);
					for (int t = 0; t < 3; ++t) { g[t] += f[t] * h; }
					den += h;
				}
			}
			cv::Vec3b g_;
			for (int t = 0; t < 3; ++t) { g_[t] = g[t] / den; }
			result.at<cv::Vec3b>(i, j) = g_;
		}
	}
	return 0;
}

int blur5x5(cv::Mat& src, cv::Mat& dst) {
	cv::Mat onedfilter(5, 1, CV_8UC1, 1);
	onedfilter.at<uchar>(1, 0) = 2;
	onedfilter.at<uchar>(2, 0) = 4;
	onedfilter.at<uchar>(3, 0) = 2;
	cv::Mat temp(src.rows, src.cols, CV_8UC3);
	convolve(src, onedfilter, temp);
	onedfilter = onedfilter.t();
	convolve(temp, onedfilter, dst);
	return 0;
}


int main() {

	// const std::string image_path = cv::samples::findFile("starry_night.jpg");
	const std::string image_path = "my_image.jpg";
	cv::Mat img = cv::imread(image_path, cv::IMREAD_COLOR);

	if (img.empty()) {
		std::cout << "Couldn't read the image" << image_path << std::endl;
		return 1;
	}


	// cv::resize(img, img, img.size() / 2);
	// cv::resize(img, img, cv::Size(10, 10));
	cv::Mat blurred(img.rows, img.cols, CV_8UC3);

	blur5x5(img, blurred);

	cv::imshow("blurred", blurred);

	while (true) {
		auto k = cv::waitKey(0);
		if (k == 'q') { return 0; }
		if (k == 's') { cv::imwrite("blurred.jpg", blurred); }
	}
}
