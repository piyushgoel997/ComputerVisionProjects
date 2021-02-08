#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

#include <iostream>

// int main() {
//
// 	// const std::string image_path = cv::samples::findFile("starry_night.jpg");
// 	const std::string image_path = "my_image.jpg";
// 	const cv::Mat img = cv::imread(image_path, cv::IMREAD_COLOR);
//
// 	if (img.empty()) {
// 		std::cout << "Couldn't read the image" << image_path << std::endl;
// 		return 1;
// 	}
//
// 	cv::imshow("My image", img);
//
// 	while (true) {
// 		auto k = cv::waitKey(0);
// 		if (k == 'q') { return 0; }
// 	}
// }
