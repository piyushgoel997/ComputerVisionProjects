#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

#include "Tasks.h"

int main() {
	cv::VideoCapture* capDev;
	capDev = new cv::VideoCapture(0);
	if (!capDev->isOpened()) {
		std::cout << "Can't open Video device" << std::endl;
		return -1;
	}

	cv::Size refS((int)capDev->get(cv::CAP_PROP_FRAME_WIDTH), (int)capDev->get(cv::CAP_PROP_FRAME_HEIGHT));

	std::cout << "Expected size: " << refS.width << " " << refS.height << std::endl;

	cv::Mat frame;
	int scNum = 0;


	while (true) {
		*capDev >> frame;

		// threshold the video
		// TODO do thresholding with the histogram and k-means
		cv::Mat thresh(frame.size(), CV_8UC1);
		threshold<cv::Vec3b>(frame, thresh, 200.0);

		cv::Mat segmented(frame.size(), CV_8UC3);
		segmentAndColorRegions(thresh, segmented);

		cv::imshow("original_video", frame);
		cv::imshow("thresholded_video", thresh);
		cv::imshow("segmented_video", segmented);
		auto k = cv::waitKey(10);
		if (k == 'q') { break; }
		if (k == 's') {
			cv::imwrite("screencapture_" + std::to_string(scNum) + ".jpg", frame);
			scNum++;
		}
	}

	delete capDev;
	return 0;
}
