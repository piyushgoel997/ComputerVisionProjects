#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

#include "Segmentation.h"
#include "ThresholdAndClean.h"
#include "Features.h"

int main(int argc, char *argv[]) {
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
		threshold<cv::Vec3b>(frame, thresh, 0.6*255);

		// cleanup the video
		// TODO do grassfire cleaning instead for extension
		// cv::Mat temp(frame.size(), CV_8UC1);
		cv::Mat cleaned(frame.size(), CV_8UC1);
		// opening(thresh, temp, 8, 4);
		// closing(temp, cleaned, 8, 4);
		grassfireClean(thresh, cleaned, 3);

		// cv::Mat colored(frame.size(), CV_8UC3);
		// segmentAndColorRegions(cleaned, colored,8, 1);
		// cv::Mat labels;
		// cv::connectedComponents(cleaned, labels);
		// auto* listOfCoords = getListOfCoordsForEachRegion(labels);

		// for (auto regionCoords : *listOfCoords) {
		// 	std::vector<double> features;
		// 	getFeatures(regionCoords, features);
		// 	for (auto f : features) {
		// 		std::cout << f << " ";
		// 	}
		// 	std::cout << std::endl;
		// }
		
		// cv::imshow("original_video", frame);
		cv::imshow("thresholded_video", thresh);
		cv::imshow("cleaned", cleaned);
		// cv::imshow("segmented_video", colored);
		
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
