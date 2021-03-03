#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

#include "Segmentation.h"
#include "ThresholdAndClean.h"
#include "Features.h"
#include "DataBase.hpp"

int main(int argc, char* argv[]) {
	cv::VideoCapture* capDev;
	capDev = new cv::VideoCapture(1);
	if (!capDev->isOpened()) {
		std::cout << "Can't open Video device" << std::endl;
		return -1;
	}

	cv::Size refS((int)capDev->get(cv::CAP_PROP_FRAME_WIDTH), (int)capDev->get(cv::CAP_PROP_FRAME_HEIGHT));

	std::cout << "Expected size: " << refS.width << " " << refS.height << std::endl;

	cv::Mat frame;
	int scNum = 0;
	std::string tr_label;
	DataBase db;
	std::vector<double> features;
	while (true) {
		features.clear();

		*capDev >> frame;
		// cv::resize(frame, frame, cv::Size(), 0.50, 0.50);
		// threshold the video
		cv::Mat thresh(frame.size(), CV_8UC1);
		threshold<cv::Vec3b>(frame, thresh, 0.4 * 255);

		// cleanup the video
		// cv::Mat temp(frame.size(), CV_8UC1);
		cv::Mat cleaned(frame.size(), CV_8UC1);
		// opening(thresh, temp, 8, 4);
		// closing(temp, cleaned, 8, 4);
		grassfireClean(thresh, cleaned, 3);

		Segmentation seg(&cleaned, 100, 1);
		auto* listOfCoords = new std::vector<std::vector<std::pair<int, int>>>;
		seg.getListOfCoordsForEachRegion(listOfCoords);

		cv::Mat colored(frame.size(), CV_8UC3);
		seg.colorRegions(colored);
		int count = 0;
		for (auto regionCoords : *listOfCoords) {
			if (regionCoords.size() < 1000 || (regionCoords.at(0).first==0&& regionCoords.at(0).second == 0)) { continue; }
			auto* bb = getFeatures(regionCoords, features);

			std::vector<cv::Point> points;
			for (auto [x,y] : *bb) {
				cv::Point p(x, y);
				points.push_back(p);
			}
			
			cv::line(colored, points.at(0), points.at(1), cv::Scalar(0, 255, 0), 2);
			cv::line(colored, points.at(1), points.at(2), cv::Scalar(0, 255, 0), 2);
			cv::line(colored, points.at(2), points.at(3), cv::Scalar(0, 255, 0), 2);
			cv::line(colored, points.at(3), points.at(0), cv::Scalar(0, 255, 0), 2);

			cv::line(colored, points.at(4), points.at(5), cv::Scalar(0, 255, 0), 2);

			cv::circle(colored, points.at(6), 2, cv::Scalar(0, 255, 255), 2);
			count++;
			std::cout << "size" << features.size() << std::endl;
		}
		char k = cv::waitKey(30);

		if (k == 'y') {
			std::cout << "Enter label Name:" << std::endl;
			std::cin >> tr_label;
			db.storeFeatureVectorInDB(features, tr_label);
		}
		if (k == 'e') {
			db.fileDB();

			std::cout << "Training mode finished!" << std::endl;
		}
		if (k == 'd') {
			db.getMatchFromDB(features, tr_label);
			std::cout << "Detected Label:" << tr_label << std::endl;
		}


		if (k == 'q') { break; }
		if (k == 's') {
			cv::imwrite("screencapture_" + std::to_string(scNum) + ".jpg", frame);
			scNum++;
		}
		//std::cout << "Regions" << count << std::endl;
		/*std::cout << std::endl;
		std::cout << std::endl;
		std::cout << std::endl;*/
		// cv::imshow("original_video", frame);
		// cv::imshow("thresholded_video", thresh);
		// cv::imshow("cleaned", cleaned);
		cv::imshow("segmented_video", colored);


		delete listOfCoords;
		k = 'a';
	}

	delete capDev;
	return 0;
}
