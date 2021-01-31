#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "filter.h"

int main() {
	cv::VideoCapture* capDev;
	capDev = new cv::VideoCapture(0);
	if (!capDev->isOpened()) {
		std::cout << "Can't open Video device" << std::endl;
		return -1;
	}

	// constructor initialization, what does this exactly do? What's the normal way of doing this?
	cv::Size refS((int)capDev->get(cv::CAP_PROP_FRAME_WIDTH), (int)capDev->get(cv::CAP_PROP_FRAME_HEIGHT));

	std::cout << "Expected size: " << refS.width << " " << refS.height << std::endl;

	cv::namedWindow("Video");
	cv::Mat frame;
	int scNum = 0;

	bool greyscale = false, mirrored = false, blur = false, sobelx = false, sobely = false;

	while (true) {
		*capDev >> frame;
		// capDev->read(frame);

		if (frame.empty()) {
			std::cout << "frame is empty" << std::endl;
			break;
		}

		if (greyscale) {
			cv::Mat tempFrame;
			cv::cvtColor(frame, tempFrame, cv::COLOR_BGR2GRAY);
			frame = tempFrame.clone();
		}

		if (mirrored) {
			cv::Mat tempFrame;
			cv::flip(frame, tempFrame, 1);
			frame = tempFrame.clone();
		}

		if (blur) {
			cv::Mat tempFrame(frame.rows, frame.cols, CV_16SC3);
			blur5x5(frame, tempFrame);
			tempFrame.convertTo(frame, CV_8UC3);
		}

		if (sobelx) {
			cv::Mat tempFrame(frame.rows, frame.cols, CV_16SC3);
			sobolX3x3(frame, tempFrame);
			cv::abs(tempFrame);
			tempFrame.convertTo(frame, CV_8UC3);
		}

		if (sobely) {
			cv::Mat tempFrame(frame.rows, frame.cols, CV_16SC3);
			sobolY3x3(frame, tempFrame);
			cv::abs(tempFrame);
			tempFrame.convertTo(frame, CV_8UC3);
		}

		cv::imshow("video", frame);
		auto k = cv::waitKey(1); // why does setting this to zero doesn't work?
		if (k == 'q') { break; }
		if (k == 's') {
			cv::imwrite("screencapture_" + std::to_string(scNum) + ".jpg", frame);
			scNum++;
		}
		if (k == 'g') { greyscale = !greyscale; }
		if (k == 'f') { mirrored = !mirrored; }
		if (k == 'b') { blur = !blur; }
		if (k == 'x') { sobelx = !sobelx; }
		if (k == 'y') { sobely = !sobely; }

	}

	delete capDev;
	return 0;
}
