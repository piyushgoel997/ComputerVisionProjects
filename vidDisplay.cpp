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

	cv::Mat other = cv::imread("my_image.jpg", cv::IMREAD_COLOR);

	// constructor initialization, what does this exactly do? What's the normal way of doing this?
	cv::Size refS((int)capDev->get(cv::CAP_PROP_FRAME_WIDTH), (int)capDev->get(cv::CAP_PROP_FRAME_HEIGHT));

	std::cout << "Expected size: " << refS.width << " " << refS.height << std::endl;

	cv::namedWindow("Video");
	cv::Mat frame;
	int scNum = 0;

	bool grey = false, mirrored = false, blur = false, sobelx = false, sobely = false, sobelgrad = false, bq15 = false,
	     cartonize = false, neg = false, lap = false, sep = false;
	double brightness = 0, contrast = 1, ratio = 0.0;
	// can't use grey with all.

	while (true) {
		*capDev >> frame;
		// capDev->read(frame);

		if (frame.empty()) {
			std::cout << "frame is empty" << std::endl;
			break;
		}

		if (grey) {
			cv::Mat tempFrame(frame.rows, frame.cols, CV_8UC3);
			greyscale(frame, tempFrame);
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

		if (sobelgrad) {
			cv::Mat x(frame.rows, frame.cols, CV_16SC3);
			sobolX3x3(frame, x);
			cv::Mat y(frame.rows, frame.cols, CV_16SC3);
			sobolY3x3(frame, y);
			cv::Mat tempFrame(frame.rows, frame.cols, CV_16SC3);
			magnitude(x, y, tempFrame);
			tempFrame.convertTo(frame, CV_8UC3);
		}

		if (bq15) {
			cv::Mat tempFrame(frame.rows, frame.cols, CV_8UC3);
			blurQuantize(frame, tempFrame, 15);
			tempFrame.convertTo(frame, CV_8UC3);
		}

		if (cartonize) {
			cv::Mat tempFrame(frame.rows, frame.cols, CV_8UC3);
			cartoon(frame, tempFrame, 15, 20);
			tempFrame.convertTo(frame, CV_8UC3);
		}

		if (neg) {
			cv::Mat tempFrame(frame.rows, frame.cols, CV_8UC3);
			bool mask[] = {true, true, true};
			negative(frame, tempFrame, mask);
			tempFrame.convertTo(frame, CV_8UC3);
		}

		if (contrast != 1 || brightness != 0) {
			cv::Mat tempFrame(frame.rows, frame.cols, CV_8UC3);
			adjustBrightnessContrast(frame, tempFrame, contrast, brightness);
			tempFrame.convertTo(frame, CV_8UC3);
		}

		if (lap) {
			cv::Mat tempFrame(frame.rows, frame.cols, CV_16SC3);
			laplacian(frame, tempFrame);
			tempFrame.convertTo(frame, CV_8UC3);
		}

		if (ratio != 0) {
			cv::Mat tempFrame(frame.rows, frame.cols, CV_8UC3);
			combine(frame, other, tempFrame, ratio);
			tempFrame.convertTo(frame, CV_8UC3);
		}

		if (sep) {
			cv::Mat tempFrame(frame.rows, frame.cols, CV_8UC3);
			sepia(frame, tempFrame);
			frame = tempFrame.clone();
		}

		cv::imshow("video", frame);
		auto k = cv::waitKey(1); // why does setting this to zero doesn't work?
		if (k == 'q') { break; }
		if (k == 's') {
			cv::imwrite("screencapture_" + std::to_string(scNum) + ".jpg", frame);
			scNum++;
		}
		if (k == 'g') { grey = !grey; }
		if (k == 'f') { mirrored = !mirrored; }
		if (k == 'b') { blur = !blur; }
		if (k == 'x') { sobelx = !sobelx; }
		if (k == 'y') { sobely = !sobely; }
		if (k == 'm') { sobelgrad = ! sobelgrad; }
		if (k == 'l') { bq15 = !bq15; }
		if (k == 'c') { cartonize = !cartonize; }
		if (k == 'n') { neg = !neg; }
		if (k == '1') { contrast -= 0.1; }
		if (k == '2') { contrast += 0.1; }
		if (k == '3') { brightness -= 10; }
		if (k == '4') { brightness += 10; }
		if (k == 'p') { lap = !lap; }
		if (k == '5') { ratio = MAX(0, ratio-0.1); }
		if (k == '6') { ratio = MIN(1, ratio + 0.1); }
		if (k == 'e') { sep = !sep; }
	}

	delete capDev;
	return 0;
}
