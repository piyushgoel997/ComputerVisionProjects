#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

#include "filters.h"

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
	     cartonize = false, lap = false, sep = false, updwn = false, cw = false, acw = false, invertR = false, invertG =
		     false, invertB = false;
	double contrast = 1, ratio = 0.0;
	int brightness = 0, mblur = 0;
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

		if (blur) {
			cv::Mat tempFrame(frame.rows, frame.cols, CV_8UC3);
			blur5x5(frame, tempFrame);
			frame = tempFrame.clone();
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
			cv::Mat tempFrame(frame.rows, frame.cols, CV_8UC3);
			magnitude(x, y, tempFrame);
			frame = tempFrame.clone();
		}

		if (bq15) {
			cv::Mat tempFrame(frame.rows, frame.cols, CV_8UC3);
			blurQuantize(frame, tempFrame, 15);
			frame = tempFrame.clone();
		}

		if (cartonize) {
			cv::Mat tempFrame(frame.rows, frame.cols, CV_8UC3);
			cartoon(frame, tempFrame, 15, 20);
			frame = tempFrame.clone();
		}

		if (invertR || invertG || invertB) {
			cv::Mat tempFrame(frame.rows, frame.cols, CV_8UC3);
			bool mask[] = {invertB, invertG, invertR};
			negative(frame, tempFrame, mask);
			frame = tempFrame.clone();
		}

		if (contrast != 1 || brightness != 0) {
			cv::Mat tempFrame(frame.rows, frame.cols, CV_8UC3);
			adjustBrightnessContrast(frame, tempFrame, contrast, brightness);
			frame = tempFrame.clone();
		}

		if (lap) {
			cv::Mat tempFrame(frame.rows, frame.cols, CV_8UC3);
			laplacian(frame, tempFrame);
			frame = tempFrame.clone();
		}

		if (ratio != 0) {
			cv::Mat tempFrame(frame.rows, frame.cols, CV_8UC3);
			combine(frame, other, tempFrame, ratio);
			frame = tempFrame.clone();
		}

		if (sep) {
			cv::Mat tempFrame(frame.rows, frame.cols, CV_8UC3);
			sepia(frame, tempFrame);
			frame = tempFrame.clone();
		}

		if (updwn) {
			cv::Mat tempFrame(frame.rows, frame.cols, CV_8UC3);
			upsideDown(frame, tempFrame);
			frame = tempFrame.clone();
		}

		if (mirrored) {
			cv::Mat tempFrame(frame.rows, frame.cols, CV_8UC3);
			mirror(frame, tempFrame);
			frame = tempFrame.clone();
		}

		if (cw) {
			cv::Mat tempFrame(frame.cols, frame.rows, CV_8UC3);
			rotateCW(frame, tempFrame);
			frame = tempFrame.clone();
		}

		if (acw) {
			cv::Mat tempFrame(frame.cols, frame.rows, CV_8UC3);
			rotateACW(frame, tempFrame);
			frame = tempFrame.clone();
		}

		if (mblur != 0) {
			cv::Mat tempFrame(frame.cols, frame.rows, CV_8UC3);
			meanBlur(frame, tempFrame, mblur);
			frame = tempFrame.clone();
		}

		cv::imshow("video", frame);
		auto k = cv::waitKey(10); // why does setting this to zero doesn't work?
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
		if (k == '7') { invertR = !invertR; }
		if (k == '8') { invertG = !invertG; }
		if (k == '9') { invertB = !invertB; }
		if (k == '1') { contrast = MAX(0, contrast-0.1); }
		if (k == '2') { contrast = MIN(4, contrast+0.1); }
		if (k == '3') { brightness = MAX(-260, brightness-10); }
		if (k == '4') { brightness = MIN(260, brightness+10); }
		if (k == 'p') { lap = !lap; }
		if (k == '5') { ratio = MAX(0, ratio-0.1); }
		if (k == '6') { ratio = MIN(1, ratio + 0.1); }
		if (k == 'e') { sep = !sep; }
		if (k == 'u') { updwn = !updwn; }
		if (k == 'a') { acw = !acw; }
		if (k == 'w') { cw = !cw; }
		if (k == 'z') { mblur = (mblur + 1) % 4; }
	}

	delete capDev;
	return 0;
}
