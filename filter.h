#include <opencv2/core/mat.hpp>


int blur5x5(cv::Mat& src, cv::Mat& dst);

int sobolX3x3(cv::Mat& src, cv::Mat& dst);

int sobolY3x3(cv::Mat& src, cv::Mat& dst);

int sobol3x3Gradient(cv::Mat& src, cv::Mat& dst);