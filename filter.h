#include <opencv2/core/mat.hpp>


int blur5x5(cv::Mat& src, cv::Mat& dst);

int sobolX3x3(cv::Mat& src, cv::Mat& dst);

int sobolY3x3(cv::Mat& src, cv::Mat& dst);

int magnitude(cv::Mat& sx, cv::Mat& sy, cv::Mat& dst);

int quantize(cv::Mat& src, cv::Mat& dst, int levels);

int blurQuantize(cv::Mat& src, cv::Mat& dst, int levels);

int threshold(cv::Mat& src, cv::Mat& dst, int magThreshold);

int cartoon(cv::Mat& src, cv::Mat& dst, int levels, int magThreshold);
