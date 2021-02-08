#include <opencv2/core/mat.hpp>

int greyscale(cv::Mat& src, cv::Mat& dst);

int blur5x5(cv::Mat& src, cv::Mat& dst);

int sobolX3x3(cv::Mat& src, cv::Mat& dst);

int sobolY3x3(cv::Mat& src, cv::Mat& dst);

int magnitude(cv::Mat& sx, cv::Mat& sy, cv::Mat& dst);

int quantize(cv::Mat& src, cv::Mat& dst, int levels);

int blurQuantize(cv::Mat& src, cv::Mat& dst, int levels);

int threshold(cv::Mat& src, cv::Mat& dst, int magThreshold);

int cartoon(cv::Mat& src, cv::Mat& dst, int levels, int magThreshold);

int negative(cv::Mat& src, cv::Mat& dst, bool mask[]);

// extension

int adjustBrightnessContrast(cv::Mat& src, cv::Mat& dst, double contrast, double brightness);

int laplacian(cv::Mat& src, cv::Mat& dst);

int combine(cv::Mat& src, cv::Mat& other, cv::Mat& dst, double ratio);

int sepia(cv::Mat& src, cv::Mat& dst);

int rotateACW(cv::Mat& src, cv::Mat& dst);

int rotateCW(cv::Mat& src, cv::Mat& dst);

int upsideDown(cv::Mat& src, cv::Mat& dst);

int mirror(cv::Mat& src, cv::Mat& dst);

int meanBlur(cv::Mat& src, cv::Mat& dst, int blurLevel);
