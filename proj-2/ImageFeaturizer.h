#pragma once
#include <opencv2/core/mat.hpp>

#include "Metrics.h"


class ImageFeaturizer {
public:
	virtual void* getFeature(const cv::Mat& img) = 0;
	void saveAfterFeaturizing(const cv::Mat& img, std::string filepath);
	double getDistance(void* f, void* g, DistanceMetric* metric, int* breakAt, double* weights, int numBreaks);
	void saveFeaturesToFile(void* features, std::string filepath);
	void* loadFeatureFromFile(std::string filepath);
	virtual double getDistance(void* f, void* g, DistanceMetric* metric);
	virtual void beforeFinishSaving(std::string featurizedDatabaseDir) {}
};

// task 1
class BaselineFeaturizer : public ImageFeaturizer {
	// uses the pixel values of a 9x9 square in the middle.
public:
	void* getFeature(const cv::Mat& img) override;
};

// extn
class HistogramFeaturizer : public ImageFeaturizer {
	// uses the color histogram of the whole image. Use the mask to determine which color to use (1 to use a color and 0 to not).
	// mask -> 0 = blue, 1 = green, 2 = red.
public:
	HistogramFeaturizer(const int mask[3]) {
		for (int i = 0; i < 3; ++i) {
			this->mask[i] = mask[i];
		}
	}
	void* getFeature(const cv::Mat& img) override;
	void* getFeature(const cv::Mat& img, int startEndIndices[4]);

	int mask[3];
};

// task 2
class RGHistogramFeaturizer : public ImageFeaturizer {
	// uses the r (r/r+g+b), g (g/r+g+b) for 2-d histogram over the whole image.
public:
	RGHistogramFeaturizer(const int bucketSize) : bucketSize(bucketSize) {}
	void* getFeature(const cv::Mat& img) override;
	void* getFeature(const cv::Mat& img, int startEndIndices[4]);

	const int bucketSize;
};

// extn
class AvgHistogramFeaturizer : public ImageFeaturizer {
	// uses (r+g+b)/3 1-d histogram over the whole image.
public:
	AvgHistogramFeaturizer(const int bucketSize) : bucketSize(bucketSize) {}
	void* getFeature(const cv::Mat& img) override;
	void* getFeature(const cv::Mat& img, int startEndIndices[4]);

	const int bucketSize;
};

// extn
class TopBottomMultiRGHistogramFeaturizer : public ImageFeaturizer {
	// uses the r (r/r+g+b), g (g/r+g+b) for 2-d histogram over the top and bottom halves of the image separately.
public:
	TopBottomMultiRGHistogramFeaturizer(const int bucketSize) : bucketSize(bucketSize) {}
	void* getFeature(const cv::Mat& img) override;
	double getDistance(void* f, void* g, DistanceMetric* metric) override;

	const int bucketSize;
};

// task 3
class CenterFullMultiRGHistogramFeaturizer : public ImageFeaturizer {
	// uses the r (r/r+g+b), g (g/r+g+b) for 2-d histogram over the central 100x100 pixels and the whole image separately.
public:
	CenterFullMultiRGHistogramFeaturizer(const int bucketSize) : bucketSize(bucketSize) {}
	void* getFeature(const cv::Mat& img) override;
	double getDistance(void* f, void* g, DistanceMetric* metric) override;

	const int bucketSize;
};

// task 4
class RGHistogramAndSobelOrientationTextureFeaturizer : public ImageFeaturizer {
	// uses the r (r/r+g+b), g (g/r+g+b) for 2-d histogram over the whole image.
public:
	RGHistogramAndSobelOrientationTextureFeaturizer(const int bucketSize) : bucketSize(bucketSize) {}
	void* getFeature(const cv::Mat& img) override;
	double getDistance(void* f, void* g, DistanceMetric* metric) override;

	const int bucketSize;
};

// task 5
class RGFullAndCenterSobelTopAndBottomFullFeaturizer : public ImageFeaturizer {
public:
	RGFullAndCenterSobelTopAndBottomFullFeaturizer(const int bucketSize) : bucketSize(bucketSize) {}
	void* getFeature(const cv::Mat& img) override;
	double getDistance(void* f, void* g, DistanceMetric* metric) override;

	const int bucketSize;
};

// extn
class CoOccurrenceMatrix : public ImageFeaturizer {
public:
	CoOccurrenceMatrix(const int axis, const int distance, const int fileStartIdx = 0) : axis(axis), distance(distance),
		fileStartIdx(fileStartIdx) {
		mins = new std::vector<double>;
		maxs = new std::vector<double>;
		for (int i = 0; i < 5; ++i) {
			mins->push_back(DBL_MAX);
			maxs->push_back(0);
		}
	}

	void* getFeature(const cv::Mat& img) override;
	void beforeFinishSaving(std::string featurizedDatabaseDir) override;
	void updateMinsMaxs(double x, int i);

private:
	const int axis, distance, fileStartIdx;
	std::vector<double>* mins;
	std::vector<double>* maxs;
};

// extn
class RGCoOccFullFeaturizer : public ImageFeaturizer {
public:
	RGCoOccFullFeaturizer(const int axis, const int distance, const int bucketSize) : axis(axis), distance(distance),
		bucketSize(bucketSize) { }

	void* getFeature(const cv::Mat& img) override;
	double getDistance(void* f, void* g, DistanceMetric* metric) override;

private:
	const int axis, distance, bucketSize;
};


