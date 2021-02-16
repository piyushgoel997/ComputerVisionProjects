#pragma once
#include <filesystem>
#include <opencv2/core.hpp>

class DistanceMetric;
class ImageFeaturizer;

class Matcher {
public:
	Matcher(ImageFeaturizer& featurizer, std::string databaseDir, std::string featurizedDatabaseDir) :
		featurizer(featurizer), databaseDir(databaseDir), featurizedDatabaseDir(featurizedDatabaseDir) {};
	std::vector<std::string>* getMatches(std::string imgname, int numMatches, DistanceMetric* metric);
	static bool validImageExtn(std::string extension);
	static std::string getFilenameFromPath(std::filesystem::path path);
	void featurizeAndSaveDataset();

private:
	ImageFeaturizer& featurizer;
	const std::string databaseDir;
	const std::string featurizedDatabaseDir;
};

class ImageFeaturizer {
public:
	virtual void* getFeature(const cv::Mat& img) = 0;
	void saveAfterFeaturizing(const cv::Mat& img, std::string filepath);
	double getDistance(void* f, void* g, DistanceMetric* metric, int* breakAt, double* weights, int numBreaks);
	void saveFeaturesToFile(void* features, std::string filepath);
	void* loadFeatureFromFile(std::string filepath);
	virtual double getDistance(void* f, void* g, DistanceMetric* metric);

	bool doubleVec = false;
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
	HistogramFeaturizer(const int mask[3]) : mask(mask) {}
	void* getFeature(const cv::Mat& img) override;
	void* getFeature(const cv::Mat& img, int startEndIndices[4]);

	const int* mask;
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
	CoOccurrenceMatrix(const int axis, const int distance) : axis(axis), distance(distance) { doubleVec = true; }
	void* getFeature(const cv::Mat& img) override;

private:
	const int axis, distance;
};

// extn
class RGCoOccFullFeaturizer : public ImageFeaturizer {
public:
	RGCoOccFullFeaturizer(const int axis, const int distance, const int bucketSize) : axis(axis), distance(distance),
		bucketSize(bucketSize) { doubleVec = true; }

	void* getFeature(const cv::Mat& img) override;
	double getDistance(void* f, void* g, DistanceMetric* metric) override;

private:
	const int axis, distance, bucketSize;
};


class DistanceMetric {
public:
	DistanceMetric(const bool normalize) : normalize(normalize) {}
	virtual double calculateDistance(const std::vector<int>& p, const std::vector<int>& q) = 0;
	static std::vector<double>* normalizeVector(const std::vector<int>& vec, bool normalize);

	const bool normalize;
};

class EuclideanDistance : public DistanceMetric {
	// sum of squared errors
public:
	EuclideanDistance(const bool normalize) : DistanceMetric(normalize) {}
	double calculateDistance(const std::vector<int>& p, const std::vector<int>& q) override;
};

class L1Norm : public DistanceMetric {
	// L-1 Norm
public:
	L1Norm(const bool normalize) : DistanceMetric(normalize) {}
	double calculateDistance(const std::vector<int>& p, const std::vector<int>& q) override;
};

class LNNorm : public DistanceMetric {
	// L-1 Norm
public:
	LNNorm(const bool normalize, const int N) : DistanceMetric(normalize), N(N) {}
	double calculateDistance(const std::vector<int>& p, const std::vector<int>& q) override;

private:
	const int N;
};

class HammingDistance : public DistanceMetric {
	// Hamming Distance (for histograms)
public:
	HammingDistance(const bool normalize) : DistanceMetric(false) {}
	double calculateDistance(const std::vector<int>& p, const std::vector<int>& q) override;
};

class NegativeOfHistogramIntersection : public DistanceMetric {
	// Histogram Distance (for histograms)
public:
	NegativeOfHistogramIntersection(const bool normalize) : DistanceMetric(normalize) {}
	double calculateDistance(const std::vector<int>& p, const std::vector<int>& q) override;
};

class Metric {
public:
	virtual double calculate(const std::vector<double>& p) = 0;
	static std::vector<double>* normalizeVector(const std::vector<int>& vec);
};

class Energy : public Metric {
public:
	double calculate(const std::vector<double>& p) override;
};

class Entropy : public Metric {
public:
	double calculate(const std::vector<double>& p) override;
};

class Contrast : public Metric {
public:
	double calculate(const std::vector<double>& p) override;
};

class Homogeneity : public Metric {
public:
	double calculate(const std::vector<double>& p) override;
};

class MaximumProbability : public Metric {
public:
	double calculate(const std::vector<double>& p) override;
};
