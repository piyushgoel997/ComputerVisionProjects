#pragma once
#include <filesystem>
#include <opencv2/core.hpp>

class DistanceMetric;
class ImageFeaturizer;

class Matcher {
public:
	Matcher(ImageFeaturizer& featurizer, std::string databaseDir, std::string featurizedDatabaseDir) :
		featurizer(featurizer), databaseDir(databaseDir), featurizedDatabaseDir(featurizedDatabaseDir) {};
	std::vector<std::string>* getMatches(const std::string imgname, const int numMatches, DistanceMetric* metric);
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
	void saveAfterFeaturizing(const cv::Mat& img, const std::string filepath);
	double getDistance(void* f, void* g, DistanceMetric* metric, int* breakAt, double* weights, int numBreaks);
	void saveFeaturesToFile(void* features, std::string filepath);
	void* loadFeatureFromFile(std::string filepath);
	virtual double getDistance(void* f, void* g, DistanceMetric* metric);
};

class BaselineFeaturizer : public ImageFeaturizer {
	// uses the pixel values of a 9x9 square in the middle.
public:
	void* getFeature(const cv::Mat& img) override;
};

class HistogramFeaturizer : public ImageFeaturizer {
	// uses the color histogram of the whole image. Use the mask to determine which color to use (1 to use a color and 0 to not).
	// mask -> 0 = blue, 1 = green, 2 = red.
public:
	HistogramFeaturizer(const int mask[3]) : mask(mask) {}
	void* getFeature(const cv::Mat& img) override;
	void* getFeature(const cv::Mat& img, int startEndIndices[4]);

	const int* mask;
};

class RGHistogramFeaturizer : public ImageFeaturizer {
	// uses the r (r/r+g+b), g (g/r+g+b) for 2-d histogram over the whole image.
public:
	RGHistogramFeaturizer(const int bucketSize) : bucketSize(bucketSize) {}
	void* getFeature(const cv::Mat& img) override;
	void* getFeature(const cv::Mat& img, int startEndIndices[4]);

	const int bucketSize;
};

class AvgHistogramFeaturizer : public ImageFeaturizer {
	// uses (r+g+b)/3 1-d histogram over the whole image.
public:
	AvgHistogramFeaturizer(const int bucketSize) : bucketSize(bucketSize) {}
	void* getFeature(const cv::Mat& img) override;
	void* getFeature(const cv::Mat& img, int startEndIndices[4]);

	const int bucketSize;
};

class TopBottomMultiRGHistogramFeaturizer : public ImageFeaturizer {
	// uses the r (r/r+g+b), g (g/r+g+b) for 2-d histogram over the top and bottom halves of the image separately.
public:
	TopBottomMultiRGHistogramFeaturizer(const int bucketSize) : bucketSize(bucketSize) {}
	void* getFeature(const cv::Mat& img) override;
	double getDistance(void* f, void* g, DistanceMetric* metric) override;

	const int bucketSize;
};

class CenterFullMultiRGHistogramFeaturizer : public ImageFeaturizer {
	// uses the r (r/r+g+b), g (g/r+g+b) for 2-d histogram over the central 100x100 pixels and the whole image separately.
public:
	CenterFullMultiRGHistogramFeaturizer(const int bucketSize) : bucketSize(bucketSize) {}
	void* getFeature(const cv::Mat& img) override;
	double getDistance(void* f, void* g, DistanceMetric* metric) override;

	const int bucketSize;
};

class RGHistogramAndSobelOrientationTextureFeaturizer : public ImageFeaturizer {
	// uses the r (r/r+g+b), g (g/r+g+b) for 2-d histogram over the whole image.
public:
	RGHistogramAndSobelOrientationTextureFeaturizer(const int bucketSize) : bucketSize(bucketSize) {}
	void* getFeature(const cv::Mat& img) override;
	double getDistance(void* f, void* g, DistanceMetric* metric) override;
	
	const int bucketSize;
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
