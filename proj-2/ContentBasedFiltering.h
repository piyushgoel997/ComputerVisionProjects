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
	void saveFeaturesToFile(void* features, std::string filepath);
	void* loadFeatureFromFile(std::string filepath);
	double getDistance(void* f, void* g, DistanceMetric* metric);
};

class BaselineFeaturizer : public ImageFeaturizer {
	// uses the pixel values of a 9x9 square in the middle.
public:
	void* getFeature(const cv::Mat& img) override;
};

class HistogramFeaturizer : public ImageFeaturizer {
	// uses the 2-d color (blue and green color) histogram of the whole image as the histogram.
public:
	void* getFeature(const cv::Mat& img) override;
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

class HistogramDistance : public DistanceMetric {
	// Hamming Distance (for histograms)
public:
	HistogramDistance(const bool normalize) : DistanceMetric(normalize) {}
	double calculateDistance(const std::vector<int>& p, const std::vector<int>& q) override;
};
