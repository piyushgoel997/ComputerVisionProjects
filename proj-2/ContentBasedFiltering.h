#pragma once
#include <filesystem>
#include <opencv2/core.hpp>

class DistanceMetric;
class ImageFeaturizer;

class Matcher {
public:
	Matcher(ImageFeaturizer& featurizer, std::string databaseDir, std::string featurizedDatabaseDir) :
		featurizer(featurizer), databaseDir(databaseDir), featurizedDatabaseDir(featurizedDatabaseDir) {};
	std::vector<std::string>* getMatches(const std::string imgname, const int numMatches);
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
	ImageFeaturizer(DistanceMetric& metric) : metric(metric) {}
	virtual void* getFeature(const cv::Mat& img) = 0;
	void saveAfterFeaturizing(const cv::Mat& img, const std::string filepath);
	void saveFeaturesToFile(void* features, std::string filepath);
	void* loadFeatureFromFile(std::string filepath);
	double getDistance(void* f, void* g);

	DistanceMetric& metric;
};

class BaselineFeaturizer : public ImageFeaturizer {
	// uses the pixel values of a 9x9 square in the middle.
public:
	BaselineFeaturizer(DistanceMetric& dm) : ImageFeaturizer(dm) {}
	void* getFeature(const cv::Mat& img) override;
};

class HistogramFeaturizer : public ImageFeaturizer {
	// uses the 2-d color (blue and green color) histogram of the whole image as the histogram.
public:
	HistogramFeaturizer(DistanceMetric& dm) : ImageFeaturizer(dm) {}
	void* getFeature(const cv::Mat& img) override;
};


class DistanceMetric {
public:
	virtual double calculateDistance(const std::vector<int>& p, const std::vector<int>& q) = 0;
};

class EuclideanDistance : public DistanceMetric {
	// sum of squared distance
public:
	double calculateDistance(const std::vector<int>& p, const std::vector<int>& q) override;
};