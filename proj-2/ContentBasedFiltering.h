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
	ImageFeaturizer(DistanceMetric& metric) : metric(metric) {};
	virtual void* getFeature(const cv::Mat& img) = 0;
	void saveAfterFeaturizing(const cv::Mat& img, const std::string filepath);
	void saveFeaturesToFile(void* features, std::string filepath);
	void* loadFeatureFromFile(std::string filepath);
	double getDistance(void* f, void* g);

	DistanceMetric& metric;
};

class BaselineFeaturizer : public ImageFeaturizer {
public:
	BaselineFeaturizer(DistanceMetric& dm) : ImageFeaturizer(dm) {};
	void* getFeature(const cv::Mat& img) override;
};


class DistanceMetric {
public:
	virtual double calculateDistance(const std::vector<uchar>& p, const std::vector<uchar>& q) = 0;
};

class EuclideanDistance : public DistanceMetric{
public:
	double calculateDistance(const std::vector<uchar>& p, const std::vector<uchar>& q) override;
};
