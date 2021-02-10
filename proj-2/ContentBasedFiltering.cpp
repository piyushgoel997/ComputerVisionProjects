#include "ContentBasedFiltering.h"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <queue>
#include <opencv2/imgcodecs.hpp>


// MATCHER

bool Matcher::validImageExtn(std::string extension) {
	return extension == ".jpg" || extension == ".png" || extension == ".jpeg";
}

std::vector<std::string>* Matcher::getMatches(const std::string imgname, const int numMatches, DistanceMetric *metric) {
	cv::Mat img = cv::imread(databaseDir + imgname);
	void* targetFeature = featurizer.getFeature(img);
	std::priority_queue<std::pair<double, std::string>> pq;
	for (const auto& entry : std::filesystem::directory_iterator(featurizedDatabaseDir)) {
		std::cout << entry.path() << std::endl;
		if (imgname.substr(0, imgname.find_last_of('.')) == entry.path().filename().string()) { continue; }
		void* imgFeature = featurizer.loadFeatureFromFile(entry.path().string());
		pq.push(std::make_pair(featurizer.getDistance(targetFeature, imgFeature, metric), entry.path().filename().string()));
		delete imgFeature;
		if (pq.size() > numMatches) { pq.pop(); }
	}
	delete targetFeature;
	auto matches = new std::vector<std::string>;
	while (!pq.empty()) {
		matches->push_back(pq.top().second);
		pq.pop();
	}
	std::reverse(matches->begin(), matches->end());
	return matches;
}

std::string Matcher::getFilenameFromPath(std::filesystem::path path) {
	return path.filename().string().substr(0, path.filename().string().find_last_of('.'));
}


void Matcher::featurizeAndSaveDataset() {
	for (const auto& entry : std::filesystem::directory_iterator(databaseDir)) {
		const auto path = entry.path();

		if (!validImageExtn(path.extension().string())) {
			std::cout << "skipping " << path.string() << "\n";
			continue;
		}
		std::cout << "processing " << path.string() << "\n";
		cv::Mat img = cv::imread(path.string());
		std::string savepath = featurizedDatabaseDir + getFilenameFromPath(path);
		featurizer.saveAfterFeaturizing(img, savepath);
	}
}


// IMAGE FEATURIZER

void ImageFeaturizer::saveFeaturesToFile(void* features, std::string filepath) {
	std::vector<int>* featureVector = (std::vector<int>*)features;
	std::ofstream file;
	file.open(filepath);
	for (int f : *featureVector) { file << f << "\n"; }
	file.close();
	delete featureVector;
}

void* ImageFeaturizer::loadFeatureFromFile(std::string filepath) {
	std::ifstream file;
	file.open(filepath);
	std::string line;
	auto* const feature = new std::vector<int>;
	int x;
	while (file >> x) { (*feature).push_back(x); }
	return feature;
}

double ImageFeaturizer::getDistance(void* f, void* g, DistanceMetric *metric) {
	const auto f_ = *(std::vector<int>*)f;
	const auto g_ = *(std::vector<int>*)g;
	return metric->calculateDistance(f_, g_);
}

void ImageFeaturizer::saveAfterFeaturizing(const cv::Mat& img, const std::string filepath) {
	void* features = getFeature(img);
	saveFeaturesToFile(features, filepath);
}


// DIFFERENT FEATURIZERS

void* BaselineFeaturizer::getFeature(const cv::Mat& img) {
	if (img.rows < 9 || img.cols < 9) { throw std::exception("The image is too small."); }
	auto* const feature = new std::vector<int>;
	int startR = img.rows / 2 - 4, startC = img.cols / 2 - 4;
	for (int i = 0; i < 9; ++i) {
		for (int j = 0; j < 9; ++j) {
			cv::Vec3b pixel = img.at<cv::Vec3b>(startR + i, startC + j);
			for (int k = 0; k < 3; ++k) { (*feature).push_back(pixel[k]); }
		}
	}
	return feature;
}

void* HistogramFeaturizer::getFeature(const cv::Mat& img) {
	const int MAX = 256;
	int* arr = new int[MAX * MAX];
	for (int i = 0; i < MAX; ++i) { for (int j = 0; j < MAX; ++j) { arr[i + j * MAX] = 0; } }
	for (int i = 0; i < img.rows; ++i) {
		for (int j = 0; j < img.cols; ++j) {
			cv::Vec3b pixel = img.at<cv::Vec3b>(i, j);
			arr[pixel[0] + pixel[1] * MAX] += 1;
		}
	}
	// unfold the matrix into a 1-d feature array
	auto* const feature = new std::vector<int>;
	for (int i = 0; i < MAX; ++i) { for (int j = 0; j < MAX; ++j) { feature->push_back(arr[i + j * MAX]); } }
	delete[] arr;
	return feature;
}


// DIFFERENT DISTANCE METRICS

double EuclideanDistance::calculateDistance(const std::vector<int>& p, const std::vector<int>& q) {
	auto distance = 0.0;
	for (int i = 0; i < p.size(); ++i) {
		int a = p.at(i), b = q.at(i);
		distance += pow(a - b, 2);
	}
	return sqrt(distance);
}
