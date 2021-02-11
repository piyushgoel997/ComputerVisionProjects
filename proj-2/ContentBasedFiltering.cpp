#include "ContentBasedFiltering.h"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <queue>
#include <opencv2/imgcodecs.hpp>
#include "filters.h"


// MATCHER

bool Matcher::validImageExtn(std::string extension) {
	return extension == ".jpg" || extension == ".png" || extension == ".jpeg";
}

std::vector<std::string>* Matcher::getMatches(const std::string imgname, const int numMatches, DistanceMetric* metric) {
	cv::Mat img = cv::imread(databaseDir + imgname);
	void* targetFeature = featurizer.getFeature(img);
	std::priority_queue<std::pair<double, std::string>> pq;
	for (const auto& entry : std::filesystem::directory_iterator(featurizedDatabaseDir)) {
		std::cout << entry.path() << std::endl;
		if (imgname.substr(0, imgname.find_last_of('.')) == entry.path().filename().string()) { continue; }
		void* imgFeature = featurizer.loadFeatureFromFile(entry.path().string());
		pq.push(std::make_pair(featurizer.getDistance(targetFeature, imgFeature, metric),
		                       entry.path().filename().string()));
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

double ImageFeaturizer::getDistance(void* f, void* g, DistanceMetric* metric) {
	const auto f_ = *(std::vector<int>*)f;
	const auto g_ = *(std::vector<int>*)g;
	return metric->calculateDistance(f_, g_);
}

void ImageFeaturizer::saveAfterFeaturizing(const cv::Mat& img, const std::string filepath) {
	saveFeaturesToFile(getFeature(img), filepath);
}


double ImageFeaturizer::getDistance(void* f, void* g, DistanceMetric* metric, int* breakAt, double* weights,
                                    int numBreaks) {
	const auto f_ = *(std::vector<int>*)f;
	const auto g_ = *(std::vector<int>*)g;
	double totalDistance = 0;
	int startIdx = 0;
	for (int i = 0; i <= numBreaks; ++i) {
		int endIdx = i < numBreaks ? breakAt[i] : f_.size();
		const std::vector<int> curr_f(f_.begin() + startIdx, f_.begin() + endIdx);
		const std::vector<int> curr_g(g_.begin() + startIdx, g_.begin() + endIdx);
		const double dist = metric->calculateDistance(curr_f, curr_g);
		totalDistance += weights[i] * dist;
		startIdx = endIdx;
	}
	double wtSum = 0;
	for (int i = 0; i < numBreaks + 1; ++i) { wtSum += weights[i]; }
	return totalDistance / wtSum;
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
	int startEndIndex[4] = {0, img.rows, 0, img.cols};
	return getFeature(img, startEndIndex);
}

void* HistogramFeaturizer::getFeature(const cv::Mat& img, int startEndIndices[4]) {
	const int max = 256;
	auto* const feature = new std::vector<int>;
	int size = 1;
	for (int i = 0; i < 3; ++i) { if (mask[i]) { size *= max; } }
	for (int i = 0; i < size; ++i) { feature->push_back(0); }
	for (int i = startEndIndices[0]; i < startEndIndices[1]; ++i) {
		for (int j = startEndIndices[2]; j < startEndIndices[3]; ++j) {
			cv::Vec3b pixel = img.at<cv::Vec3b>(i, j);
			int multiplier = 1;
			int idx = 0;
			for (int k = 0; k < 3; ++k) {
				if (mask[k]) {
					idx += pixel[k] * multiplier;
					multiplier *= max;
				}
			}
			feature->at(idx) += 1;
		}
	}
	return feature;
}

void* RGHistogramFeaturizer::getFeature(const cv::Mat& img) {
	int startEndIndex[4] = {0, img.rows, 0, img.cols};
	return getFeature(img, startEndIndex);
}

void* RGHistogramFeaturizer::getFeature(const cv::Mat& img, int startEndIndices[4]) {
	auto* const feature = new std::vector<int>;
	int size = bucketSize * bucketSize;
	for (int i = 0; i < size; ++i) { feature->push_back(0); }
	for (int i = startEndIndices[0]; i < startEndIndices[1]; ++i) {
		for (int j = startEndIndices[2]; j < startEndIndices[3]; ++j) {
			cv::Vec3b pixel = img.at<cv::Vec3b>(i, j);
			const double p0 = pixel[0], p1 = pixel[1], p2 = pixel[2];
			const double sum = p0 + p1 + p2;
			const int r = MAX(0, (bucketSize - 1) * p2 / sum), g = MAX(0, (bucketSize - 1) * p1 / sum);
			feature->at(r * bucketSize + g) += 1;
		}
	}
	return feature;
}

void* AvgHistogramFeaturizer::getFeature(const cv::Mat& img) {
	int startEndIndex[4] = {0, img.rows, 0, img.cols};
	return getFeature(img, startEndIndex);
}

void* AvgHistogramFeaturizer::getFeature(const cv::Mat& img, int startEndIndices[4]) {
	// assumes the range is 1.
	auto* const feature = new std::vector<int>;
	int size = bucketSize;
	for (int i = 0; i < size; ++i) { feature->push_back(0); }
	for (int i = startEndIndices[0]; i < startEndIndices[1]; ++i) {
		for (int j = startEndIndices[2]; j < startEndIndices[3]; ++j) {
			cv::Vec3d pixel = img.at<cv::Vec3d>(i, j);
			const double p0 = pixel[0], p1 = pixel[1], p2 = pixel[2];
			const double avg = (p0 + p1 + p2)/3;
			int idx = avg * (bucketSize - 1);
			feature->at(idx) += 1;
		}
	}
	return feature;
}


void* TopBottomMultiRGHistogramFeaturizer::getFeature(const cv::Mat& img) {
	RGHistogramFeaturizer* hist = new RGHistogramFeaturizer(100);
	int sei1[4] = {0, img.rows / 2, 0, img.cols};
	auto* topFeature = (std::vector<int>*)hist->getFeature(img, sei1);
	int sei2[4] = {img.rows / 2, img.rows, 0, img.cols};
	auto* bottomFeature = (std::vector<int>*)hist->getFeature(img, sei2);

	std::vector<int>* feature = new std::vector<int>;
	for (int f : *topFeature) { feature->push_back(f); }
	for (int f : *bottomFeature) { feature->push_back(f); }
	delete bottomFeature;
	delete topFeature;
	delete hist;
	return feature;
}

double TopBottomMultiRGHistogramFeaturizer::getDistance(void* f, void* g, DistanceMetric* metric) {
	int breakAt[1] = {bucketSize * bucketSize};
	double weights[2] = {1.0, 1.0};
	return ImageFeaturizer::getDistance(f, g, metric, breakAt, weights, 1);
}

void* CenterFullMultiRGHistogramFeaturizer::getFeature(const cv::Mat& img) {
	RGHistogramFeaturizer* hist = new RGHistogramFeaturizer(bucketSize);
	auto* fullFeature = (std::vector<int>*)hist->getFeature(img);
	int sei[4] = {img.rows / 2 - 49, img.rows / 2 + 50, img.cols / 2 - 49, img.cols / 2 + 50};
	auto* centerFeature = (std::vector<int>*)hist->getFeature(img, sei);

	std::vector<int>* feature = new std::vector<int>;
	for (int f : *fullFeature) { feature->push_back(f); }
	for (int f : *centerFeature) { feature->push_back(f); }
	delete fullFeature;
	delete centerFeature;
	delete hist;
	return feature;
}

double CenterFullMultiRGHistogramFeaturizer::getDistance(void* f, void* g, DistanceMetric* metric) {
	int breakAt[1] = {bucketSize * bucketSize};
	double weights[2] = {1.0, 5.0};
	return ImageFeaturizer::getDistance(f, g, metric, breakAt, weights, 1);
}


void* RGHistogramAndSobelOrientationTextureFeaturizer::getFeature(const cv::Mat& img) {
	RGHistogramFeaturizer* hist = new RGHistogramFeaturizer(bucketSize);
	auto* fullFeature = (std::vector<int>*)hist->getFeature(img);
	cv::Mat sx(img.rows, img.cols, CV_16SC3);
	cv::Mat sy(img.rows, img.cols, CV_16SC3);
	sobolX3x3(img, sx);
	sobolY3x3(img, sy);
	cv::Mat orientation(img.rows, img.cols, CV_64FC3);
	sobolOrientation(sx, sy, orientation);
	// get an averaged histogram
	AvgHistogramFeaturizer* avg = new AvgHistogramFeaturizer(bucketSize);
	auto* texture = (std::vector<int>*)avg->getFeature(orientation);

	std::vector<int>* feature = new std::vector<int>;
	for (int f : *fullFeature) { feature->push_back(f); }
	for (int f : *texture) { feature->push_back(f); }

	delete fullFeature;
	delete texture;
	delete hist;
	delete avg;
	return feature;
}

double RGHistogramAndSobelOrientationTextureFeaturizer::getDistance(void* f, void* g, DistanceMetric* metric) {
	int breakAt[1] = { bucketSize * bucketSize };
	double weights[2] = { 1.0, 1.0 };
	return ImageFeaturizer::getDistance(f, g, metric, breakAt, weights, 1);
}



// DIFFERENT DISTANCE METRICS

std::vector<double>* DistanceMetric::normalizeVector(const std::vector<int>& vec, bool normalize) {
	auto normalized = new std::vector<double>;
	double sum = 0;
	for (int i : vec) { sum += i; }
	for (double i : vec) {
		if (normalize) { normalized->push_back(i / sum); }
		else { normalized->push_back(i); }
	}
	return normalized;
}


double EuclideanDistance::calculateDistance(const std::vector<int>& p, const std::vector<int>& q) {
	auto distance = 0.0;
	auto p_ = normalizeVector(p, normalize);
	auto q_ = normalizeVector(q, normalize);
	for (int i = 0; i < p_->size(); ++i) {
		auto a = p_->at(i), b = q_->at(i);
		distance += pow(a - b, 2);
	}
	delete p_;
	delete q_;
	return sqrt(distance);
}

double L1Norm::calculateDistance(const std::vector<int>& p, const std::vector<int>& q) {
	auto distance = 0.0;
	auto p_ = normalizeVector(p, normalize);
	auto q_ = normalizeVector(q, normalize);
	for (int i = 0; i < p_->size(); ++i) {
		auto a = p_->at(i), b = q_->at(i);
		distance += abs(a - b);
	}
	delete p_;
	delete q_;
	return distance;
}

double HammingDistance::calculateDistance(const std::vector<int>& p, const std::vector<int>& q) {
	auto distance = 0.0;
	for (int i = 0; i < p.size(); ++i) {
		auto a = p.at(i) > 0 ? 1 : 0, b = q.at(i) > 0 ? 1 : 0;
		distance += abs(a - b);
	}
	return distance;
}

double NegativeOfHistogramIntersection::calculateDistance(const std::vector<int>& p, const std::vector<int>& q) {
	auto intersection = 0.0;
	auto p_ = normalizeVector(p, normalize);
	auto q_ = normalizeVector(q, normalize);
	for (int i = 0; i < p.size(); ++i) {
		auto a = p_->at(i), b = q_->at(i);
		intersection += std::min(a, b);
	}
	delete p_;
	delete q_;
	return -intersection;
}
