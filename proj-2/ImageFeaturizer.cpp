#include "ImageFeaturizer.h"

#include <filesystem>
#include <fstream>

#include "filters.h"

void ImageFeaturizer::saveFeaturesToFile(void* features, std::string filepath) {
	auto featureVector = static_cast<std::vector<double>*>(features);
	std::ofstream file;
	file.open(filepath);
	for (auto f : *featureVector) { file << f << "\n"; }
	file.close();
	delete featureVector;
}

void* ImageFeaturizer::loadFeatureFromFile(std::string filepath) {
	std::ifstream file;
	file.open(filepath);
	std::string line;
	auto* const feature = new std::vector<double>;
	double x;
	while (file >> x) { (*feature).push_back(x); }
	return feature;
}

double ImageFeaturizer::getDistance(void* f, void* g, DistanceMetric* metric) {
	const auto f_ = *static_cast<std::vector<double>*>(f);
	const auto g_ = *static_cast<std::vector<double>*>(g);
	return metric->calculateDistance(f_, g_);
}

void ImageFeaturizer::saveAfterFeaturizing(const cv::Mat& img, const std::string filepath) {
	saveFeaturesToFile(getFeature(img), filepath);
}


double ImageFeaturizer::getDistance(void* f, void* g, DistanceMetric* metric, int* breakAt, double* weights,
                                    int numBreaks) {
	const auto f_ = *static_cast<std::vector<double>*>(f);
	const auto g_ = *static_cast<std::vector<double>*>(g);
	double totalDistance = 0;
	int startIdx = 0;
	for (int i = 0; i <= numBreaks; ++i) {
		int endIdx = i < numBreaks ? breakAt[i] : f_.size();
		const std::vector<double> curr_f(f_.begin() + startIdx, f_.begin() + endIdx);
		const std::vector<double> curr_g(g_.begin() + startIdx, g_.begin() + endIdx);
		const auto dist = metric->calculateDistance(curr_f, curr_g);
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
	auto* const feature = new std::vector<double>;
	auto startR = img.rows / 2 - 4, startC = img.cols / 2 - 4;
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
	auto* const feature = new std::vector<double>;
	int size = 1;
	for (auto i = 0; i < 3; ++i) { if (mask[i]) { size *= max; } }
	for (int i = 0; i < size; ++i) { feature->push_back(0); }
	for (auto i = startEndIndices[0]; i < startEndIndices[1]; ++i) {
		for (auto j = startEndIndices[2]; j < startEndIndices[3]; ++j) {
			auto pixel = img.at<cv::Vec3b>(i, j);
			int multiplier = 1;
			int idx = 0;
			for (auto k = 0; k < 3; ++k) {
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
	auto* const feature = new std::vector<double>;
	auto size = bucketSize * bucketSize;
	for (auto i = 0; i < size; ++i) { feature->push_back(0); }
	for (auto i = startEndIndices[0]; i < startEndIndices[1]; ++i) {
		for (int j = startEndIndices[2]; j < startEndIndices[3]; ++j) {
			auto pixel = img.at<cv::Vec3b>(i, j);
			const double p0 = pixel[0], p1 = pixel[1], p2 = pixel[2];
			const auto sum = p0 + p1 + p2;
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
	auto* const feature = new std::vector<double>;
	const int size = bucketSize;
	for (auto i = 0; i < size; ++i) { feature->push_back(0); }
	for (auto i = startEndIndices[0]; i < startEndIndices[1]; ++i) {
		for (auto j = startEndIndices[2]; j < startEndIndices[3]; ++j) {
			auto pixel = img.at<cv::Vec3d>(i, j);
			const auto p0 = pixel[0], p1 = pixel[1], p2 = pixel[2];
			const auto avg = (p0 + p1 + p2) / 3;
			int idx = avg * (bucketSize - 1);
			feature->at(idx) += 1;
		}
	}
	return feature;
}


void* TopBottomMultiRGHistogramFeaturizer::getFeature(const cv::Mat& img) {
	auto* hist = new RGHistogramFeaturizer(100);
	int sei1[4] = {0, img.rows / 2, 0, img.cols};
	auto* topFeature = static_cast<std::vector<double>*>(hist->getFeature(img, sei1));
	int sei2[4] = {img.rows / 2, img.rows, 0, img.cols};
	auto* bottomFeature = static_cast<std::vector<double>*>(hist->getFeature(img, sei2));

	auto* feature = new std::vector<double>;
	for (auto f : *topFeature) { feature->push_back(f); }
	for (auto f : *bottomFeature) { feature->push_back(f); }
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
	auto hist = new RGHistogramFeaturizer(bucketSize);
	auto* fullFeature = static_cast<std::vector<double>*>(hist->getFeature(img));
	int sei[4] = {img.rows / 2 - 49, img.rows / 2 + 50, img.cols / 2 - 49, img.cols / 2 + 50};
	auto* centerFeature = static_cast<std::vector<double>*>(hist->getFeature(img, sei));

	auto* feature = new std::vector<double>;
	for (auto f : *fullFeature) { feature->push_back(f); }
	for (auto f : *centerFeature) { feature->push_back(f); }
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
	auto* hist = new RGHistogramFeaturizer(bucketSize);
	auto* fullFeature = static_cast<std::vector<double>*>(hist->getFeature(img));
	cv::Mat sx(img.rows, img.cols, CV_16SC3);
	cv::Mat sy(img.rows, img.cols, CV_16SC3);
	sobolX3x3(img, sx);
	sobolY3x3(img, sy);
	cv::Mat orientation(img.rows, img.cols, CV_64FC3);
	sobolOrientation(sx, sy, orientation);
	// get an averaged histogram
	auto* avg = new AvgHistogramFeaturizer(bucketSize);
	auto* texture = static_cast<std::vector<double>*>(avg->getFeature(orientation));

	auto* feature = new std::vector<double>;
	for (auto f : *fullFeature) { feature->push_back(f); }
	for (auto f : *texture) { feature->push_back(f); }

	delete fullFeature;
	delete texture;
	delete hist;
	delete avg;
	return feature;
}

double RGHistogramAndSobelOrientationTextureFeaturizer::getDistance(void* f, void* g, DistanceMetric* metric) {
	int breakAt[1] = {bucketSize * bucketSize};
	double weights[2] = {1.0, 1.0};
	return ImageFeaturizer::getDistance(f, g, metric, breakAt, weights, 1);
}

void* RGFullAndCenterSobelTopAndBottomFullFeaturizer::getFeature(const cv::Mat& img) {
	auto hist = new CenterFullMultiRGHistogramFeaturizer(bucketSize);
	auto* fullFeature = static_cast<std::vector<double>*>(hist->getFeature(img));
	cv::Mat sx(img.rows, img.cols, CV_16SC3);
	cv::Mat sy(img.rows, img.cols, CV_16SC3);
	sobolX3x3(img, sx);
	sobolY3x3(img, sy);
	cv::Mat orientation(img.rows, img.cols, CV_64FC3);
	sobolOrientation(sx, sy, orientation);
	// get an averaged histogram
	auto* avg = new AvgHistogramFeaturizer(bucketSize);
	int sei1[4] = {0, img.rows / 2, 0, img.cols};
	auto* textureTop = static_cast<std::vector<double>*>(avg->getFeature(orientation, sei1));
	int sei2[4] = {img.rows / 2, img.rows, 0, img.cols};
	auto* textureBottom = static_cast<std::vector<double>*>(avg->getFeature(orientation, sei2));

	auto* feature = new std::vector<double>;
	for (auto f : *fullFeature) { feature->push_back(f); }
	for (auto f : *textureTop) { feature->push_back(f); }
	for (auto f : *textureBottom) { feature->push_back(f); }

	delete fullFeature;
	delete textureTop;
	delete textureBottom;
	delete hist;
	delete avg;
	return feature;
}

double RGFullAndCenterSobelTopAndBottomFullFeaturizer::getDistance(void* f, void* g, DistanceMetric* metric) {
	int breakAt[3] = {bucketSize * bucketSize, 2 * bucketSize * bucketSize, 3 * bucketSize * bucketSize};
	double weights[4] = {2.0, 5.0, 1.0, 1.0};
	return ImageFeaturizer::getDistance(f, g, metric, breakAt, weights, 4);
}

void* CoOccurrenceMatrix::getFeature(const cv::Mat& img) {
	cv::Mat grey(img.rows, img.cols, CV_8UC3);
	greyscale(img, grey);
	auto* histogram = new std::vector<double>;
	int max = 256;
	for (int i = 0; i < max * max; ++i) { histogram->push_back(0.0); }
	for (int i = 0; i < img.rows - (1 - axis) * distance; ++i) {
		for (int j = 0; j < img.cols - axis * distance; ++j) {
			histogram->at(
				max * grey.at<cv::Vec3b>(i, j)[0] + grey.at<cv::Vec3b>(i + (1 - axis) * distance, j + axis * distance)[0
				]) += 1;
		}
	}

	auto sum = 0.0;
	for (auto e : *histogram) { sum += e; }
	for (auto& i : *histogram) { i /= sum; }

	const auto hist = static_cast<const std::vector<double>>(*histogram);
	auto* feature = new std::vector<double>;
	Energy energy{};
	const auto e = energy.calculate(hist);
	feature->push_back(e);
	updateMinsMaxs(e, 0);

	Entropy entropy{};
	const auto h = entropy.calculate(hist);
	feature->push_back(h);
	updateMinsMaxs(h, 1);

	Contrast contrast{};
	const auto c = contrast.calculate(hist);
	feature->push_back(c);
	updateMinsMaxs(c, 2);

	Homogeneity homogeniety{};
	const auto m = homogeniety.calculate(hist);
	feature->push_back(m);
	updateMinsMaxs(m, 3);

	MaximumProbability maxpr{};
	const auto p = maxpr.calculate(hist);
	feature->push_back(p);
	updateMinsMaxs(p, 4);

	delete histogram;
	return feature;
}

void CoOccurrenceMatrix::updateMinsMaxs(double x, int i) {
	mins->at(i) = MIN(x, mins->at(i));
	maxs->at(i) = MAX(x, maxs->at(i));
}

void CoOccurrenceMatrix::beforeFinishSaving(std::string featurizedDatabaseDir) {
	// rewrite all files to have the feature values updated by the formula (x-min)/(max-min).
	// This is a sort of a work around for making this update, but I think this is the best way with the current program design/structure.

	for (const auto& entry : std::filesystem::directory_iterator(featurizedDatabaseDir)) {
		std::ifstream oldFile(entry.path());
		// I know that appending to strings again and again is ineffective but due to the files not being extremely large, this inefficiency is okay.
		std::string newContent;
		double line = 0.0;
		int idx = 0;
		while (idx != fileStartIdx) {
			oldFile >> newContent;
			newContent += "n";
			idx++;
		}
		idx = 0;
		while (oldFile >> line) {
			newContent += std::to_string((line - mins->at(idx)) / (maxs->at(idx) - mins->at(idx)));
			newContent += "\n";
			idx++;
		}
		oldFile.close();

		std::ofstream newFile(entry.path());
		newFile << newContent;
		newFile.close();
	}
}

void* RGCoOccFullFeaturizer::getFeature(const cv::Mat& img) {
	RGHistogramFeaturizer rg(bucketSize);
	auto* colorHist = static_cast<std::vector<double>*>(rg.getFeature(img));
	CoOccurrenceMatrix com(axis, distance, bucketSize * bucketSize);
	auto* texture = static_cast<std::vector<double>*>(com.getFeature(img));
	auto* feature = new std::vector<double>;
	for (auto e : *colorHist) { feature->push_back(1.0 * e); }
	for (auto e : *texture) { feature->push_back(e); }
	delete colorHist;
	delete texture;
	return feature;
}

double RGCoOccFullFeaturizer::getDistance(void* f, void* g, DistanceMetric* metric) {
	int breakAt[5] = {
		bucketSize * bucketSize, bucketSize * bucketSize + 1, bucketSize * bucketSize + 2, bucketSize * bucketSize + 3,
		bucketSize * bucketSize + 4
	};
	double weights[6] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
	return ImageFeaturizer::getDistance(f, g, metric, breakAt, weights, 5);
}
