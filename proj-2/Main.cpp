#include <iostream>
#include <string>

#include "ContentBasedFiltering.h"

int main(int argc, char* argv) {
	// const std::string dir =
	// 	"C:\\MyFolder\\Courses\\CS5330-ComputerVisionAndPatternRecognition\\CompVisionProjects\\proj-2\\dataset\\images\\";
	// ImageFeaturizer* featurizer = new BaselineFeaturizer();
	// Matcher* matcher = new Matcher(*featurizer, dir,
	//                 "C:\\MyFolder\\Courses\\CS5330-ComputerVisionAndPatternRecognition\\CompVisionProjects\\proj-2\\dataset\\baseline\\");
	// // matcher->featurizeAndSaveDataset();
	// DistanceMetric* dm = new EuclideanDistance(false);
	// DistanceMetric* dm = new L1Norm(false);
	// std::vector<std::string> matches = *(matcher->getMatches("pic.1016.jpg", 3, dm));
	// for (std::string m : matches) { std::cout << m << std::endl; }
	// delete dm;
	// delete matcher;
	// delete featurizer;


	// const std::string dir =
	// 	"C:\\MyFolder\\Courses\\CS5330-ComputerVisionAndPatternRecognition\\CompVisionProjects\\proj-2\\dataset\\images\\";
	// int mask[3] = {1, 1, 0};
	// ImageFeaturizer* featurizer = new HistogramFeaturizer(mask);
	// Matcher* matcher = new Matcher(*featurizer, dir,
	// 	"C:\\MyFolder\\Courses\\CS5330-ComputerVisionAndPatternRecognition\\CompVisionProjects\\proj-2\\dataset\\bg-histogram\\");
	// matcher->featurizeAndSaveDataset();
	// // DistanceMetric* dm = new EuclideanDistance(true);
	// // DistanceMetric* dm = new HammingDistance(true);
	// DistanceMetric* dm = new HistogramDistance(true);
	// std::vector<std::string> matches = *(matcher->getMatches("pic.0164.jpg", 3, dm));
	// for (std::string m : matches) { std::cout << m << std::endl; }
	// delete dm;
	// delete matcher;
	// delete featurizer;


	const std::string dir =
		"C:\\MyFolder\\Courses\\CS5330-ComputerVisionAndPatternRecognition\\CompVisionProjects\\proj-2\\dataset\\images\\";
	int mask[3] = { 1, 1, 0 };
	ImageFeaturizer* featurizer = new TopBottomMultiHistogramFeaturizer(mask);
	Matcher* matcher = new Matcher(*featurizer, dir,
		"C:\\MyFolder\\Courses\\CS5330-ComputerVisionAndPatternRecognition\\CompVisionProjects\\proj-2\\dataset\\bg-topbottom-multihistogram\\");
	matcher->featurizeAndSaveDataset();
	// DistanceMetric* dm = new EuclideanDistance(true);
	// DistanceMetric* dm = new HammingDistance(true);
	DistanceMetric* dm = new HistogramDistance(true);
	std::vector<std::string> matches = *(matcher->getMatches("pic.0164.jpg", 3, dm));
	for (std::string m : matches) { std::cout << m << std::endl; }
	delete dm;
	delete matcher;
	delete featurizer;
}
