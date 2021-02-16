// #include <iostream>
// #include <string>
//
// #include "Matcher.h"
//
// int main(int argc, char* argv) {
// 	// const std::string dir =
// 	// 	"C:\\MyFolder\\Courses\\CS5330-ComputerVisionAndPatternRecognition\\CompVisionProjects\\proj-2\\dataset\\images\\";
// 	// ImageFeaturizer* featurizer = new BaselineFeaturizer();
// 	// Matcher* matcher = new Matcher(*featurizer, dir,
// 	//                 "C:\\MyFolder\\Courses\\CS5330-ComputerVisionAndPatternRecognition\\CompVisionProjects\\proj-2\\dataset\\baseline\\");
// 	// // matcher->featurizeAndSaveDataset();
// 	// DistanceMetric* dm = new EuclideanDistance(false);
// 	// DistanceMetric* dm = new L1Norm(false);
// 	// std::vector<std::string> matches = *(matcher->getMatches("pic.1016.jpg", 3, dm));
// 	// for (std::string m : matches) { std::cout << m << std::endl; }
// 	// delete dm;
// 	// delete matcher;
// 	// delete featurizer;
//
//
// 	// const std::string dir =
// 	// 	"C:\\MyFolder\\Courses\\CS5330-ComputerVisionAndPatternRecognition\\CompVisionProjects\\proj-2\\dataset\\images\\";
// 	// int mask[3] = {1, 1, 0};
// 	// ImageFeaturizer* featurizer = new HistogramFeaturizer(mask);
// 	// Matcher* matcher = new Matcher(*featurizer, dir,
// 	// 	"C:\\MyFolder\\Courses\\CS5330-ComputerVisionAndPatternRecognition\\CompVisionProjects\\proj-2\\dataset\\bg-histogram\\");
// 	// matcher->featurizeAndSaveDataset();
// 	// // DistanceMetric* dm = new EuclideanDistance(true);
// 	// // DistanceMetric* dm = new HammingDistance(true);
// 	// DistanceMetric* dm = new NegativeOfHistogramIntersection(true);
// 	// std::vector<std::string> matches = *(matcher->getMatches("pic.0164.jpg", 3, dm));
// 	// for (std::string m : matches) { std::cout << m << std::endl; }
// 	// delete dm;
// 	// delete matcher;
// 	// delete featurizer;
//
// 	// const std::string dir =
// 	// 	"C:\\MyFolder\\Courses\\CS5330-ComputerVisionAndPatternRecognition\\CompVisionProjects\\proj-2\\dataset\\images\\";
// 	// ImageFeaturizer* featurizer = new RGHistogramFeaturizer(100);
// 	// Matcher* matcher = new Matcher(*featurizer, dir,
// 	// 	"C:\\MyFolder\\Courses\\CS5330-ComputerVisionAndPatternRecognition\\CompVisionProjects\\proj-2\\dataset\\rghistogram\\");
// 	// // matcher->featurizeAndSaveDataset();
// 	// // DistanceMetric* dm = new EuclideanDistance(true);
// 	// // DistanceMetric* dm = new HammingDistance(true);
// 	// DistanceMetric* dm = new NegativeOfHistogramIntersection(true);
// 	// std::vector<std::string> matches = *(matcher->getMatches("pic.0164.jpg", 3, dm));
// 	// for (std::string m : matches) { std::cout << m << std::endl; }
// 	// delete dm;
// 	// delete matcher;
// 	// delete featurizer;
//
//
// 	// const std::string dir =
// 	// 	"C:\\MyFolder\\Courses\\CS5330-ComputerVisionAndPatternRecognition\\CompVisionProjects\\proj-2\\dataset\\images\\";
// 	// ImageFeaturizer* featurizer = new CenterFullMultiRGHistogramFeaturizer(100);
// 	// Matcher* matcher = new Matcher(*featurizer, dir,
// 	// 	"C:\\MyFolder\\Courses\\CS5330-ComputerVisionAndPatternRecognition\\CompVisionProjects\\proj-2\\dataset\\centerfull-multi-rghistogram\\");
// 	// // matcher->featurizeAndSaveDataset();
// 	// // DistanceMetric* dm = new EuclideanDistance(true);
// 	// // DistanceMetric* dm = new HammingDistance(true);
// 	// DistanceMetric* dm = new NegativeOfHistogramIntersection(true);
// 	// std::vector<std::string> matches = *(matcher->getMatches("pic.0135.jpg", 3, dm));
// 	// for (std::string m : matches) { std::cout << m << std::endl; }
// 	// delete dm;
// 	// delete matcher;
// 	// delete featurizer;
//
// 	// const std::string dir =
// 	// 	"C:\\MyFolder\\Courses\\CS5330-ComputerVisionAndPatternRecognition\\CompVisionProjects\\proj-2\\dataset\\images\\";
// 	// ImageFeaturizer* featurizer = new RGHistogramAndSobelOrientationTextureFeaturizer(100);
// 	// Matcher* matcher = new Matcher(*featurizer, dir,
// 	// 	"C:\\MyFolder\\Courses\\CS5330-ComputerVisionAndPatternRecognition\\CompVisionProjects\\proj-2\\dataset\\rg-sobel\\");
// 	// // matcher->featurizeAndSaveDataset();
// 	// // DistanceMetric* dm = new EuclideanDistance(true);
// 	// // DistanceMetric* dm = new HammingDistance(true);
// 	// DistanceMetric* dm = new NegativeOfHistogramIntersection(true);
// 	// std::vector<std::string> matches = *(matcher->getMatches("pic.0535.jpg", 3, dm));
// 	// for (std::string m : matches) { std::cout << m << std::endl; }
// 	// delete dm;
// 	// delete matcher;
// 	// delete featurizer;
//
//
// 	// const std::string dir =
// 	// 	"C:\\MyFolder\\Courses\\CS5330-ComputerVisionAndPatternRecognition\\CompVisionProjects\\proj-2\\dataset\\Task-5\\";
// 	// ImageFeaturizer* featurizer = new RGHistogramAndSobelOrientationTextureFeaturizer(32);
// 	// Matcher* matcher = new Matcher(*featurizer, dir,
// 	//                                "C:\\MyFolder\\Courses\\CS5330-ComputerVisionAndPatternRecognition\\CompVisionProjects\\proj-2\\dataset\\Task-5-features\\");
// 	// // matcher->featurizeAndSaveDataset();
// 	// // DistanceMetric* dm = new EuclideanDistance(true);
// 	// DistanceMetric* dm = new HammingDistance(true);
// 	// // DistanceMetric* dm = new LNNorm(true, 5);
// 	// // DistanceMetric* dm = new LNNorm(true, 5);
// 	// std::vector<std::string> matches = *(matcher->getMatches("pic.0754.jpg", 30, dm));
// 	// for (std::string m : matches) { std::cout << m << std::endl; }
// 	// delete dm;
// 	// delete matcher;
// 	// delete featurizer;
//
// 	const std::string dir =
// 		"C:\\MyFolder\\Courses\\CS5330-ComputerVisionAndPatternRecognition\\CompVisionProjects\\proj-2\\dataset\\images\\";
// 	ImageFeaturizer* featurizer = new RGCoOccFullFeaturizer(0, 5, 32);
// 	Matcher* matcher = new Matcher(*featurizer, dir,
// 		"C:\\MyFolder\\Courses\\CS5330-ComputerVisionAndPatternRecognition\\CompVisionProjects\\proj-2\\dataset\\co-rg\\");
// 	// matcher->featurizeAndSaveDataset();
// 	DistanceMetric* dm = new EuclideanDistance(true);
// 	std::vector<std::string> matches = *(matcher->getMatches("pic.0251.jpg", 10, dm));
// 	for (std::string m : matches) { std::cout << m << std::endl; }
// 	delete dm;
// 	delete matcher;
// 	delete featurizer;
// }
//
