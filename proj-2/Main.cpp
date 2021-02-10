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
	// DistanceMetric* dm = new EuclideanDistance();
	// std::vector<std::string> matches = *(matcher->getMatches("pic.1016.jpg", 3, dm));
	// for (std::string m : matches) { std::cout << m << std::endl; }


	const std::string dir =
		"C:\\MyFolder\\Courses\\CS5330-ComputerVisionAndPatternRecognition\\CompVisionProjects\\proj-2\\dataset\\images\\";
	ImageFeaturizer* featurizer = new HistogramFeaturizer();
	Matcher* matcher = new Matcher(*featurizer, dir,
		"C:\\MyFolder\\Courses\\CS5330-ComputerVisionAndPatternRecognition\\CompVisionProjects\\proj-2\\dataset\\histogram\\");
	// matcher->featurizeAndSaveDataset();
	DistanceMetric* dm = new EuclideanDistance();
	std::vector<std::string> matches = *(matcher->getMatches("pic.1016.jpg", 3, dm));
	for (std::string m : matches) { std::cout << m << std::endl; }

}
