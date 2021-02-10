#include <iostream>
#include <string>

#include "ContentBasedFiltering.h"

int main(int argc, char* argv) {
	const std::string dir =
		"C:\\MyFolder\\Courses\\CS5330-ComputerVisionAndPatternRecognition\\CompVisionProjects\\proj-2\\dataset\\images\\";
	DistanceMetric* dm = new EuclideanDistance();
	ImageFeaturizer* featurizer = new BaselineFeaturizer(*dm);
	Matcher matcher(*featurizer, dir,
	                "C:\\MyFolder\\Courses\\CS5330-ComputerVisionAndPatternRecognition\\CompVisionProjects\\proj-2\\dataset\\baseline\\");
	// matcher.featurizeAndSaveDataset();
	std::vector<std::string> matches = *matcher.getMatches("pic.1016.jpg", 3);
	for (std::string m : matches) {
		std::cout << m << std::endl;
	}
}
