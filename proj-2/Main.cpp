#include <iostream>
#include <opencv2/core.hpp>

#include "Matcher.h"


ImageFeaturizer* getFeaturizer(const std::string& F);
DistanceMetric* getDistanceMetric(const std::string& d);


int main(int argc, char* argv[]) {
	if (argc < 6) {
		std::cout << "Incorrect number of arguments. Found " << argc << ", required either 6 or 7.\n";
		return 1;
	}
	// either provide the feature database corresponding to the correct featurizer or don't provide it at all, as an incorrect database will result in undefined behavior.
	std::string T = argv[1], B = argv[2], F = argv[3], D = argv[4], F_db = "temp\\";

	// create a temporary working directory
	if (argc > 6) { F_db = argv[6]; }
	else {
		F_db = std::filesystem::current_path().append(F_db).string();
		if (!std::filesystem::create_directory(F_db)) {
			std::cout <<
				"ERROR: couldn't create temporary directory. Make sure the folder where you're running this code doesn't already contain a temp folder.\n";
			return 1;
		}
	}

	// read N and check if valid.
	int N;
	std::istringstream iss(argv[5]);
	if (!(iss >> N) || N <= 0) {
		std::cout << "ERROR:Invalid number of matches to be found.\n";
		return 1;
	}

	// create featurizer object if valid input.
	ImageFeaturizer* featurizer = getFeaturizer(F);
	if (featurizer == nullptr) {
		std::cout << "ERROR:Incorrect method name.\n";
		return 1;
	}

	// create metric object if valid input.
	DistanceMetric* dm = getDistanceMetric(D);
	if (dm == nullptr) {
		std::cout << "ERROR:Incorrect distance metric name.\n";
		return 1;
	}

	Matcher* matcher = new Matcher(*featurizer, B, F_db);

	// only create a database if the feature database is not yet filled up.
	if (std::filesystem::is_empty(F_db)) { matcher->featurizeAndSaveDataset(); }
	
	std::string target = T;
	while (1) {
		if (T == "i") {
			// multi-image mode
			std::cout <<
				"Enter the image file name (with the correct extension and the image should be in the image database).\n";
			std::cin >> target;
		}

		if (target == "q") { break; }

		std::vector<std::string>* matches = matcher->getMatches(target, N, dm);
		std::cout << "Top " << N << " matches (in order of best to worst): ";
		for (std::string m : *matches) { std::cout << m << ", "; }
		std::cout << std::endl;
		delete matches;
		if (T != "i") { break;; }
	}

	// delete the temporary directory
	if (argc <= 6) { std::filesystem::remove_all(F_db); }

	delete matcher;
	delete featurizer;
	delete dm;
}


ImageFeaturizer* getFeaturizer(const std::string& F) {
	// This function creates the appropriate featurizer given the command line argument. Returns nullptr if the argument doesn't correspond to any of the featurizers.

	if (F == "baseline" || F == "b") { return new BaselineFeaturizer(); }
	if (F == "histogram-32" || F == "h-32") { return new RGHistogramFeaturizer(32); }
	if (F == "histogram" || F == "h") {
		int b = 0;
		while (b <= 0) {
			std::cout << "Enter a valid number of buckets (> 0):\n";
			std::cin >> b;
		}
		return new RGHistogramFeaturizer(b);
	}
	if (F == "multihistogram-topbottom-32" || F == "mh-tb-32") { return new TopBottomMultiRGHistogramFeaturizer(32); }
	if (F == "multihistogram-topbottom" || F == "mh-tb") {
		int b = 0;
		while (b <= 0) {
			std::cout << "Enter a valid number of buckets (> 0):\n";
			std::cin >> b;
		}
		return new TopBottomMultiRGHistogramFeaturizer(b);
	}
	if (F == "multihistogram-centerfull-32" || F == "mh-cf-32") { return new CenterFullMultiRGHistogramFeaturizer(32); }
	if (F == "multihistogram-centerfull" || F == "mh-cf") {
		int b = 0;
		while (b <= 0) {
			std::cout << "Enter a valid number of buckets (> 0):\n";
			std::cin >> b;
		}
		return new CenterFullMultiRGHistogramFeaturizer(b);
	}
	if (F == "histogram-sobel-32" || F == "h-s-32") { return new RGHistogramAndSobelOrientationTextureFeaturizer(32); }
	if (F == "histogram-sobel" || F == "h-s") {
		int b = 0;
		while (b <= 0) {
			std::cout << "Enter a valid number of buckets (> 0):\n";
			std::cin >> b;
		}
		return new RGHistogramAndSobelOrientationTextureFeaturizer(b);
	}
	if (F == "multihistogram-centerfull-multisobel-topbottom-32" || F == "mh-cf-ms-tb-32") {
		return new RGFullAndCenterSobelTopAndBottomFullFeaturizer(32);
	}
	if (F == "multihistogram-centerfull-multisobel-topbottom" || F == "mh-cf-ms-tb") {
		int b = 0;
		while (b <= 0) {
			std::cout << "Enter a valid number of buckets (> 0):\n";
			std::cin >> b;
		}
		return new RGFullAndCenterSobelTopAndBottomFullFeaturizer(b);
	}
	if (F == "cooc-0" || F == "c0") { return new CoOccurrenceMatrix(0, 5); }
	if (F == "cooc-1" || F == "c1") { return new CoOccurrenceMatrix(1, 5); }
	if (F == "cooc" || F == "c") {
		int a = -1;
		while (a != 0 && a != 1) {
			std::cout << "Enter a valid axis number (either 0 or 1):\n";
			std::cin >> a;
		}
		int d = 0;
		while (d <= 0 || d > 10) {
			std::cout << "Enter a valid distance (> 0, <= 10):\n";
			std::cin >> d;
		}
		return new CoOccurrenceMatrix(a, d);
	}
	if (F == "histogram-cooc-0" || F == "h-c0") { return new RGCoOccFullFeaturizer(0, 5, 32); }
	if (F == "histogram-cooc-1" || F == "h-c1") { return new RGCoOccFullFeaturizer(1, 5, 32); }
	if (F == "histogram-cooc" || F == "h-c") {
		int a = -1;
		while (a != 0 && a != 1) {
			std::cout << "Enter a valid axis number (either 0 or 1):\n";
			std::cin >> a;
		}
		int d = 0;
		while (d <= 0 || d > 10) {
			std::cout << "Enter a valid distance (> 0, <= 10):\n";
			std::cin >> d;
		}
		int b = 0;
		while (b <= 0) {
			std::cout << "Enter a valid number of buckets (> 0):\n";
			std::cin >> b;
		}
		return new RGCoOccFullFeaturizer(a, d, b);
	}
	if (F == "pure-histogram") {
		int mask[3] = {0, 0, 0};
		int sum = 0;
		while (sum == 0) {
			std::cout <<
				"Please enter a valid color string, containing at least one of the three characters - r, g, b.\n";
			std::string colors;
			std::cin >> colors;
			if (colors.find('b') != std::string::npos) { mask[0] = 1; }
			if (colors.find('g') != std::string::npos) { mask[1] = 1; }
			if (colors.find('r') != std::string::npos) { mask[2] = 1; }
			for (auto m : mask) { sum += m; }
		}
		return new HistogramFeaturizer(mask);
	}
	return nullptr;
}

DistanceMetric* getDistanceMetric(const std::string& D) {
	// This function creates the appropriate distance metric given the command line argument. Returns nullptr if the argument doesn't correspond to any of the distance metrics.
	bool norm = false;
	std::string endsWith = "-n";
	std::string d = D;
	if (D.length() > endsWith.length() && D.compare(D.length() - endsWith.length(), endsWith.length(), endsWith) == 0) {
		norm = true;
		d = D.substr(0, D.length() - endsWith.length());
	}

	if (d == "euclid" || d == "e" || d == "l2") { return new EuclideanDistance(norm); }
	if (d == "l1") { return new L1Norm(norm); }
	if (d == "ln") {
		int n = 0;
		while (n <= 0) {
			std::cout << "Enter a valid value of n (>0):\n";
			std::cin >> n;
		}
		return new LNNorm(norm, n);
	}
	if (d == "hamming-distance" || d == "h") { return new HammingDistance(norm); }
	if (d == "histogram-intersection" || d == "i") { return new NegativeOfHistogramIntersection(norm); }
	return nullptr;
}
