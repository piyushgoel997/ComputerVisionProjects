#pragma once
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

// ref: https://docs.opencv.org/3.4/de/d01/samples_2cpp_2connected_components_8cpp-example.html
// task 2
// TODO implement own connected components algorithm
void segmentAndColorRegions(cv::Mat& src, cv::Mat& dst, int connectivity = 4) {
	srand(1);
	cv::Mat labels;
	cv::Mat stats;
	cv::Mat centroids;
	int numLabels = cv::connectedComponentsWithStats(src, labels, stats, centroids);
	std::vector<cv::Vec3b> colors(numLabels);
	colors[0] = cv::Vec3b(0, 0, 0); // black for the bkg
	for (int i = 1; i < numLabels; ++i) {
		// TODO instead select top N
		if (stats.at<int>(i, cv::CC_STAT_AREA) < 2000) { colors[i] = cv::Vec3b(0, 0, 0); }
		else { colors[i] = cv::Vec3b((rand() & 255), (rand() & 255), (rand() & 255)); }
	}

	for (auto i = 0; i < src.rows; ++i) {
		for (auto j = 0; j < src.cols; ++j) { dst.at<cv::Vec3b>(i, j) = colors[labels.at<int>(i, j)]; }
	}
}

/**
 * @brief Creates a list of (x,y) pairs for each region.
 * @param labels A cv matrix (of the type int) for the image with each
 * @return A list of list of pairs of integers
*/
std::vector<std::vector<std::pair<int, int>>>* getListOfCoordsForEachRegion(cv::Mat& labels) {
	auto* listOfCoords = new std::vector<std::vector<std::pair<int, int>>>;
	for (int i = 0; i < labels.rows; ++i) {
		for (int j = 0; j < labels.cols; ++j) {
			if (labels.at<int>(i, j) >= listOfCoords->size()) {
				std::vector<std::pair<int, int>> vec;
				// flip i and j because y corresponds to the row and x corresponds to the col
				vec.emplace_back(j, i);
				listOfCoords->push_back(vec);
			}
			else {
				// used .emplace_back instead of .push_back(std::pair<int, int>(i, j))
				listOfCoords->at(labels.at<int>(i, j)).emplace_back(j, i);
			}
		}
	}
	return listOfCoords;
}

