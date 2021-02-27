#pragma once

#include <vector>
#include <algorithm>
#define M_PI 3.14159265358979323846

// TODO what should f(x,y) be?


// define a generic moment function
static double moment(std::vector<std::pair<int, int>>& vector, int p, int q) {
	auto m = 0.0;
	for (auto coord : vector) {
		auto x = coord.first;
		auto y = coord.second;
		m += pow(x, p) * pow(y, q);
	}
	return m;
}

// centroid function
static std::pair<double, double> findCentroid(std::vector<std::pair<int, int>>& vector, double m00 = -1) {
	if (m00 == -1) { m00 = moment(vector, 0, 0); }
	return std::pair<double, double>(moment(vector, 1, 0) / m00, moment(vector, 0, 1) / m00);
}

// a general central moment
static double centralMoment(std::vector<std::pair<int, int>>& vector, int p, int q, double m00 = -1) {
	auto m = 0.0;
	auto c = findCentroid(vector);
	auto c_x = c.first;
	auto c_y = c.second;
	for (auto coord : vector) {
		auto x = coord.first;
		auto y = coord.second;
		m += pow(x - c_x, p) * pow(y - c_y, q);
	}
	if (m00 == -1) { m00 = moment(vector, 0, 0); }
	return m / m00;
}

// orientation of the central axis
static double findAlpha(std::vector<std::pair<int, int>>& vector) {
	return 0.5 * atan2(2 * centralMoment(vector, 1, 1),
	                   centralMoment(vector, 2, 0) - centralMoment(vector, 0, 2));
}

// Feature 1: second moment about the central axis
static double mu22Alpha(std::vector<std::pair<int, int>>& vector, double alpha, double m00 = -1) {
	auto m = 0.0;
	auto c = findCentroid(vector);
	auto c_x = c.first;
	auto c_y = c.second;
	auto beta = alpha + M_PI / 2;
	for (const auto [x, y] : vector) { m += pow((x - c_x) * sin(beta) + (y - c_y) * cos(beta), 2); }
	if (m00 == -1) { m00 = moment(vector, 0, 0); }
	return m / m00;
}

// project points to the new co-ordinate system with the centroid as the origin
static void projectPoints(std::vector<std::pair<int, int>>& points, std::vector<std::pair<double, double>>& projected,
                   double alpha, std::pair<int, int> centroid) {
	auto c_x = centroid.first;
	auto c_y = centroid.second;
	for (const auto [x, y] : points) {
		projected.push_back({
			cos(alpha) * (1.0 * x - c_x) + sin(alpha) * (1.0 * y - c_y),
			- sin(alpha) * (1.0 * x - c_x) + cos(alpha) * (1.0 * y - c_y)
		});
	}
}

// get the dims of bounding box [min_x, miny_y, max_x, max_y]
template <typename T>
static T* boundingBoxDims(std::vector<std::pair<T, T>>& points) {
	T bb[4] = {0, 0, 0, 0};
	for (const auto [x, y] : points) {
		bb[0] = std::min(x, bb[0]);
		bb[1] = std::min(y, bb[1]);
		bb[2] = std::max(x, bb[2]);
		bb[3] = std::max(y, bb[3]);
	}
	return bb;
}

// one function which creates the bounding box

// then the function which combines it all together and gets the feature
static void getFeatures(std::vector<std::pair<int, int>>& points, std::vector<double>& features) {
	auto alpha = findAlpha(points);
	auto m00 = moment(points, 0, 0);
	auto centroid = findCentroid(points, m00);
	// mu 2 2 alpha
	features.push_back(mu22Alpha(points, alpha, m00));

	std::vector<std::pair<double, double>> projected;
	projectPoints(points, projected, alpha, centroid);

	// ht/wd of the bounding box
	auto* bb = boundingBoxDims<double>(projected);
	auto bb_h = bb[2] - bb[0];
	auto bb_w = bb[3] - bb[1];
	features.push_back((bb_h) / (bb_w));

	// % of the box filled
	features.push_back(m00 / (bb_h * bb_w));

	// TODO histogram of projection (16 buckets wide)
}
