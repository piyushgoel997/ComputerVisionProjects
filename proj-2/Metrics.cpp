#include "Metrics.h"

#include <opencv2/core/base.hpp>

// Distance metrics. Names are self explanatory.

std::vector<double>* DistanceMetric::normalizeVector(const std::vector<double>& vec, bool normalize) {
	auto normalized = new std::vector<double>;
	double sum = 0;
	for (auto i : vec) { sum += i; }
	for (double i : vec) {
		if (normalize) { normalized->push_back(i / sum); }
		else { normalized->push_back(i); }
	}
	return normalized;
}


double EuclideanDistance::calculateDistance(const std::vector<double>& p, const std::vector<double>& q) {
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

double L1Norm::calculateDistance(const std::vector<double>& p, const std::vector<double>& q) {
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

double LNNorm::calculateDistance(const std::vector<double>& p, const std::vector<double>& q) {
	auto distance = 0.0;
	auto p_ = normalizeVector(p, normalize);
	auto q_ = normalizeVector(q, normalize);
	auto ct = 0.0;
	for (int i = 0; i < p_->size(); ++i) {
		auto a = p_->at(i), b = q_->at(i);
		distance += pow(abs(a - b), N);
	}
	delete p_;
	delete q_;
	return distance;
}


double HammingDistance::calculateDistance(const std::vector<double>& p, const std::vector<double>& q) {
	auto distance = 0.0;
	for (int i = 0; i < p.size(); ++i) {
		auto a = p.at(i) > 0 ? 1 : 0, b = q.at(i) > 0 ? 1 : 0;
		distance += abs(a - b);
	}
	return distance;
}

double NegativeOfHistogramIntersection::calculateDistance(const std::vector<double>& p, const std::vector<double>& q) {
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


// Metrics for co-occurrence matrix. Names are self-explanatory.

double Energy::calculate(const std::vector<double>& p) {
	auto e = 0.0;
	for (auto i : p) { e += i * i; }
	return e;
}

double Entropy::calculate(const std::vector<double>& p) {
	auto e = 0.0;
	for (auto i : p) { if (i != 0) { e += i * log(i); } }
	return -e;
}

double Contrast::calculate(const std::vector<double>& p) {
	auto e = 0.0;
	int max = 256;
	for (auto i = 0; i < p.size(); ++i) { e += pow((i / max - i % max), 2) * p.at(i); }
	return e;
}

double Homogeneity::calculate(const std::vector<double>& p) {
	auto e = 0.0;
	int max = 256;
	for (auto i = 0; i < p.size(); ++i) { e += (1.0 * p.at(i)) / (1 + abs(i / max - i % max)); }
	return e;
}

double MaximumProbability::calculate(const std::vector<double>& p) {
	auto e = 0.0;
	for (auto i : p) { e = MAX(i, e); }
	return e;
}
