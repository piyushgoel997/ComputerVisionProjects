#pragma once
#include <vector>

class DistanceMetric {
public:
	DistanceMetric(const bool normalize) : normalize(normalize) {}
	virtual double calculateDistance(const std::vector<double>& p, const std::vector<double>& q) = 0;
	static std::vector<double>* normalizeVector(const std::vector<double>& vec, bool normalize);

	const bool normalize;
};

class EuclideanDistance : public DistanceMetric {
	// sum of squared errors
public:
	EuclideanDistance(const bool normalize) : DistanceMetric(normalize) {}
	double calculateDistance(const std::vector<double>& p, const std::vector<double>& q) override;
};

class L1Norm : public DistanceMetric {
	// L-1 Norm
public:
	L1Norm(const bool normalize) : DistanceMetric(normalize) {}
	double calculateDistance(const std::vector<double>& p, const std::vector<double>& q) override;
};

class LNNorm : public DistanceMetric {
	// L-1 Norm
public:
	LNNorm(const bool normalize, const int N) : DistanceMetric(normalize), N(N) {}
	double calculateDistance(const std::vector<double>& p, const std::vector<double>& q) override;

private:
	const int N;
};

class HammingDistance : public DistanceMetric {
	// Hamming Distance (for histograms)
public:
	HammingDistance(const bool normalize) : DistanceMetric(false) {}
	double calculateDistance(const std::vector<double>& p, const std::vector<double>& q) override;
};

class NegativeOfHistogramIntersection : public DistanceMetric {
	// Histogram Distance (for histograms)
public:
	NegativeOfHistogramIntersection(const bool normalize) : DistanceMetric(normalize) {}
	double calculateDistance(const std::vector<double>& p, const std::vector<double>& q) override;
};

class Metric {
public:
	virtual double calculate(const std::vector<double>& p) = 0;
	static std::vector<double>* normalizeVector(const std::vector<int>& vec);
};

class Energy : public Metric {
public:
	double calculate(const std::vector<double>& p) override;
};

class Entropy : public Metric {
public:
	double calculate(const std::vector<double>& p) override;
};

class Contrast : public Metric {
public:
	double calculate(const std::vector<double>& p) override;
};

class Homogeneity : public Metric {
public:
	double calculate(const std::vector<double>& p) override;
};

class MaximumProbability : public Metric {
public:
	double calculate(const std::vector<double>& p) override;
};
