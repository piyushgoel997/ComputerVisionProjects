#pragma once
#include <filesystem>

#include "ImageFeaturizer.h"


class Matcher {
public:
	Matcher(ImageFeaturizer& featurizer, std::string& databaseDir, std::string& featurizedDatabaseDir) :
		featurizer(featurizer), databaseDir(databaseDir), featurizedDatabaseDir(featurizedDatabaseDir) {};
	std::vector<std::string>* getMatches(std::string imgname, int numMatches, DistanceMetric* metric);
	static bool validImageExtn(std::string extension);
	static std::string getFilenameFromPath(std::filesystem::path path);
	void featurizeAndSaveDataset();

private:
	ImageFeaturizer& featurizer;
	const std::string databaseDir;
	const std::string featurizedDatabaseDir;
};
