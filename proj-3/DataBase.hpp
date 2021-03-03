//
//  DataBase.hpp
//  
//
//  Created by Amit Mulay on 2/27/21.
//

#ifndef DataBase_hpp
#define DataBase_hpp

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <functional>
#include <algorithm>
#include <iterator>
#include <string>
#include <vector>
#include <numeric>
#include <opencv2/core.hpp>

class DataBase
{
private:
    std::fstream dbfile;
    std::vector<double> mean;
    std::vector<double> stdDev;
    int setMeanAndStdDevForEachFeature();
    double getEuclideanDistance(std::vector<double> &feature1, std::vector<double> &feature2);
public:
    DataBase();
    ~DataBase();
    int storeFeatureVectorInDB(std::vector<double> &features, std::string &label);
    int getMatchFromDB(std::vector<double> &features, std::string &label);
    int fileDB();
    int openDB();
};

#endif /* DataBase_hpp */


