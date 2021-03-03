//
//  DataBase.cpp
//  
//
//  Created by Amit Mulay on 2/27/21.
//

#include "DataBase.hpp"

DataBase :: DataBase()
{
    dbfile.open("FeatureVector_DB.txt", std::ios::out);
    if(!dbfile.is_open())
        std::cout << "DB file not open!" << std::endl;
}

DataBase :: ~DataBase(){
    std::cout << "File closed!" << std::endl;
    if(dbfile.is_open())
        dbfile.close();
}

int DataBase::storeFeatureVectorInDB(std::vector<double> &features, std::string &label)
{
    if(!dbfile.is_open())
        std::cout << "DB file not open!" << std::endl;
    else
    {
        dbfile << label;
        for (auto f : features) {
            dbfile << " " << f;
        }
        dbfile << std::endl;
    }
    return 0;
}

int DataBase::getMatchFromDB(std::vector<double> &features, std::string &label)
{
    dbfile.clear();
    dbfile.seekg(0, std::ios::beg);
    std::vector<double> tempFeatures;
    double currentDistance = std::numeric_limits<double>::infinity();
    double newDistance;
    double f1;
    std::string tempLabel;
    
    std::string line;
    while (std::getline(dbfile, line))
    {
        tempFeatures.clear();
        std::stringstream lineStream(line);
        lineStream >> tempLabel;
        while(lineStream >> f1)
        {
            tempFeatures.push_back(f1);
        }
        newDistance = getEuclideanDistance(features, tempFeatures);
        if(currentDistance > newDistance)
        {
            label.assign(tempLabel);
            currentDistance = newDistance;
        }
    }
    
    return 0;
}

double DataBase::getEuclideanDistance(std::vector<double> &feature1, std::vector<double> &feature2)
{
    for (int i = 0; i < 3; i++)
    {
        feature1[i] = (feature1[i] - mean[i])/stdDev[i];
        feature2[i] = (feature2[i] - mean[i])/stdDev[i];
    }
    return cv::norm(feature1,feature1);
}

int DataBase::setMeanAndStdDevForEachFeature()
{
    dbfile.clear();
    dbfile.seekg(0, std::ios::beg);
    std::vector<double> feature1;
    std::vector<double> feature2;
    std::vector<double> feature3;
    std::string tempLabel;
    double f1,f2,f3;
    double sum;
    double sq_sum;
    double stddev;
    
    while(dbfile >> tempLabel >> f1 >> f2 >> f3)
    {
        feature1.push_back(f1);
        feature2.push_back(f2);
        feature3.push_back(f3);
    }
    
    //Calculate mean
    sum = std::accumulate(feature1.begin(), feature1.end(), 0.0);
    mean.push_back(sum/feature1.size());
    sum = std::accumulate(feature2.begin(), feature2.end(), 0.0);
    mean.push_back(sum/feature2.size());
    sum = std::accumulate(feature3.begin(), feature3.end(), 0.0);
    mean.push_back(sum/feature3.size());
    
    //Calclate standard deviation
    std::vector<double> diff1(feature1.size());
    std::transform(feature1.begin(), feature1.end(), diff1.begin(),
                   std::bind(std::minus<double>(), std::placeholders::_1, mean[0]));
    sq_sum = std::inner_product(diff1.begin(), diff1.end(), diff1.begin(), 0.0);
    stddev = std::sqrt(sq_sum / feature1.size());
    stdDev.push_back(stddev);
    
    std::vector<double> diff2(feature2.size());
    std::transform(feature2.begin(), feature2.end(), diff2.begin(),
                   std::bind(std::minus<double>(), std::placeholders::_1, mean[2]));
    sq_sum = std::inner_product(diff2.begin(), diff2.end(), diff2.begin(), 0.0);
    stddev = std::sqrt(sq_sum / feature2.size());
    stdDev.push_back(stddev);
    
    std::vector<double> diff3(feature3.size());
    std::transform(feature3.begin(), feature3.end(), diff3.begin(),
                   std::bind(std::minus<double>(), std::placeholders::_1, mean[2]));
    sq_sum = std::inner_product(diff3.begin(), diff3.end(), diff3.begin(), 0.0);
    stddev = std::sqrt(sq_sum / feature3.size());
    stdDev.push_back(stddev);
    return 0;
}

int DataBase::fileDB()
{
    dbfile.close();
    return 0;
}

int DataBase::openDB()
{
    dbfile.open("FeatureVector_DB.txt", std::ios::in);
    if(!dbfile.is_open())
        std::cout << "DB file not open!" << std::endl;

    setMeanAndStdDevForEachFeature();
    return 0;
}
