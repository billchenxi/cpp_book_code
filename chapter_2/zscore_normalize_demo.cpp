// zscore_normalize_demo.cpp
// g++ -std=c++17 -O2 zscore_normalize_demo.cpp -o znorm && ./znorm

#include <vector>
#include <cmath>
#include <numeric>
#include <iostream>
#include <iomanip>

// Standardize features to have a mean of 0 and standard deviation of 1
std::vector<double> zScoreNormalize(const std::vector<double>& data) {
    double mean = std::accumulate(data.begin(), data.end(), 0.0) / data.size();
    double variance = 0.0;
    for (const auto& val : data) {
        variance += std::pow(val - mean, 2);
    }
    variance /= data.size();
    double stddev = std::sqrt(variance);

    std::vector<double> normalizedData;
    for (const auto& val : data) {
        normalizedData.push_back((val - mean) / stddev);
    }
    return normalizedData;
}

void printVec(const char* label, const std::vector<double>& v) {
    std::cout << label << " [";
    for (size_t i=0;i<v.size();++i) {
        std::cout << std::fixed << std::setprecision(3) << v[i]
                  << (i+1<v.size()? ", ":"");
    }
    std::cout << "]\n";
}

int main() {
    std::vector<double> x = {2, 4, 6, 8, 10};

    auto z = zScoreNormalize(x);

    printVec("Original :", x);
    printVec("Z-score  :", z);
    return 0;
}
