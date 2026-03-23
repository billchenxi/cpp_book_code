// log_transform_demo.cpp
// g++ -std=c++17 -O2 log_transform_demo.cpp -o logdemo
// ./logdemo

#include <vector>
#include <cmath>
#include <iostream>
#include <iomanip>

// Apply a log transformation to reduce skewness in data
std::vector<double> logTransform(const std::vector<double>& data) {
    std::vector<double> transformedData;
    transformedData.reserve(data.size());
    for (const auto& val : data) {
        transformedData.push_back(std::log1p(val)); // log(1 + x) to handle zero values
    }
    return transformedData;
}

void printVec(const char* label, const std::vector<double>& v) {
    std::cout << label << " [";
    for (size_t i=0;i<v.size();++i) {
        std::cout << std::fixed << std::setprecision(4) << v[i]
                  << (i+1<v.size()? ", ":"");
    }
    std::cout << "]\n";
}

int main() {
    // Example with positive and zero values
    std::vector<double> x = {0.0, 1.0, 3.0, 10.0, 100.0};

    auto y = logTransform(x);

    printVec("Original    :", x);
    printVec("log1p(x)    :", y);
    return 0;
}
