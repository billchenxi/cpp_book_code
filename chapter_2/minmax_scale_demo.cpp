// minmax_scale_demo.cpp
// g++ -std=c++17 -O2 minmax_scale_demo.cpp -o mm && ./mm

#include <vector>
#include <algorithm>
#include <iostream>
#include <iomanip>

// Scale features to a specific range [minRange, maxRange]
std::vector<double> minMaxScale(const std::vector<double>& data, double minRange, double maxRange) {
    double minVal = *std::min_element(data.begin(), data.end());
    double maxVal = *std::max_element(data.begin(), data.end());

    std::vector<double> scaledData;
    scaledData.reserve(data.size());
    for (const auto& val : data) {
        double scaled = minRange + ((val - minVal) / (maxVal - minVal)) * (maxRange - minRange);
        scaledData.push_back(scaled);
    }
    return scaledData;
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

    auto scaled01 = minMaxScale(x, 0.0, 1.0);
    auto scaledm1p1 = minMaxScale(x, -1.0, 1.0);

    printVec("Original     :", x);
    printVec("MinMax [0,1] :", scaled01);
    printVec("MinMax [-1,1]:", scaledm1p1);
    return 0;
}
