// power_transform_demo.cpp
// g++ -std=c++17 -O2 power_transform_demo.cpp -o pwr && ./pwr

#include <vector>
#include <cmath>
#include <iostream>
#include <iomanip>

// Apply a power transformation to stabilize variance
std::vector<double> powerTransform(const std::vector<double>& data, double power) {
    std::vector<double> transformedData;
    transformedData.reserve(data.size());
    for (const auto& val : data) {
        transformedData.push_back(std::pow(val, power));
    }
    return transformedData;
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
    // Positive values to avoid domain issues with fractional powers
    std::vector<double> x = {0.0, 1.0, 4.0, 9.0, 16.0};

    auto sqrt_x = powerTransform(x, 0.5); // square-root
    auto sqr_x  = powerTransform(x, 2.0); // square

    printVec("Original :", x);
    printVec("Power 0.5:", sqrt_x);
    printVec("Power 2.0:", sqr_x);
    return 0;
}
