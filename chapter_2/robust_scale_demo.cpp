// robust_scale_demo.cpp
// g++ -std=c++17 -O2 robust_scale_demo.cpp -o rscale
// ./rscale

#include <vector>
#include <algorithm>
#include <iostream>
#include <iomanip>

// Scale features using the median and interquartile range (IQR)
std::vector<double> robustScale(const std::vector<double>& data) {
    std::vector<double> sortedData = data;
    std::sort(sortedData.begin(), sortedData.end());

    double median = sortedData[sortedData.size() / 2];
    double q1 = sortedData[sortedData.size() / 4];
    double q3 = sortedData[3 * sortedData.size() / 4];
    double iqr = q3 - q1;

    std::vector<double> scaledData;
    for (const auto& val : data) {
        scaledData.push_back((val - median) / iqr);
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
    // Example with an outlier to highlight robustness
    std::vector<double> x = {2, 4, 6, 8, 10, 100};

    auto r = robustScale(x);

    printVec("Original :", x);
    printVec("Robust   :", r);
    return 0;
}
