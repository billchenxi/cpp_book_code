// exponential_smoothing_demo.cpp
// g++ -std=c++17 -O2 exponential_smoothing_demo.cpp -o esmooth
// ./esmooth

#include <vector>
#include <iostream>
#include <iomanip>

// Apply exponential smoothing (given)
std::vector<double> exponentialSmoothing(const std::vector<double>& data, double alpha) {
    std::vector<double> smoothed(data.size(), 0.0);
    smoothed[0] = data[0];
    for (size_t i = 1; i < data.size(); ++i) {
        smoothed[i] = alpha * data[i] + (1 - alpha) * smoothed[i - 1];
    }
    return smoothed;
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
    // Example series
    std::vector<double> x = {10, 11, 13, 12, 14, 20, 18};

    auto s_low  = exponentialSmoothing(x, 0.2); // smoother, slower to react
    auto s_high = exponentialSmoothing(x, 0.7); // more responsive

    printVec("Data    :", x);
    printVec("alpha=0.2:", s_low);
    printVec("alpha=0.7:", s_high);
    return 0;
}
