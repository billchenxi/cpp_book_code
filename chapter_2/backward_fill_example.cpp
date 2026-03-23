// backward_fill_example.cpp
// % g++ -std=c++17 -O2 backward_fill_example.cpp -o bfill
// % ./bfill

#include <vector>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <limits>

// Replace missing values with the next non-missing value
std::vector<double> backwardFill(const std::vector<double>& data) {
    std::vector<double> filledData = data;
    if (filledData.empty()) return filledData;
    for (size_t i = filledData.size() - 1; i > 0; --i) {
        if (std::isnan(filledData[i - 1])) {
            filledData[i - 1] = filledData[i];
        }
    }
    return filledData;
}

int main() {
    using std::numeric_limits;
    double NaN = numeric_limits<double>::quiet_NaN();

    // Example data (note: last value is not NaN so backward-fill can propagate)
    std::vector<double> data = {NaN, NaN, 10.0, NaN, 12.5, NaN, 15.0};

    // Print original
    std::cout << "Original:       [";
    for (size_t i = 0; i < data.size(); ++i) {
        if (std::isnan(data[i])) std::cout << "NaN";
        else std::cout << std::fixed << std::setprecision(2) << data[i];
        if (i + 1 < data.size()) std::cout << ", ";
    }
    std::cout << "]\n";

    // Apply backward fill
    auto filled = backwardFill(data);

    // Print filled
    std::cout << "Backward-filled: [";
    for (size_t i = 0; i < filled.size(); ++i) {
        if (std::isnan(filled[i])) std::cout << "NaN";
        else std::cout << std::fixed << std::setprecision(2) << filled[i];
        if (i + 1 < filled.size()) std::cout << ", ";
    }
    std::cout << "]\n";

    return 0;
}
