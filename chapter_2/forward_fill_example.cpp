// forward_fill_example.cpp
// % g++ -std=c++17 -O2 forward_fill_example.cpp -o ffill 
// % ./ffill

#include <vector>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <limits>

// Replace missing values with the last non-missing value
std::vector<double> forwardFill(const std::vector<double>& data) {
    std::vector<double> filledData = data;
    for (size_t i = 1; i < filledData.size(); ++i) {
        if (std::isnan(filledData[i])) {
            filledData[i] = filledData[i - 1];
        }
    }
    return filledData;
}

int main() {
    using std::numeric_limits;
    double NaN = numeric_limits<double>::quiet_NaN();

    // Example data (note: first value is not NaN so forward-fill can propagate)
    std::vector<double> data = {10.0, NaN, NaN, 12.5, NaN, 15.0, NaN};

    // Print original
    std::cout << "Original:      [";
    for (size_t i = 0; i < data.size(); ++i) {
        if (std::isnan(data[i])) std::cout << "NaN";
        else std::cout << std::fixed << std::setprecision(2) << data[i];
        if (i + 1 < data.size()) std::cout << ", ";
    }
    std::cout << "]\n";

    // Apply forward fill
    auto filled = forwardFill(data);

    // Print filled
    std::cout << "Forward-filled: [";
    for (size_t i = 0; i < filled.size(); ++i) {
        if (std::isnan(filled[i])) std::cout << "NaN";
        else std::cout << std::fixed << std::setprecision(2) << filled[i];
        if (i + 1 < filled.size()) std::cout << ", ";
    }
    std::cout << "]\n";

    return 0;
}
