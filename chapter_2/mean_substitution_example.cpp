// mean_substitution_example.cpp
// % g++ -std=c++17 -O2 mean_substitution_example.cpp -o mean_sub 
// % ./mean_sub

#include <vector>
#include <numeric>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <limits>

// Replace missing values (NaN) with the mean of the column
std::vector<double> meanSubstitution(const std::vector<double>& data) {
    double sum = 0;
    int count = 0;
    for (const auto& val : data) {
        if (!std::isnan(val)) {
            sum += val;
            count++;
        }
    }
    double mean = sum / count; // assumes at least one non-NaN

    std::vector<double> filledData = data;
    for (auto& val : filledData) {
        if (std::isnan(val)) {
            val = mean;
        }
    }
    return filledData;
}

int main() {
    // Example data with NaNs
    std::vector<double> data = {
        1.0,
        std::numeric_limits<double>::quiet_NaN(),
        3.0,
        4.0,
        std::numeric_limits<double>::quiet_NaN(),
        6.0
    };

    // Print original
    std::cout << "Original: [";
    for (size_t i = 0; i < data.size(); ++i) {
        if (std::isnan(data[i])) std::cout << "NaN";
        else std::cout << std::fixed << std::setprecision(2) << data[i];
        if (i + 1 < data.size()) std::cout << ", ";
    }
    std::cout << "]\n";

    // Apply mean substitution
    auto filled = meanSubstitution(data);

    // Print filled
    std::cout << "Filled:   [";
    for (size_t i = 0; i < filled.size(); ++i) {
        std::cout << std::fixed << std::setprecision(2) << filled[i];
        if (i + 1 < filled.size()) std::cout << ", ";
    }
    std::cout << "]\n";

    return 0;
}
