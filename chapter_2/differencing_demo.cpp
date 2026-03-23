// differencing_demo.cpp
// g++ -std=c++17 -O2 differencing_demo.cpp -o diff_demo
// ./diff_demo

#include <vector>
#include <iostream>
#include <iomanip>

// Perform first-order differencing to remove trends (given)
std::vector<double> differencing(const std::vector<double>& data) {
    std::vector<double> diff;
    if (data.size() < 2) return diff;
    diff.reserve(data.size() - 1);
    for (size_t i = 1; i < data.size(); ++i)
        diff.push_back(data[i] - data[i - 1]);
    return diff;
}

int main() {
    // Example series with a gentle upward trend
    std::vector<double> x = {100.0, 101.2, 102.7, 104.0, 105.1, 106.0, 106.8};

    auto d = differencing(x);

    std::cout << "Original: [";
    for (size_t i=0;i<x.size();++i)
        std::cout << std::fixed << std::setprecision(1) << x[i] << (i+1<x.size()? ", ":"");
    std::cout << "]\n";

    std::cout << "Diff(1):  [";
    for (size_t i=0;i<d.size();++i)
        std::cout << std::fixed << std::setprecision(1) << d[i] << (i+1<d.size()? ", ":"");
    std::cout << "]\n";

    return 0;
}
