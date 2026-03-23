// rolling_mean_demo.cpp
// g++ -std=c++17 -O2 rolling_mean_demo.cpp -o roll && ./roll

#include <vector>
#include <iostream>
#include <iomanip>

// Compute rolling averages over a fixed window size (provided)
std::vector<double> rollingMean(const std::vector<double>& data, int windowSize) {
    std::vector<double> result(data.size(), 0.0);
    for (size_t i = 0; i <= data.size() - windowSize; ++i) {
        double sum = 0;
        for (int j = 0; j < windowSize; ++j) sum += data[i + j];
        result[i + windowSize - 1] = sum / windowSize;
    }
    return result;
}

int main() {
    std::vector<double> x = {1,2,3,4,5,6,7,8,9,10};
    int W = 3; // window size

    auto m = rollingMean(x, W);

    // Print original
    std::cout << "Data:   [";
    for (size_t i=0;i<x.size();++i){
        std::cout << x[i] << (i+1<x.size()? ", ":"");
    }
    std::cout << "]\n";

    // Print rolling mean (only defined from index W-1 onward)
    std::cout << "Mean(" << W << "): [";
    for (size_t i=0;i<m.size();++i){
        if (i+1 < (size_t)W) std::cout << "--";
        else std::cout << std::fixed << std::setprecision(2) << m[i];
        if (i+1<m.size()) std::cout << ", ";
    }
    std::cout << "]\n";

    return 0;
}
