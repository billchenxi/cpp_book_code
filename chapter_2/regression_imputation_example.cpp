// regression_imputation_example.cpp
// g++ -std=c++17 -O2 regression_imputation_example.cpp -o reg_imp
// ./reg_imp

#include <vector>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <limits>

// Replace missing values using linear regression coefficients
std::vector<double> regressionImputation(const std::vector<double>& target,
                                         const std::vector<double>& predictor,
                                         double slope, double intercept) {
    std::vector<double> filledData = target;
    for (size_t i = 0; i < filledData.size(); ++i) {
        if (std::isnan(filledData[i])) {
            filledData[i] = slope * predictor[i] + intercept;
        }
    }
    return filledData;
}

int main() {
    using std::numeric_limits;
    double NaN = numeric_limits<double>::quiet_NaN();

    // Toy relation: y = 2*x + 1, with missing targets
    std::vector<double> x = {1, 2, 3, 4, 5, 6};
    std::vector<double> y = {3, NaN, 7, NaN, 11, NaN};

    // Assume these were fit elsewhere
    double slope = 2.0;
    double intercept = 1.0;

    auto filled = regressionImputation(y, x, slope, intercept);

    auto print = [](const char* label, const std::vector<double>& v){
        std::cout << label << " [";
        for (size_t i=0;i<v.size();++i){
            if (std::isnan(v[i])) std::cout << "NaN";
            else std::cout << std::fixed << std::setprecision(2) << v[i];
            if (i+1<v.size()) std::cout << ", ";
        }
        std::cout << "]\n";
    };

    print("Original:", y);
    print("Filled:  ", filled);
    return 0;
}
