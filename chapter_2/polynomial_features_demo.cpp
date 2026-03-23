// polynomial_features_demo.cpp
// g++ -std=c++17 -O2 polynomial_features_demo.cpp -o poly_feats
// ./poly_feats


#include <vector>
#include <iostream>
#include <iomanip>

// Generate polynomial features of degree 2 (given)
std::vector<std::vector<double>> generatePolynomialFeatures(const std::vector<double>& data) {
    std::vector<std::vector<double>> polynomialFeatures;
    polynomialFeatures.reserve(data.size());
    for (double v : data) polynomialFeatures.push_back({v, v*v}); // [x, x^2]
    return polynomialFeatures;
}

int main() {
    std::vector<double> x = {-2.0, -1.0, 0.0, 1.5, 3.0};
    auto feats = generatePolynomialFeatures(x);

    std::cout << "x  -> [x, x^2]\n";
    for (size_t i = 0; i < x.size(); ++i) {
        std::cout << std::fixed << std::setprecision(2)
                  << x[i] << " -> [" << feats[i][0] << ", " << feats[i][1] << "]\n";
    }
    return 0;
}
