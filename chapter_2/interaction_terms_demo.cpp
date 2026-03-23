// interaction_terms_demo.cpp
// g++ -std=c++17 -O2 interaction_terms_demo.cpp -o inter_demo
// ./inter_demo

#include <vector>
#include <iostream>
#include <iomanip>

// Compute interaction terms between features (given)
std::vector<std::vector<double>> computeInteractionTerms(const std::vector<std::vector<double>>& features) {
    std::vector<std::vector<double>> interactionTerms(features.size());
    for (size_t i = 0; i < features.size(); ++i) {
        for (size_t j = 0; j < features[i].size(); ++j) {
            for (size_t k = j + 1; k < features[i].size(); ++k) {
                interactionTerms[i].push_back(features[i][j] * features[i][k]); // x_j * x_k
            }
        }
    }
    return interactionTerms;
}

int main() {
    // Each row is a sample with 3 features: [x1, x2, x3]
    std::vector<std::vector<double>> X = {
        {1.0, 2.0, 3.0},
        {0.5, -1.0, 4.0},
        {2.2, 0.0, 1.3},
        {-1.0, 3.5, -2.0}
    };

    auto Ix = computeInteractionTerms(X); // For 3 features: [x1*x2, x1*x3, x2*x3]

    std::cout << "Row\tFeatures\t\tInteractions [x1*x2, x1*x3, x2*x3]\n";
    for (size_t i = 0; i < X.size(); ++i) {
        std::cout << i << "\t[";
        for (size_t j = 0; j < X[i].size(); ++j) {
            std::cout << std::fixed << std::setprecision(2) << X[i][j]
                      << (j + 1 < X[i].size() ? ", " : "");
        }
        std::cout << "]\t->\t[";
        for (size_t j = 0; j < Ix[i].size(); ++j) {
            std::cout << std::fixed << std::setprecision(2) << Ix[i][j]
                      << (j + 1 < Ix[i].size() ? ", " : "");
        }
        std::cout << "]\n";
    }
    return 0;
}
