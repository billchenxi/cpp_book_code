// tsne_demo.cpp
// g++ -std=c++17 -O2 tsne_demo.cpp -o tsne_demo
// ./tsne_demo

#include <vector>
#include <iostream>
#include <iomanip>
#include <random>
#include <cmath>

// ---------- Real TSNE (if you have TSNE.h) or a mock fallback ----------
#ifdef USE_REAL_TSNE
  #include "TSNE.h"
#else
// Minimal mock: projects high-D data to 2D via random projection (demo only)
struct TSNE {
  void run(const std::vector<std::vector<double>>& data,
           std::vector<std::vector<double>>& reduced,
           int dimensions = 2) {
    if (data.empty() || data[0].empty() || dimensions <= 0) { reduced.clear(); return; }
    size_t N = data.size(), D = data[0].size();
    std::mt19937 rng(42);
    std::normal_distribution<double> nd(0.0, 1.0);

    // Random projection matrix D x dimensions
    std::vector<std::vector<double>> R(D, std::vector<double>(dimensions));
    for (size_t j = 0; j < D; ++j) for (int k = 0; k < dimensions; ++k) R[j][k] = nd(rng) / std::sqrt((double)D);

    // Multiply: reduced = data * R  => N x dimensions
    reduced.assign(N, std::vector<double>(dimensions, 0.0));
    for (size_t i = 0; i < N; ++i)
      for (size_t j = 0; j < D; ++j)
        for (int k = 0; k < dimensions; ++k)
          reduced[i][k] += data[i][j] * R[j][k];

    // Center for readability
    for (int k = 0; k < dimensions; ++k) {
      double mu = 0.0; for (size_t i = 0; i < N; ++i) mu += reduced[i][k]; mu /= (double)N;
      for (size_t i = 0; i < N; ++i) reduced[i][k] -= mu;
    }
  }
};
#endif

// API wrapper
std::vector<std::vector<double>> performTSNE(const std::vector<std::vector<double>>& data, int dimensions = 2) {
    TSNE tsne;
    std::vector<std::vector<double>> reducedData;
    tsne.run(data, reducedData, dimensions);
    return reducedData;
}

int main() {
    // --- Build a tiny high-D toy dataset: 3 clusters in 5D ---
    std::mt19937 rng(123);
    std::normal_distribution<double> noise(0.0, 0.2);

    auto make_cluster = [&](std::vector<double> center, int n){
        std::vector<std::vector<double>> out; out.reserve(n);
        for (int i=0;i<n;++i){
            std::vector<double> x = center;
            for (double &v : x) v += noise(rng);
            out.push_back(std::move(x));
        }
        return out;
    };

    std::vector<std::vector<double>> data;
    auto A = make_cluster({ 2.0,  2.0,  2.0,  2.0,  2.0}, 20);
    auto B = make_cluster({-2.0, -1.5, -2.0, -1.0, -1.5}, 20);
    auto C = make_cluster({ 0.0,  3.0, -3.0,  1.0, -2.0}, 20);
    data.insert(data.end(), A.begin(), A.end());
    data.insert(data.end(), B.begin(), B.end());
    data.insert(data.end(), C.begin(), C.end());

    // --- Run t-SNE (or mock) to 2D ---
    auto Y = performTSNE(data, 2);

    // --- Print a few points ---
    std::cout << "Reduced points (first 10 of " << Y.size() << "):\n";
    for (size_t i = 0; i < std::min<size_t>(10, Y.size()); ++i) {
        std::cout << std::fixed << std::setprecision(4)
                  << "[" << std::setw(8) << Y[i][0] << ", " << std::setw(8) << Y[i][1] << "]\n";
    }
    return 0;
}