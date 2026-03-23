// fft_demo.cpp
// g++ -std=c++17 -O2 fft_demo.cpp -o fft_demo
// ./fft_demo

#include <vector>
#include <complex>
#include <iostream>
#include <iomanip>
#include <cmath>

inline double TWO_PI() { return 2.0 * std::acos(-1.0); }

// Naive DFT: X[k] = sum_t x[t] * e^{-i 2π t k / N}
std::vector<std::complex<double>> dft(const std::vector<double>& x) {
    const size_t N = x.size();
    std::vector<std::complex<double>> X(N, {0.0, 0.0});
    for (size_t k = 0; k < N; ++k) {
        std::complex<double> acc(0.0, 0.0);
        for (size_t t = 0; t < N; ++t) {
            double angle = TWO_PI() * t * k / N;
            std::complex<double> ph(std::cos(angle), -std::sin(angle)); // e^{-i angle}
            acc += x[t] * ph;
        }
        X[k] = acc;
    }
    return X;
}

int main() {
    // Test signal: sin(2π*3*t/N) + 0.5*sin(2π*7*t/N)
    const size_t N = 64;
    std::vector<double> x(N);
    for (size_t t = 0; t < N; ++t) {
        x[t] = std::sin(TWO_PI() * 3 * t / N) + 0.5 * std::sin(TWO_PI() * 7 * t / N);
    }

    auto X = dft(x);

    std::cout << "k : |X[k]| (first " << N/2 << " bins)\n";
    for (size_t k = 0; k < N/2; ++k) {
        std::cout << std::setw(2) << k << " : "
                  << std::fixed << std::setprecision(4) << std::abs(X[k]) << "\n";
    }
    return 0;
}
