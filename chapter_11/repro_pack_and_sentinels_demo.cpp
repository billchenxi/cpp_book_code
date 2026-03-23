#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

struct ReproPack {
  std::string model_ver;
  std::string preprocess_ver;
  std::string device;
  std::string flags;
  uint64_t seed;
};

inline std::mt19937 make_rng(uint64_t seed) {
  return std::mt19937{static_cast<unsigned>(seed)};
}

template <typename T>
bool has_nan_inf(const T* data, size_t n) {
  for (size_t i = 0; i < n; ++i) {
    if (!std::isfinite(static_cast<double>(data[i]))) {
      return true;
    }
  }
  return false;
}

float stable_logsumexp(const float* x, int n) {
  float m = x[0];
  for (int i = 1; i < n; ++i) {
    m = std::max(m, x[i]);
  }
  double acc = 0.0;
  for (int i = 0; i < n; ++i) {
    acc += std::exp(static_cast<double>(x[i] - m));
  }
  return m + static_cast<float>(std::log(acc));
}

int main() {
  ReproPack pack{"resnet50@1.12.3", "prep@4", "cpu", "shadow=false", 424242};
  auto rng = make_rng(pack.seed);
  std::uniform_real_distribution<float> dist(-2.0f, 2.0f);

  std::vector<float> logits(5);
  for (float& v : logits) {
    v = dist(rng);
  }

  std::cout << "repro_pack model=" << pack.model_ver
            << " preprocess=" << pack.preprocess_ver
            << " device=" << pack.device
            << " seed=" << pack.seed << "\n";
  std::cout << "stable_logsumexp=" << std::fixed << std::setprecision(6)
            << stable_logsumexp(logits.data(), static_cast<int>(logits.size()))
            << "\n";

  std::vector<float> bad_values{
      1.0f, std::numeric_limits<float>::infinity(),
      std::numeric_limits<float>::quiet_NaN()};
  std::cout << "has_nan_inf(logits)="
            << (has_nan_inf(logits.data(), logits.size()) ? "true" : "false")
            << "\n";
  std::cout << "has_nan_inf(bad_values)="
            << (has_nan_inf(bad_values.data(), bad_values.size()) ? "true"
                                                                  : "false")
            << "\n";
  return 0;
}
