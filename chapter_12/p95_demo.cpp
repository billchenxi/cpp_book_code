#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

double p95(std::vector<double>& samples) {
  if (samples.empty()) {
    return 0.0;
  }
  const auto k = static_cast<size_t>(std::ceil(0.95 * samples.size())) - 1;
  std::nth_element(samples.begin(), samples.begin() + k, samples.end());
  return samples[k];
}

int main() {
  std::vector<double> samples{14, 15, 15, 16, 17, 18, 20, 21, 22, 24,
                              25, 28, 31, 33, 38, 45, 54, 66, 91, 120};
  std::cout << "p95_ms=" << p95(samples) << "\n";
  return 0;
}
