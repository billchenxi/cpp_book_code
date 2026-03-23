#pragma once

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

struct CohortKey {
  std::string region;
  std::string device;
  std::string app;

  bool operator==(const CohortKey& o) const {
    return region == o.region && device == o.device && app == o.app;
  }
};

struct CohortKeyHash {
  size_t operator()(const CohortKey& k) const noexcept {
    std::hash<std::string> h;
    return h(k.region) ^ (h(k.device) << 1) ^ (h(k.app) << 2);
  }
};

struct PredRecord {
  std::string id;
  CohortKey cohort;
  float score;
  int decision;
  int64_t ts_ms;
};

struct RollingQuality {
  uint64_t n = 0;
  uint64_t n_pos = 0;
  double brier_sum = 0.0;
  uint64_t bins[10]{};
  uint64_t bin_pos[10]{};

  void observe(float score, int label) {
    n++;
    if (label) {
      n_pos++;
    }

    const double d = static_cast<double>(score) - static_cast<double>(label);
    brier_sum += d * d;

    const double clipped = std::min(0.999999, std::max(0.0, double(score)));
    const int b = std::min(9, std::max(0, int(clipped * 10)));
    bins[b]++;
    if (label) {
      bin_pos[b]++;
    }
  }

  double ece() const {
    if (n == 0) {
      return 0.0;
    }
    double e = 0.0;
    for (int b = 0; b < 10; ++b) {
      if (!bins[b]) {
        continue;
      }
      const double conf = (b + 0.5) / 10.0;
      const double acc = double(bin_pos[b]) / double(bins[b]);
      e += (double(bins[b]) / double(n)) * std::abs(acc - conf);
    }
    return e;
  }

  double brier() const { return n ? brier_sum / double(n) : 0.0; }
};

class DelayedLabelJoiner {
 public:
  void on_prediction(const PredRecord& pr) {
    std::lock_guard<std::mutex> lk(mu_);
    preds_[pr.id] = pr;
    prune();
  }

  void on_label(const std::string& id, int label) {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = preds_.find(id);
    if (it == preds_.end()) {
      return;
    }

    const PredRecord& pr = it->second;
    quality_[pr.cohort].observe(pr.score, label);
    preds_.erase(it);
  }

  RollingQuality get(const CohortKey& k) const {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = quality_.find(k);
    return (it == quality_.end()) ? RollingQuality{} : it->second;
  }

 private:
  void prune() {
    // Optional: drop stale predictions by TTL to bound memory.
  }

  mutable std::mutex mu_;
  std::unordered_map<std::string, PredRecord> preds_;
  std::unordered_map<CohortKey, RollingQuality, CohortKeyHash> quality_;
};

inline float softmax_entropy(const std::vector<float>& logits) {
  const float m = *std::max_element(logits.begin(), logits.end());
  double z = 0.0;
  std::vector<double> p(logits.size());

  for (size_t i = 0; i < logits.size(); ++i) {
    p[i] = std::exp(double(logits[i] - m));
    z += p[i];
  }

  double h = 0.0;
  for (double& pi : p) {
    pi /= z;
    if (pi > 0) {
      h -= pi * std::log(pi + 1e-12);
    }
  }
  return float(h);
}

inline float top2_margin(const std::vector<float>& probs) {
  float a = 0.0f;
  float b = 0.0f;
  for (float v : probs) {
    if (v > a) {
      b = a;
      a = v;
    } else if (v > b) {
      b = v;
    }
  }
  return a - b;
}

struct Disagreement {
  std::atomic<uint64_t> n{0};
  std::atomic<uint64_t> diff{0};

  void observe(int y_hat, int y_hat_shadow) {
    n.fetch_add(1, std::memory_order_relaxed);
    if (y_hat != y_hat_shadow) {
      diff.fetch_add(1, std::memory_order_relaxed);
    }
  }

  double rate() const {
    const uint64_t N = n.load(std::memory_order_relaxed);
    const uint64_t D = diff.load(std::memory_order_relaxed);
    return N ? double(D) / double(N) : 0.0;
  }
};
