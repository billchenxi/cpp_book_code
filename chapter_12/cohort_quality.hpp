#pragma once

#include "quality.hpp"

#include <mutex>
#include <unordered_map>

struct CohortLeading {
  double ent_ewma = 0.0;
  double margin_ewma = 0.0;
  double disagree_ewma = 0.0;
  double abstain_ewma = 0.0;
  bool inited = false;

  void update(double ent,
              double margin,
              double disagree,
              double abstain,
              double alpha = 0.2) {
    if (!inited) {
      ent_ewma = ent;
      margin_ewma = margin;
      disagree_ewma = disagree;
      abstain_ewma = abstain;
      inited = true;
      return;
    }

    ent_ewma = (1 - alpha) * ent_ewma + alpha * ent;
    margin_ewma = (1 - alpha) * margin_ewma + alpha * margin;
    disagree_ewma = (1 - alpha) * disagree_ewma + alpha * disagree;
    abstain_ewma = (1 - alpha) * abstain_ewma + alpha * abstain;
  }
};

class CohortMonitor {
 public:
  void on_request(const CohortKey& k,
                  double entropy,
                  double margin,
                  bool abstain) {
    std::lock_guard<std::mutex> lk(mu_);
    auto& c = lead_[k];
    c.update(entropy, margin, /*disagree=*/0.0, abstain ? 1.0 : 0.0);
  }

  void on_shadow_pair(const CohortKey& k, int y_hat, int y_hat_shadow) {
    std::lock_guard<std::mutex> lk(mu_);
    auto& c = lead_[k];
    const double disagree = (y_hat != y_hat_shadow) ? 1.0 : 0.0;
    c.update(c.ent_ewma, c.margin_ewma, disagree, c.abstain_ewma);
  }

  CohortLeading get(const CohortKey& k) const {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = lead_.find(k);
    return (it == lead_.end()) ? CohortLeading{} : it->second;
  }

 private:
  mutable std::mutex mu_;
  std::unordered_map<CohortKey, CohortLeading, CohortKeyHash> lead_;
};
