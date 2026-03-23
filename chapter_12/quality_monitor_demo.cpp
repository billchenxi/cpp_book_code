#include "cohort_quality.hpp"
#include "quality.hpp"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

static std::vector<float> softmax_probs(const std::vector<float>& logits) {
  float m = logits.empty() ? 0.0f : logits.front();
  for (float v : logits) {
    if (v > m) {
      m = v;
    }
  }

  double z = 0.0;
  std::vector<double> tmp(logits.size());
  for (size_t i = 0; i < logits.size(); ++i) {
    tmp[i] = std::exp(double(logits[i] - m));
    z += tmp[i];
  }

  std::vector<float> probs(logits.size(), 0.0f);
  for (size_t i = 0; i < logits.size(); ++i) {
    probs[i] = static_cast<float>(tmp[i] / z);
  }
  return probs;
}

int main() {
  CohortKey eu_ios{"EU", "ios", "4.9"};
  CohortKey na_android{"NA", "android", "5.1"};

  DelayedLabelJoiner joiner;
  CohortMonitor monitor;
  Disagreement disagree;

  const std::vector<std::vector<float>> logits_samples{
      {2.4f, 1.1f, 0.2f},
      {1.9f, 1.7f, 0.4f},
      {0.8f, 0.7f, 0.6f},
      {2.0f, 0.3f, -0.1f}};
  const std::vector<int> labels{1, 1, 0, 1};
  const std::vector<int> shadow_preds{1, 0, 0, 1};

  for (size_t i = 0; i < logits_samples.size(); ++i) {
    const auto probs = softmax_probs(logits_samples[i]);
    const float score = probs[0];
    const int pred = score >= 0.5f ? 1 : 0;

    const CohortKey cohort = (i % 2 == 0) ? eu_ios : na_android;
    const float entropy = softmax_entropy(logits_samples[i]);
    const float margin = top2_margin(probs);
    const bool abstain = margin < 0.15f;

    monitor.on_request(cohort, entropy, margin, abstain);
    monitor.on_shadow_pair(cohort, pred, shadow_preds[i]);
    disagree.observe(pred, shadow_preds[i]);

    joiner.on_prediction(
        {"req-" + std::to_string(i), cohort, score, pred, int64_t(1000 + i)});
    joiner.on_label("req-" + std::to_string(i), labels[i]);
  }

  const auto eu_quality = joiner.get(eu_ios);
  const auto na_quality = joiner.get(na_android);
  const auto eu_leading = monitor.get(eu_ios);
  const auto na_leading = monitor.get(na_android);

  std::cout << "global_disagreement_rate=" << disagree.rate() << "\n";
  std::cout << "eu_ios ece=" << eu_quality.ece()
            << " brier=" << eu_quality.brier()
            << " ent_ewma=" << eu_leading.ent_ewma
            << " margin_ewma=" << eu_leading.margin_ewma
            << " disagree_ewma=" << eu_leading.disagree_ewma
            << " abstain_ewma=" << eu_leading.abstain_ewma << "\n";
  std::cout << "na_android ece=" << na_quality.ece()
            << " brier=" << na_quality.brier()
            << " ent_ewma=" << na_leading.ent_ewma
            << " margin_ewma=" << na_leading.margin_ewma
            << " disagree_ewma=" << na_leading.disagree_ewma
            << " abstain_ewma=" << na_leading.abstain_ewma << "\n";
  return 0;
}
