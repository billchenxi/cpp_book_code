#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#include "explainability.hpp"

namespace {

class ToyRiskModel : public chapter13::Model {
 public:
  ToyRiskModel() : weights_(6) {
    weights_ << 1.25, -0.85, 0.55, 1.4, -1.1, 0.7;
  }

  double score(const Eigen::VectorXd& x) const override {
    const double interaction = 0.18 * x[0] * x[3] - 0.10 * x[1] * x[4];
    return weights_.dot(x) + bias_ + interaction;
  }

 private:
  Eigen::VectorXd weights_;
  double bias_ = -0.2;
};

std::vector<Eigen::VectorXd> make_background(const Eigen::VectorXd& mu,
                                             const Eigen::VectorXd& sigma,
                                             int count, unsigned seed) {
  std::mt19937 rng(seed);
  std::normal_distribution<double> n01(0.0, 1.0);

  std::vector<Eigen::VectorXd> background;
  background.reserve(static_cast<size_t>(count));
  const Eigen::VectorXd safe_sigma = chapter13::clamp_sigma(sigma);
  for (int i = 0; i < count; ++i) {
    Eigen::VectorXd x(mu.size());
    for (Eigen::Index j = 0; j < mu.size(); ++j) {
      x[j] = mu[j] + 0.65 * safe_sigma[j] * n01(rng);
    }
    background.push_back(std::move(x));
  }
  return background;
}

void print_ranked_weights(const std::vector<double>& weights) {
  struct Item {
    int index;
    double value;
  };

  std::vector<Item> items;
  items.reserve(weights.size());
  for (size_t i = 0; i < weights.size(); ++i) {
    items.push_back({static_cast<int>(i), weights[i]});
  }

  std::sort(items.begin(), items.end(), [](const Item& a, const Item& b) {
    return std::fabs(a.value) > std::fabs(b.value);
  });

  for (const auto& item : items) {
    std::cout << "  feature[" << item.index << "] = " << std::showpos
              << item.value << std::noshowpos << "\n";
  }
}

}  // namespace

int main() {
  const Eigen::VectorXd mu =
      (Eigen::VectorXd(6) << 0.0, 1.2, -0.4, 0.3, 1.8, -1.1).finished();
  const Eigen::VectorXd sigma =
      (Eigen::VectorXd(6) << 1.0, 0.8, 1.1, 0.7, 0.9, 1.2).finished();
  const Eigen::VectorXd x0 =
      (Eigen::VectorXd(6) << 1.1, 0.1, 0.5, 1.6, 1.0, -0.2).finished();

  ToyRiskModel model;

  chapter13::LimeConfig lime_cfg;
  lime_cfg.n_samples = 768;
  lime_cfg.top_k = 4;
  lime_cfg.kernel_width = 0.9;

  const auto lime_points = chapter13::sample_lime_points(
      x0, mu, sigma, lime_cfg.n_samples, lime_cfg.seed);
  const auto lime_scores = chapter13::score_lime_batch(model, lime_points);
  const auto beta = chapter13::fit_lime_ridge(
      lime_points, lime_scores, x0, mu, sigma, lime_cfg.kernel_width,
      lime_cfg.ridge_lambda);
  const auto lime =
      chapter13::build_lime_explanation(beta, lime_cfg.top_k);

  chapter13::ShapConfig shap_cfg;
  shap_cfg.n_coalitions = 2048;
  shap_cfg.background_K = 64;

  const auto background =
      make_background(mu, sigma, shap_cfg.background_K, shap_cfg.seed);

  Eigen::MatrixXd Z;
  Eigen::VectorXd w;
  std::vector<std::vector<uint8_t>> masks;
  chapter13::build_kernelshap_design(static_cast<int>(x0.size()), shap_cfg, Z,
                                     w, masks);

  const auto masked_batch =
      chapter13::build_masked_batch(x0, background, masks);
  const auto shap_scores =
      chapter13::score_kernelshap_batch(model, masked_batch);
  const auto shap = chapter13::solve_kernelshap(Z, w, shap_scores,
                                                shap_cfg.ridge_lambda);

  const double pred = model.score(x0);
  const double shap_total = chapter13::shap_sum(shap);

  std::cout << std::fixed << std::setprecision(4);
  std::cout << "Prediction score: " << pred << "\n\n";

  std::cout << "LIME explanation\n";
  std::cout << "  intercept = " << lime.intercept << "\n";
  for (size_t i = 0; i < lime.feat_index.size(); ++i) {
    std::cout << "  feature[" << lime.feat_index[i]
              << "] = " << std::showpos << lime.feat_weight[i]
              << std::noshowpos << "\n";
  }

  std::cout << "\nKernelSHAP explanation\n";
  std::cout << "  phi0 = " << shap.phi0 << "\n";
  print_ranked_weights(shap.phi);
  std::cout << "  additivity residual = " << (pred - shap_total) << "\n";

  return 0;
}
