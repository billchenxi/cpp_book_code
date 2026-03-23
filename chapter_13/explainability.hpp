#pragma once

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

namespace chapter13 {

struct Model {
  virtual ~Model() = default;

  virtual double score(const Eigen::VectorXd& x) const = 0;

  virtual void score_batch(const std::vector<Eigen::VectorXd>& X,
                           std::vector<double>& out) const {
    out.resize(X.size());
    for (size_t i = 0; i < X.size(); ++i) {
      out[i] = score(X[i]);
    }
  }
};

struct LimeConfig {
  int n_samples = 512;
  int top_k = 8;
  double ridge_lambda = 1e-3;
  double kernel_width = 0.75;
  unsigned seed = 42;
};

struct LimeExplanation {
  std::vector<int> feat_index;
  std::vector<double> feat_weight;
  double intercept = 0.0;
};

struct ShapConfig {
  int n_coalitions = 2048;
  int background_K = 64;
  double ridge_lambda = 1e-6;
  unsigned seed = 1337;
};

struct ShapValues {
  std::vector<double> phi;
  double phi0 = 0.0;
};

inline Eigen::VectorXd clamp_sigma(const Eigen::VectorXd& sigma) {
  Eigen::VectorXd out = sigma;
  for (Eigen::Index i = 0; i < out.size(); ++i) {
    if (std::abs(out[i]) < 1e-12) {
      out[i] = 1.0;
    }
  }
  return out;
}

inline std::vector<Eigen::VectorXd> sample_lime_points(
    const Eigen::VectorXd& x0, const Eigen::VectorXd& mu,
    const Eigen::VectorXd& sigma, int n_samples, unsigned seed = 42) {
  using Eigen::VectorXd;

  if (x0.size() != mu.size() || x0.size() != sigma.size()) {
    throw std::invalid_argument("sample_lime_points: incompatible dimensions");
  }

  const int d = static_cast<int>(x0.size());
  const VectorXd safe_sigma = clamp_sigma(sigma);
  const VectorXd z0 = (x0 - mu).cwiseQuotient(safe_sigma);

  std::vector<VectorXd> Xs;
  Xs.reserve(static_cast<size_t>(std::max(0, n_samples)));

  std::mt19937 rng(seed);
  std::normal_distribution<double> n01(0.0, 1.0);
  for (int i = 0; i < n_samples; ++i) {
    VectorXd z = z0;
    for (int j = 0; j < d; ++j) {
      z[j] += 0.2 * n01(rng);
    }
    Xs.push_back(z.cwiseProduct(safe_sigma) + mu);
  }
  return Xs;
}

inline std::vector<double> score_lime_batch(
    const Model& model, const std::vector<Eigen::VectorXd>& Xs) {
  std::vector<double> ys;
  model.score_batch(Xs, ys);
  return ys;
}

inline Eigen::VectorXd fit_lime_ridge(
    const std::vector<Eigen::VectorXd>& Xs, const std::vector<double>& ys,
    const Eigen::VectorXd& x0, const Eigen::VectorXd& mu,
    const Eigen::VectorXd& sigma, double kernel_width, double ridge_lambda) {
  using Eigen::MatrixXd;
  using Eigen::VectorXd;

  if (Xs.empty() || ys.empty() || Xs.size() != ys.size()) {
    throw std::invalid_argument("fit_lime_ridge: invalid sample set");
  }

  const int n = static_cast<int>(Xs.size());
  const int d = static_cast<int>(x0.size());
  const VectorXd safe_sigma = clamp_sigma(sigma);

  MatrixXd X(n, d + 1);
  VectorXd y(n);
  VectorXd w(n);
  const VectorXd z0 = (x0 - mu).cwiseQuotient(safe_sigma);

  for (int i = 0; i < n; ++i) {
    if (Xs[i].size() != x0.size()) {
      throw std::invalid_argument("fit_lime_ridge: inconsistent feature size");
    }
    const VectorXd zi = (Xs[i] - mu).cwiseQuotient(safe_sigma);
    const double dist2 = (zi - z0).squaredNorm();
    w[i] = std::exp(-dist2 / (kernel_width * kernel_width));
    X(i, 0) = 1.0;
    for (int j = 0; j < d; ++j) {
      X(i, j + 1) = Xs[i][j];
    }
    y[i] = ys[i];
  }

  MatrixXd W = w.asDiagonal();
  MatrixXd XtWX = X.transpose() * W * X;
  for (int j = 1; j <= d; ++j) {
    XtWX(j, j) += ridge_lambda;
  }
  return XtWX.ldlt().solve(X.transpose() * W * y);
}

inline LimeExplanation build_lime_explanation(const Eigen::VectorXd& beta,
                                              int top_k) {
  struct Item {
    int j;
    double w;
  };

  std::vector<Item> items;
  items.reserve(beta.size() > 1 ? static_cast<size_t>(beta.size() - 1) : 0);
  for (Eigen::Index j = 1; j < beta.size(); ++j) {
    items.push_back({static_cast<int>(j - 1), beta[j]});
  }

  const int keep = std::min(top_k, static_cast<int>(items.size()));
  std::partial_sort(items.begin(), items.begin() + keep, items.end(),
                    [](const Item& a, const Item& b) {
                      return std::fabs(a.w) > std::fabs(b.w);
                    });

  LimeExplanation out;
  out.intercept = beta.size() > 0 ? beta[0] : 0.0;
  for (int k = 0; k < keep; ++k) {
    out.feat_index.push_back(items[k].j);
    out.feat_weight.push_back(items[k].w);
  }
  return out;
}

inline std::vector<uint8_t> sample_mask(int M, std::mt19937& rng) {
  std::uniform_int_distribution<int> size_dist(1, std::max(1, M - 1));
  const int k = size_dist(rng);

  std::vector<uint8_t> z(static_cast<size_t>(M), 0);
  std::vector<int> idx(static_cast<size_t>(M));
  std::iota(idx.begin(), idx.end(), 0);
  std::shuffle(idx.begin(), idx.end(), rng);
  for (int i = 0; i < k; ++i) {
    z[static_cast<size_t>(idx[i])] = 1;
  }
  return z;
}

inline void build_kernelshap_design(
    int M, const ShapConfig& cfg, Eigen::MatrixXd& Z, Eigen::VectorXd& w,
    std::vector<std::vector<uint8_t>>& masks) {
  Z.resize(cfg.n_coalitions, M + 1);
  w.resize(cfg.n_coalitions);
  masks.clear();
  masks.reserve(static_cast<size_t>(cfg.n_coalitions));

  std::mt19937 rng(cfg.seed);
  auto comb = [](int n, int k) -> double {
    if (k < 0 || k > n) {
      return 0.0;
    }
    k = std::min(k, n - k);
    double c = 1.0;
    for (int i = 1; i <= k; ++i) {
      c = c * (n - k + i) / i;
    }
    return c;
  };

  for (int i = 0; i < cfg.n_coalitions; ++i) {
    auto z = sample_mask(M, rng);
    const int k = std::accumulate(z.begin(), z.end(), 0);
    const double omega =
        (M > 1) ? static_cast<double>(M - 1) / (comb(M, k) * k * (M - k))
                : 1.0;
    masks.push_back(z);
    w[i] = std::max(omega, 1e-12);
    Z(i, 0) = 1.0;
    for (int j = 0; j < M; ++j) {
      Z(i, j + 1) = static_cast<double>(z[static_cast<size_t>(j)]);
    }
  }
}

inline std::vector<Eigen::VectorXd> build_masked_batch(
    const Eigen::VectorXd& x,
    const std::vector<Eigen::VectorXd>& background,
    const std::vector<std::vector<uint8_t>>& masks) {
  if (background.empty()) {
    throw std::invalid_argument("build_masked_batch: background is empty");
  }

  std::vector<Eigen::VectorXd> batch;
  batch.reserve(masks.size());
  for (size_t i = 0; i < masks.size(); ++i) {
    const auto& z = masks[i];
    const auto& b = background[i % background.size()];
    if (b.size() != x.size()) {
      throw std::invalid_argument(
          "build_masked_batch: background dimension mismatch");
    }
    Eigen::VectorXd xmask = x;
    for (Eigen::Index j = 0; j < x.size(); ++j) {
      if (!z[static_cast<size_t>(j)]) {
        xmask[j] = b[j];
      }
    }
    batch.push_back(std::move(xmask));
  }
  return batch;
}

inline std::vector<double> score_kernelshap_batch(
    const Model& model, const std::vector<Eigen::VectorXd>& batch) {
  std::vector<double> y;
  model.score_batch(batch, y);
  return y;
}

inline ShapValues solve_kernelshap(const Eigen::MatrixXd& Z,
                                   const Eigen::VectorXd& w,
                                   const std::vector<double>& y,
                                   double ridge_lambda) {
  using Eigen::Map;
  using Eigen::MatrixXd;
  using Eigen::VectorXd;

  if (Z.rows() != w.size() ||
      Z.rows() != static_cast<Eigen::Index>(y.size())) {
    throw std::invalid_argument("solve_kernelshap: incompatible dimensions");
  }

  const VectorXd vy =
      Map<const VectorXd>(y.data(), static_cast<Eigen::Index>(y.size()));
  MatrixXd W = w.asDiagonal();
  MatrixXd ZtWZ = Z.transpose() * W * Z;
  for (Eigen::Index j = 0; j < Z.cols(); ++j) {
    ZtWZ(j, j) += ridge_lambda;
  }
  const VectorXd theta = ZtWZ.ldlt().solve(Z.transpose() * W * vy);

  ShapValues out;
  out.phi0 = theta[0];
  out.phi.resize(static_cast<size_t>(Z.cols() - 1));
  for (size_t j = 0; j < out.phi.size(); ++j) {
    out.phi[j] = theta[static_cast<Eigen::Index>(j + 1)];
  }
  return out;
}

inline double shap_sum(const ShapValues& shap) {
  return shap.phi0 +
         std::accumulate(shap.phi.begin(), shap.phi.end(), 0.0);
}

}  // namespace chapter13
