// pca_mlpack4_demo.cpp
// g++ -std=c++17 -O2 pca_mlpack4_demo.cpp -o pca_demo \
//   $(pkg-config --cflags --libs mlpack armadillo) \
//   -Wno-deprecated-declarations
// ./pca_demo

#include <mlpack/core.hpp>
#include <mlpack/methods/pca/pca.hpp>
#include <armadillo>
#include <iostream>
#include <iomanip>

int main() {
  using arma::mat; using arma::vec;

  // Make a small 3D toy dataset (D x N; columns = samples)
  const size_t N = 200;
  vec t  = arma::linspace<vec>(0.0, 6.0, N);
  vec x1 = t + 0.1 * arma::randn<vec>(N);
  vec x2 = 2.0 * t + 0.1 * arma::randn<vec>(N);
  vec x3 = 0.5 * t + 0.1 * arma::randn<vec>(N);

  mat data(3, N);
  data.row(0) = x1.t();
  data.row(1) = x2.t();
  data.row(2) = x3.t();

  // PCA → keep 2 components (mlpack 4 API)
  mlpack::PCA pca;          // (optionally: mlpack::PCA(true) to scale variance)
  mat reduced;              // will be 2 x N
  pca.Apply(data, reduced, /*newDimension=*/2);

  std::cout << "Input shape:   " << data.n_rows << " x " << data.n_cols << "\n";
  std::cout << "Reduced shape: " << reduced.n_rows << " x " << reduced.n_cols << "\n\n";
  std::cout << "First 5 points after PCA:\n";
  for (size_t i = 0; i < 5; ++i) {
    std::cout << std::fixed << std::setprecision(4)
              << "[" << reduced(0,i) << ", " << reduced(1,i) << "]\n";
  }
  return 0;
}
