// mlpack_impute_demo.cpp
// % g++ -std=c++17 mlpack_impute_demo.cpp -o impute \ 
//   $(pkg-config --cflags --libs mlpack armadillo) \
//   -I"$(brew --prefix cereal)/include" \
//   -Wno-deprecated-declarations
// % ./impute
#include <mlpack/core.hpp>
#include <iostream>
#include <limits>

using arma::mat; using arma::uword; using arma::vec;

void MeanImputeCols(mat& X) {
  for (uword j = 0; j < X.n_cols; ++j) {
    vec col = X.col(j);                               // copy to a vector
    arma::uvec good = arma::find_finite(col);
    if (good.is_empty()) continue;                    // all-missing column
    double mu = arma::mean(col(good));
    arma::uvec bad = arma::find_nonfinite(col);
    col(bad).fill(mu);                                // fill NaNs with mean
    X.col(j) = col;                                   // write back
  }
}

void ForwardFillCols(mat& X) {
  for (uword j = 0; j < X.n_cols; ++j) {
    vec col = X.col(j);
    double last = std::numeric_limits<double>::quiet_NaN();
    for (uword i = 0; i < col.n_rows; ++i) {
      if (std::isfinite(col(i))) last = col(i);
      else if (std::isfinite(last)) col(i) = last;    // forward-fill gap
    }
    X.col(j) = col;
  }
}

int main() {
  // Tiny toy matrix with NaNs (rows=samples, cols=features)
  mat X = {
    { 1.0, arma::datum::nan,  3.0 },
    { 2.0, 2.5,               arma::datum::nan },
    { arma::datum::nan, 2.0,  6.0 },
    { 4.0, 3.5,               9.0 }
  };

  std::cout << "Before:\n" << X << "\n";
  MeanImputeCols(X);
  ForwardFillCols(X);
  std::cout << "After:\n" << X << "\n";

  mlpack::data::Save("processed_data.csv", X);
  std::cout << "Saved: processed_data.csv\n";
}
