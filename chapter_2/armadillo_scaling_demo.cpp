// armadillo_scaling_demo.cpp
// g++ -std=c++17 -O2 armadillo_scaling_demo.cpp -o scale_demo \
//   $(pkg-config --cflags --libs armadillo)
// ./scale_demo

#include <armadillo>
#include <iostream>

int main() {
    using arma::mat; using arma::rowvec; using arma::uword; using arma::vec;

    // Toy data (rows = samples, cols = features)
    mat data = {
        {1, 10, 100},
        {2, 12, 120},
        {3,  9, 140},
        {4, 11, 160},
        {5,  8, 180}
    };
    std::cout << "Original:\n" << data << "\n";

    // ----- Min–max (per column) -----
    rowvec minv = arma::min(data, 0);
    rowvec maxv = arma::max(data, 0);
    rowvec range = maxv - minv; range.transform([](double r){ return r==0.0? 1.0 : r; });
    mat minmax = (data.each_row() - minv).each_row() / range;
    std::cout << "Min–max [0,1]:\n" << minmax << "\n";

    // ----- Z-score (per column) -----
    rowvec mu = arma::mean(data, 0);
    rowvec sd = arma::stddev(data, 0, 0);  // N-1
    sd.transform([](double s){ return s==0.0? 1.0 : s; });
    mat z = (data.each_row() - mu).each_row() / sd;
    std::cout << "Z-score:\n" << z << "\n";

    // ----- Robust: (x - median) / IQR (per column) -----
    rowvec med(data.n_cols), q1v(data.n_cols), q3v(data.n_cols);
    vec probs = {0.25, 0.75};
    for (uword j = 0; j < data.n_cols; ++j) {
        vec col = data.col(j);
        med(j) = arma::median(col);
        vec qs = arma::quantile(col, probs); // returns [Q1, Q3]
        q1v(j) = qs(0); q3v(j) = qs(1);
    }
    rowvec iqr = q3v - q1v; iqr.transform([](double v){ return v==0.0? 1.0 : v; });
    mat robust = (data.each_row() - med).each_row() / iqr;
    std::cout << "Robust (median/IQR):\n" << robust << "\n";

    // ----- Log1p & Power (sqrt) -----
    mat logt = arma::log(1.0 + data);
    mat powt = arma::pow(data, 0.5);
    std::cout << "Log1p:\n" << logt << "\n";
    std::cout << "Power 0.5 (sqrt):\n" << powt << "\n";
    return 0;
}
