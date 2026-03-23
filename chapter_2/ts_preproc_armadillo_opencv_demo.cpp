// ts_preproc_armadillo_opencv_demo.cpp
// g++ -std=c++17 -O2 ts_preproc_armadillo_opencv_demo.cpp -o ts_demo \
//   $(pkg-config --cflags --libs armadillo opencv4)
// ./ts_demo

#include <armadillo>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <iomanip>

// Rolling mean via conv + "same" trim (odd window size)
arma::mat rollingMeanSame(const arma::mat& X, arma::uword w) {
    if (w % 2 == 0 || w < 1) throw std::runtime_error("window must be odd and >=1");
    arma::mat out(X.n_rows, X.n_cols, arma::fill::zeros);
    arma::vec k = arma::ones<arma::vec>(w) / double(w);
    for (arma::uword j = 0; j < X.n_cols; ++j) {
        arma::vec full = arma::conv(X.col(j), k); // length = N + w - 1
        arma::uword start = (w - 1) / 2;
        out.col(j) = full.rows(start, start + X.n_rows - 1);
    }
    return out;
}

// Exponential smoothing (per column)
arma::mat expSmooth(const arma::mat& X, double alpha) {
    arma::mat Y = X;
    for (arma::uword j = 0; j < X.n_cols; ++j)
        for (arma::uword i = 1; i < X.n_rows; ++i)
            Y(i,j) = alpha * X(i,j) + (1.0 - alpha) * Y(i-1,j);
    return Y;
}

// First difference (per column)
arma::mat firstDifference(const arma::mat& X) {
    return X.rows(1, X.n_rows - 1) - X.rows(0, X.n_rows - 2);
}

// 1-D FFT on a single column via OpenCV; returns magnitude spectrum
arma::vec fftMagnitude(const arma::vec& x) {
    cv::Mat in(x.n_rows, 1, CV_64F);
    for (int i = 0; i < in.rows; ++i) in.at<double>(i,0) = x(i);

    cv::Mat X; // complex output
    cv::dft(in, X, cv::DFT_COMPLEX_OUTPUT);

    // magnitude = sqrt(Re^2 + Im^2)
    std::vector<cv::Mat> planes; cv::split(X, planes);
    cv::Mat mag;
    cv::magnitude(planes[0], planes[1], mag);

    arma::vec out(mag.rows);
    for (int i = 0; i < mag.rows; ++i) out(i) = mag.at<double>(i,0);
    return out;
}

int main() {
    using std::cout;

    // Toy time series: N x D (rows = time, cols = features)
    const arma::uword N = 32, D = 2;
    arma::mat data(N, D);
    arma::vec t = arma::linspace<arma::vec>(0.0, 2.0 * arma::datum::pi, N);

    // Feature 0: trend + sine; Feature 1: just sine with phase
    data.col(0) = 0.2 * t + arma::sin(3 * t);
    data.col(1) = arma::sin(5 * t + 0.5);

    auto roll = rollingMeanSame(data, /*window=*/5);
    auto smth = expSmooth(data, /*alpha=*/0.3);
    auto diff = firstDifference(data);

    // FFT magnitude of first column (just to illustrate)
    arma::vec mag = fftMagnitude(data.col(0));

    cout << std::fixed << std::setprecision(3);
    cout << "Original (first 6 rows):\n" << data.rows(0,5) << "\n";
    cout << "Rolling mean (w=5, first 6 rows):\n" << roll.rows(0,5) << "\n";
    cout << "Exp smoothing (alpha=0.3, first 6 rows):\n" << smth.rows(0,5) << "\n";
    cout << "First difference (first 5 rows):\n" << diff.rows(0,4) << "\n";

    cout << "FFT |X[k]| of column 0 (first 10 bins):\n";
    for (int k = 0; k < 10 && k < (int)mag.n_rows; ++k)
        cout << "k=" << k << " : " << mag(k) << "\n";

    return 0;
}
