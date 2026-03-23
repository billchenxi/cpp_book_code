// chapter_2/poly_interact_armadillo_demo.cpp
// g++ -std=c++17 -O2 chapter_2/poly_interact_armadillo_demo.cpp -o chapter_2/poly_demo \
//   $(pkg-config --cflags --libs armadillo)
// ./chapter_2/poly_demo

#include <armadillo>
#include <iostream>

int main() {
    using arma::mat; using arma::uword;

    // Toy data: 4 samples x 3 features
    mat data = {
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9},
        {2, 3, 4}
    };

    // Polynomial features: [X, X.^2]
    mat polyFeatures(data.n_rows, data.n_cols * 2);
    polyFeatures.cols(0, data.n_cols - 1) = data;
    polyFeatures.cols(data.n_cols, polyFeatures.n_cols - 1) = arma::square(data);

    // Interaction terms: all pairwise x_i * x_j (i<j)
    mat interactionTerms(data.n_rows, data.n_cols * (data.n_cols - 1) / 2);
    uword col = 0;
    for (uword i = 0; i < data.n_cols; ++i)
        for (uword j = i + 1; j < data.n_cols; ++j)
            interactionTerms.col(col++) = data.col(i) % data.col(j);

    std::cout << "Data (" << data.n_rows << "x" << data.n_cols << "):\n" << data << "\n";
    std::cout << "Poly [X | X.^2] (" << polyFeatures.n_rows << "x" << polyFeatures.n_cols << "):\n"
              << polyFeatures << "\n";
    std::cout << "Interactions x_i*x_j (" << interactionTerms.n_rows << "x" << interactionTerms.n_cols << "):\n"
              << interactionTerms << "\n";
    return 0;
}
