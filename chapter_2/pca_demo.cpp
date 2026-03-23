// pca_demo.cpp
// g++ -std=c++17 -O2 pca_demo.cpp -o pca_demo -I "$(brew --prefix eigen)/include/eigen3"
// ./pca_demo

#include <Eigen/Dense>
#include <iostream>
#include <iomanip>

Eigen::MatrixXd performPCA(const Eigen::MatrixXd& data, int numComponents) {
    Eigen::RowVectorXd mean = data.colwise().mean();          // Row vector
    Eigen::MatrixXd centered = data.rowwise() - mean;         // broadcast subtract
    Eigen::MatrixXd cov = (centered.adjoint() * centered) / double(data.rows() - 1);

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(cov);
    Eigen::MatrixXd eigenvectors = solver.eigenvectors().rightCols(numComponents);

    return centered * eigenvectors;                            // scores
}

int main() {
    // Toy dataset: 8 x 3
    Eigen::MatrixXd X(8, 3);
    X << 2.0,  4.1, 1.9,
         3.0,  6.0, 2.6,
         4.0,  8.2, 2.7,
         5.0, 10.1, 3.8,
         6.0, 12.2, 4.0,
         7.0, 13.9, 4.7,
         8.0, 16.0, 4.8,
         9.0, 17.8, 5.9;

    // Project to 2 PCs
    int k = 2;
    Eigen::MatrixXd Z = performPCA(X, k);

    // Explained variance ratios (fit on centered covariance)
    Eigen::RowVectorXd mu = X.colwise().mean();
    Eigen::MatrixXd centered = X.rowwise() - mu;
    Eigen::MatrixXd cov = (centered.adjoint() * centered) / double(X.rows() - 1);
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(cov);
    Eigen::VectorXd evals = es.eigenvalues(); // ascending
    double total = evals.sum();
    double evr1 = evals(evals.size()-1) / total;
    double evr2 = evals(evals.size()-2) / total;

    std::cout << "Projected shape: " << Z.rows() << " x " << Z.cols() << "\n\n";
    std::cout << "Scores (first 2 PCs):\n";
    for (int i = 0; i < Z.rows(); ++i) {
        std::cout << std::fixed << std::setprecision(5)
                  << "[" << std::setw(9) << Z(i,0) << ", " << std::setw(9) << Z(i,1) << "]\n";
    }
    std::cout << "\nExplained variance ratio:\n"
              << "  PC1: " << std::setprecision(4) << evr1 << "\n"
              << "  PC2: " << std::setprecision(4) << evr2 << "\n"
              << "  PC1+PC2: " << std::setprecision(4) << (evr1+evr2) << "\n";
    return 0;
}
