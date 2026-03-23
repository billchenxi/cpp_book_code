// chapter_2/knn_imputation_multidim_example.cpp
// g++ -std=c++17 -O2 chapter_2/knn_imputation_multidim_example.cpp -o chapter_2/knn_multi
// ./chapter_2/knn_multi

#include <vector>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <limits>
#include <numeric>

using Matrix = std::vector<std::vector<double>>;
constexpr double NaN = std::numeric_limits<double>::quiet_NaN();

static inline bool finite(double x){ return std::isfinite(x); }

// Compute per-column mean/std ignoring NaNs
void col_stats(const Matrix& X, std::vector<double>& mean, std::vector<double>& stdev){
    size_t C = X[0].size();
    mean.assign(C, 0.0);
    stdev.assign(C, 0.0);
    std::vector<int> cnt(C, 0);

    for (const auto& row : X){
        for (size_t j=0;j<C;++j){
            if (finite(row[j])){ mean[j] += row[j]; cnt[j]++; }
        }
    }
    for (size_t j=0;j<C;++j){
        mean[j] = cnt[j] ? mean[j]/cnt[j] : 0.0;
    }
    // variance
    for (const auto& row : X){
        for (size_t j=0;j<C;++j){
            if (finite(row[j])){
                double d = row[j] - mean[j];
                stdev[j] += d*d;
            }
        }
    }
    for (size_t j=0;j<C;++j){
        stdev[j] = cnt[j] > 1 ? std::sqrt(stdev[j]/(cnt[j]-1)) : 1.0;
        if (stdev[j] == 0.0) stdev[j] = 1.0; // avoid div-by-zero
    }
}

// Z-score scale for distance calculation (keeps NaNs)
Matrix zscore(const Matrix& X){
    Matrix Z = X;
    std::vector<double> mu, sd;
    col_stats(X, mu, sd);
    for (auto& row : Z){
        for (size_t j=0;j<row.size();++j){
            if (finite(row[j])) row[j] = (row[j]-mu[j])/sd[j];
        }
    }
    return Z;
}

// Overlap-aware Euclidean distance between two rows
double overlap_distance(const std::vector<double>& a, const std::vector<double>& b){
    double s2 = 0.0; int used = 0;
    for (size_t j=0;j<a.size();++j){
        if (finite(a[j]) && finite(b[j])){
            double d = a[j] - b[j];
            s2 += d*d; used++;
        }
    }
    // If no overlapping features, return "far" distance
    return used ? std::sqrt(s2) : std::numeric_limits<double>::infinity();
}

// Return indices of k nearest neighbors (excluding target)
std::vector<int> knn_indices(const Matrix& Z, int target, int k){
    std::vector<std::pair<double,int>> dists;
    dists.reserve(Z.size()-1);
    for (size_t i=0;i<Z.size();++i){
        if ((int)i == target) continue;
        double d = overlap_distance(Z[target], Z[i]);
        dists.emplace_back(d, (int)i);
    }
    std::sort(dists.begin(), dists.end(),
              [](auto& A, auto& B){ return A.first < B.first; });
    std::vector<int> idx;
    for (int i=0; i<(int)dists.size() && (int)idx.size()<k; ++i){
        if (std::isfinite(dists[i].first)) idx.push_back(dists[i].second);
    }
    return idx;
}

// Distance-weighted mean from neighbors for a specific column
double weighted_mean_col(const Matrix& X, const Matrix& Z,
                         int target, int col, const std::vector<int>& nbrs){
    double num = 0.0, den = 0.0;
    for (int ni : nbrs){
        double v = X[ni][col];
        if (!finite(v)) continue; // neighbor missing this column
        double d = overlap_distance(Z[target], Z[ni]);
        double w = 1.0 / (d + 1e-9); // inverse-distance weight
        num += w * v;
        den += w;
    }
    return (den > 0.0) ? (num/den) : NaN;
}

// Impute all rows with missing values
Matrix knn_impute_all(const Matrix& X, int k){
    Matrix Y = X;
    Matrix Z = zscore(X); // for distances only
    for (size_t i=0;i<Y.size();++i){
        // Check if row i has any NaN
        bool has_nan = false;
        for (double v : Y[i]) if (!finite(v)) { has_nan = true; break; }
        if (!has_nan) continue;

        auto nbrs = knn_indices(Z, (int)i, k);
        for (size_t j=0;j<Y[i].size();++j){
            if (!finite(Y[i][j])){
                double m = weighted_mean_col(X, Z, (int)i, (int)j, nbrs);
                Y[i][j] = m; // may remain NaN if no neighbor has this col
            }
        }
    }
    return Y;
}

void print_matrix(const Matrix& X, const char* title){
    std::cout << title << "\n";
    for (size_t i=0;i<X.size();++i){
        std::cout << std::setw(2) << i << ": [";
        for (size_t j=0;j<X[i].size();++j){
            if (finite(X[i][j])) std::cout << std::fixed << std::setprecision(3) << X[i][j];
            else std::cout << "NaN";
            if (j+1<X[i].size()) std::cout << ", ";
        }
        std::cout << "]\n";
    }
    std::cout << std::endl;
}

int main(){
    // Build a 20x5 dataset with mixed scales and several NaNs
    Matrix X(20, std::vector<double>(5, 0.0));
    for (int i=0;i<20;++i){
        X[i][0] = 0.5 + i;                 // roughly linear
        X[i][1] = 1.0 + 2.0 * (i % 5);     // periodic-ish categorical-like numeric
        X[i][2] = 100.0 + 0.3 * i;         // larger scale
        X[i][3] = 50.0 + std::sin(i)*5.0;  // wavy
        X[i][4] = 0.1 * i * i;             // quadratic
    }
    // Inject some missing values (row, col)
    std::vector<std::pair<int,int>> holes = {
        {1,2},{3,1},{4,4},{6,0},{7,3},{9,2},{12,1},{15,4},{18,0},{19,3},
        {2,4},{5,3},{8,1},{10,0},{11,2},{13,4},{14,3}
    };
    for (auto [r,c] : holes) X[r][c] = NaN;

    print_matrix(X, "Before (20x5 with NaNs):");

    int k = 5;
    Matrix Y = knn_impute_all(X, k);

    print_matrix(Y, "After KNN imputation (k=5):");

    return 0;
}
