// knn_imputation_example.cpp
// g++ -std=c++17 -O2 knn_imputation_example.cpp -o knn_imp
// ./knn_imp


#include <vector>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <limits>

// Dummy KNN imputation example based on Euclidean distance
std::vector<double> knnImputation(const std::vector<std::vector<double>>& dataset, int k, int targetIndex) {
    std::vector<double> imputedRow = dataset[targetIndex];

    for (size_t col = 0; col < imputedRow.size(); ++col) {
        if (std::isnan(imputedRow[col])) {
            std::vector<std::pair<double, int>> distances;

            for (size_t i = 0; i < dataset.size(); ++i) {
                if (i != static_cast<size_t>(targetIndex)) {
                    double dist = 0.0;
                    for (size_t j = 0; j < dataset[i].size(); ++j) {
                        if (!std::isnan(dataset[targetIndex][j]) && !std::isnan(dataset[i][j])) {
                            double d = dataset[targetIndex][j] - dataset[i][j];
                            dist += d * d;
                        }
                    }
                    distances.emplace_back(std::sqrt(dist), static_cast<int>(i));
                }
            }

            std::sort(distances.begin(), distances.end()); // ascending by distance

            double valueSum = 0.0;
            int count = 0;
            for (int i = 0; i < k && i < static_cast<int>(distances.size()); ++i) {
                // NOTE: This simple version assumes neighbors have a value for this column.
                valueSum += dataset[distances[i].second][col];
                count++;
            }
            imputedRow[col] = valueSum / std::max(1, count);
        }
    }
    return imputedRow;
}

int main() {
    using std::numeric_limits;
    double NaN = numeric_limits<double>::quiet_NaN();

    // Toy dataset: 5 rows x 3 features
    // Target row (index 1) has a missing value in column 1
    std::vector<std::vector<double>> data = {
        {1.00, 2.00, 3.00},
        {1.10, NaN,  3.20},  // <- target row to impute (col 1 is NaN)
        {0.90, 2.10, 2.90},
        {1.20, 1.90, 3.10},
        {1.05, 2.05, 3.05}
    };

    int targetIndex = 1;
    int k = 3;

    // Print original target row
    std::cout << "Original target row: [";
    for (size_t j = 0; j < data[targetIndex].size(); ++j) {
        if (std::isnan(data[targetIndex][j])) std::cout << "NaN";
        else std::cout << std::fixed << std::setprecision(2) << data[targetIndex][j];
        if (j + 1 < data[targetIndex].size()) std::cout << ", ";
    }
    std::cout << "]\n";

    // Impute
    auto imputed = knnImputation(data, k, targetIndex);

    // Print imputed target row
    std::cout << "Imputed target row:  [";
    for (size_t j = 0; j < imputed.size(); ++j) {
        std::cout << std::fixed << std::setprecision(2) << imputed[j];
        if (j + 1 < imputed.size()) std::cout << ", ";
    }
    std::cout << "]\n";

    return 0;
}
