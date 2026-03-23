// mlpack_encoding_demo_min.cpp
// g++ -std=c++17 mlpack_encoding_demo_min.cpp -o enc \
//   $(pkg-config --cflags --libs mlpack armadillo)
// ./enc

#include <mlpack/core.hpp>
#include <unordered_map>
#include <string>
#include <vector>
#include <iostream>
#include <algorithm>

int main() {
    // Example categories (with repeats)
    std::vector<std::string> categories = {"red","blue","green","blue","red","yellow","green","green"};

    // One-hot (first-seen mapping)
    std::unordered_map<std::string,int> categoryToIndex;
    int idx = 0;
    for (const auto& c : categories)
        if (!categoryToIndex.count(c)) categoryToIndex[c] = idx++;

    arma::mat oneHot(categories.size(), categoryToIndex.size(), arma::fill::zeros);
    for (size_t i = 0; i < categories.size(); ++i)
        oneHot(i, categoryToIndex[categories[i]]) = 1.0;

    // Frequency encoding
    std::unordered_map<std::string,int> freq;
    for (const auto& c : categories) freq[c]++;
    arma::vec freqEnc(categories.size());
    for (size_t i = 0; i < categories.size(); ++i)
        freqEnc(i) = freq[categories[i]];

    // Pretty-print column order by index
    std::vector<std::pair<int,std::string>> byIndex;
    byIndex.reserve(categoryToIndex.size());
    for (const auto& kv : categoryToIndex) byIndex.emplace_back(kv.second, kv.first);
    std::sort(byIndex.begin(), byIndex.end());

    std::cout << "Column order (index: category):\n";
    for (const auto& p : byIndex) std::cout << "  " << p.first << ": " << p.second << "\n";

    std::cout << "\nOne-hot matrix (" << categories.size() << " x "
              << categoryToIndex.size() << "):\n" << oneHot << "\n";

    std::cout << "Frequency per row:\n" << freqEnc.t();
    return 0;
}
