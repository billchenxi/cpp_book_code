// chapter_2/embedding_encode_demo.cpp
// g++ -std=c++17 -O2 chapter_2/embedding_encode_demo.cpp -o chapter_2/emb_demo
// ./chapter_2/emb_demo

#include <vector>
#include <map>
#include <string>
#include <cmath>
#include <iostream>
#include <iomanip>

// Dummy embedding logic (provided)
std::vector<std::vector<double>> embeddingEncode(const std::vector<std::string>& categories, int embeddingSize) {
    std::map<std::string, int> categoryMap;
    int index = 0;
    for (const auto& cat : categories) {
        if (categoryMap.find(cat) == categoryMap.end()) {
            categoryMap[cat] = index++;
        }
    }

    std::vector<std::vector<double>> embeddings(categoryMap.size(), std::vector<double>(embeddingSize, 0));
    for (size_t i = 0; i < embeddings.size(); ++i) {
        for (size_t j = 0; j < (size_t)embeddingSize; ++j) {
            embeddings[i][j] = std::sin(i + j);  // Example: sinusoidal encoding
        }
    }

    std::vector<std::vector<double>> encoded;
    for (const auto& cat : categories) {
        encoded.push_back(embeddings[categoryMap[cat]]);
    }
    return encoded;
}

int main() {
    std::vector<std::string> cats = {"cat", "dog", "dog", "mouse", "cat"};
    int embDim = 4;

    auto enc = embeddingEncode(cats, embDim);

    for (size_t i = 0; i < cats.size(); ++i) {
        std::cout << std::setw(6) << cats[i] << " -> [";
        for (int j = 0; j < embDim; ++j) {
            std::cout << std::fixed << std::setprecision(3) << enc[i][j]
                      << (j + 1 < embDim ? ", " : "");
        }
        std::cout << "]\n";
    }
    return 0;
}
