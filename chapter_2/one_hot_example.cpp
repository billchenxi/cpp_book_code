// one_hot_example.cpp
// g++ -std=c++17 -O2 one_hot_example.cpp -o one_hot
// ./one_hot

#include <vector>
#include <map>
#include <string>
#include <iostream>
#include <iomanip>
#include <algorithm>

// Convert categorical variables into binary vectors
std::vector<std::vector<int>> oneHotEncode(const std::vector<std::string>& categories) {
    std::map<std::string, int> categoryMap;
    int index = 0;
    for (const auto& cat : categories) {
        if (categoryMap.find(cat) == categoryMap.end()) {
            categoryMap[cat] = index++;
        }
    }

    std::vector<std::vector<int>> encoded(categories.size(), std::vector<int>(categoryMap.size(), 0));
    for (size_t i = 0; i < categories.size(); ++i) {
        encoded[i][categoryMap[categories[i]]] = 1;
    }
    return encoded;
}

int main() {
    // Sample categories (with repeats)
    std::vector<std::string> cats = {"red", "blue", "green", "blue", "red", "yellow", "green"};

    // Encode
    auto encoded = oneHotEncode(cats);

    // Recreate the (category -> column index) mapping using the same first-seen rule
    std::map<std::string,int> mapFirstSeen;
    int idx = 0;
    for (const auto& c : cats) if (mapFirstSeen.find(c) == mapFirstSeen.end()) mapFirstSeen[c] = idx++;

    // Print column order by index
    std::vector<std::pair<int,std::string>> byIndex;
    for (const auto& kv : mapFirstSeen) byIndex.push_back({kv.second, kv.first});
    std::sort(byIndex.begin(), byIndex.end()); // sort by column index

    std::cout << "Column order (index: category):\n";
    for (const auto& p : byIndex) std::cout << "  " << p.first << ": " << p.second << "\n";

    // Print encoded matrix
    std::cout << "\nEncoded matrix (" << cats.size() << " x " << encoded[0].size() << "):\n";
    for (size_t i = 0; i < encoded.size(); ++i) {
        std::cout << std::setw(7) << cats[i] << " : [";
        for (size_t j = 0; j < encoded[i].size(); ++j) {
            std::cout << encoded[i][j] << (j + 1 < encoded[i].size() ? ", " : "");
        }
        std::cout << "]\n";
    }
    return 0;
}
