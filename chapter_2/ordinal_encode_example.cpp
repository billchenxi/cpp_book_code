// ordinal_encode_example.cpp
// g++ -std=c++17 -O2 ordinal_encode_example.cpp -o ordinal_enc
// ./ordinal_enc

#include <vector>
#include <map>
#include <string>
#include <iostream>
#include <iomanip>
#include <algorithm>

// Assign an ordinal value to each category
std::vector<int> ordinalEncode(const std::vector<std::string>& categories) {
    std::map<std::string, int> categoryMap;
    int index = 0;
    for (const auto& cat : categories) {
        if (categoryMap.find(cat) == categoryMap.end()) {
            categoryMap[cat] = index++;
        }
    }
    std::vector<int> encoded;
    encoded.reserve(categories.size());
    for (const auto& cat : categories) {
        encoded.push_back(categoryMap[cat]);
    }
    return encoded;
}

int main() {
    // Example data (note: order of first appearance defines the codes)
    std::vector<std::string> sizes = {"XS","S","M","L","XL","S","M","XS","XL"};

    // Encode
    auto codes = ordinalEncode(sizes);

    // Reconstruct the same first-seen mapping for display
    std::map<std::string,int> firstSeen;
    int idx = 0;
    for (const auto& s : sizes) if (firstSeen.find(s) == firstSeen.end()) firstSeen[s] = idx++;

    // Invert mapping to print by index
    std::vector<std::pair<int,std::string>> byIndex;
    for (const auto& kv : firstSeen) byIndex.push_back({kv.second, kv.first});
    std::sort(byIndex.begin(), byIndex.end(),
              [](auto& a, auto& b){ return a.first < b.first; });

    std::cout << "Ordinal mapping (index: category):\n";
    for (const auto& p : byIndex) std::cout << "  " << p.first << ": " << p.second << "\n";

    std::cout << "\nEncoded sequence:\n[";
    for (size_t i = 0; i < codes.size(); ++i) {
        std::cout << codes[i] << (i + 1 < codes.size() ? ", " : "");
    }
    std::cout << "]\n";

    return 0;
}
