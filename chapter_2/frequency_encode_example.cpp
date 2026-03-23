// frequency_encode_example.cpp
// g++ -std=c++17 -O2 frequency_encode_example.cpp -o freq_enc
// ./freq_enc

#include <vector>
#include <map>
#include <string>
#include <iostream>
#include <iomanip>

// Encode categories based on their frequency
std::vector<int> frequencyEncode(const std::vector<std::string>& categories) {
    std::map<std::string, int> freqMap;
    for (const auto& cat : categories) {
        freqMap[cat]++;
    }
    std::vector<int> encoded;
    encoded.reserve(categories.size());
    for (const auto& cat : categories) {
        encoded.push_back(freqMap[cat]);
    }
    return encoded;
}

int main() {
    std::vector<std::string> cats = {
        "red", "blue", "green", "blue", "red", "yellow", "green", "green"
    };

    auto encoded = frequencyEncode(cats);

    std::cout << "Category -> Frequency (per occurrence)\n";
    for (size_t i = 0; i < cats.size(); ++i) {
        std::cout << std::setw(7) << cats[i] << " -> " << encoded[i] << "\n";
    }
    return 0;
}
