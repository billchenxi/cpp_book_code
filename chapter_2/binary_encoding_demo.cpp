// binary_encoding_demo.cpp
// g++ -std=c++17 -O2 binary_encoding_demo.cpp -o bin_demo
// ./bin_demo

#include <vector>
#include <string>
#include <bitset>
#include <iostream>

std::vector<std::string> binaryEncode(const std::vector<int>& categories) {
    std::vector<std::string> encoded;
    for (const auto& cat : categories) {
        encoded.push_back(std::bitset<8>(cat).to_string()); // 8-bit demo
    }
    return encoded;
}

int main() {
    std::vector<int> cats = {0, 1, 2, 5, 15, 128, 255};

    auto encoded = binaryEncode(cats);

    for (size_t i = 0; i < cats.size(); ++i) {
        std::cout << cats[i] << " -> " << encoded[i] << "\n";
    }
    return 0;
}
