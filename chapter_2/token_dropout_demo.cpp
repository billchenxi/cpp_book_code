// chapter_2/token_dropout_demo.cpp
// g++ -std=c++17 -O2 chapter_2/token_dropout_demo.cpp -o chapter_2/token_dropout_demo
// ./chapter_2/token_dropout_demo

#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

std::vector<std::string> token_dropout(const std::vector<std::string>& toks,
                                       float p) {
  std::mt19937 rng(42);
  std::bernoulli_distribution keep(1.0f - p);
  std::vector<std::string> out;
  out.reserve(toks.size());
  for (const auto& tok : toks) {
    if (keep(rng)) {
      out.push_back(tok);
    }
  }
  return out;
}

int main() {
  const std::vector<std::string> toks = {
      "shipping", "was", "fast", "but", "the", "packaging", "was", "damaged"};
  const float p = 0.35f;

  const auto dropped = token_dropout(toks, p);

  std::cout << "input  : ";
  for (const auto& tok : toks) {
    std::cout << tok << ' ';
  }
  std::cout << "\n";

  std::cout << "output : ";
  for (const auto& tok : dropped) {
    std::cout << tok << ' ';
  }
  std::cout << "\n";

  std::cout << "kept " << dropped.size() << " of " << toks.size()
            << " tokens at p=" << std::fixed << std::setprecision(2) << p
            << "\n";
  return 0;
}
