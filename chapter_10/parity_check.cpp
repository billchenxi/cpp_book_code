#include <torch/script.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include "tinynet.h"

static float max_abs_diff(const std::vector<float>& a,
                          const std::vector<float>& b) {
  float max_diff = 0.0f;
  for (size_t i = 0; i < a.size(); ++i) {
    max_diff = std::max(max_diff, std::fabs(a[i] - b[i]));
  }
  return max_diff;
}

int main(int argc, char** argv) {
  const std::string model_path =
      argc > 1 ? argv[1] : "chapter_10/tinynet.ts";

  torch::NoGradGuard ng;
  torch::manual_seed(0);

  TinyNet model(3, 10);
  model->eval();

  auto x = torch::randn({2, 3, 224, 224});
  auto ref = model->forward(x).contiguous();
  std::vector<float> ref_values(
      ref.data_ptr<float>(), ref.data_ptr<float>() + ref.numel());

  auto traced = torch::jit::load(model_path);
  traced.eval();
  auto traced_out = traced.forward({x}).toTensor().contiguous();
  std::vector<float> traced_values(
      traced_out.data_ptr<float>(),
      traced_out.data_ptr<float>() + traced_out.numel());

  std::cout << "max|native - ts| = "
            << max_abs_diff(ref_values, traced_values) << "\n";
  return 0;
}
