#include <torch/script.h>

#include <iostream>
#include <string>

#include "device_utils.h"

int main(int argc, char** argv) {
  const std::string model_path =
      argc > 1 ? argv[1] : "chapter_10/tinynet_slim.ts";

  torch::NoGradGuard ng;

  auto module = torch::jit::load(model_path);
  module.eval();

  auto x = torch::randn({8, 3, 224, 224});
  if (chapter10::cuda_available()) {
    module.to(torch::kCUDA);
    x = x.to(torch::kCUDA).to(torch::MemoryFormat::ChannelsLast);
  }

  for (int i = 0; i < 5; ++i) {
    (void)module.forward({x});
  }

  auto y = module.forward({x}).toTensor();
  std::cout << "slim output: " << y.sizes() << "\n";
  return 0;
}
