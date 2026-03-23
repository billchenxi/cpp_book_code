#include <torch/script.h>

#include <iostream>
#include <string>

#include "device_utils.h"

int main(int argc, char** argv) {
  const std::string model_path =
      argc > 1 ? argv[1] : "chapter_10/tinynet.ts";

  torch::NoGradGuard ng;

  torch::jit::script::Module module = torch::jit::load(model_path);
  module.eval();

  auto x = torch::randn({1, 3, 224, 224}, torch::kFloat);
  if (chapter10::cuda_available()) {
    module.to(torch::kCUDA);
    x = x.to(torch::kCUDA).to(torch::MemoryFormat::ChannelsLast);
  }

  for (int i = 0; i < 5; ++i) {
    (void)module.forward({x});
  }

  auto y = module.forward({x}).toTensor();
  std::cout << "TorchScript output shape: " << y.sizes() << "\n";
  return 0;
}
