#include <torch/script.h>

#include <chrono>
#include <iostream>
#include <string>

#include "device_utils.h"

using Clock = std::chrono::steady_clock;

int main(int argc, char** argv) {
  const std::string model_path =
      argc > 1 ? argv[1] : "chapter_10/tinynet.ts";

  torch::NoGradGuard ng;
  auto module = torch::jit::load(model_path);
  module.eval();

  if (!chapter10::cuda_available()) {
    std::cerr << "GPU required for FP16 speedups.\n";
    return 1;
  }

  module.to(torch::kCUDA);
  auto x = torch::randn({8, 3, 224, 224}).to(torch::kCUDA).to(torch::kHalf);
  x = x.to(torch::MemoryFormat::ChannelsLast);

  for (int i = 0; i < 10; ++i) {
    (void)module.forward({x});
  }

  auto t0 = Clock::now();
  auto y = module.forward({x}).toTensor();
  auto elapsed_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - t0)
          .count();
  auto y32 = y.to(torch::kFloat);

  std::cout << "batch=8 elapsed(ms): " << elapsed_ms
            << " first=" << y32[0][0].item<float>() << "\n";
  return 0;
}
