#include <torch/script.h>

#include <iostream>
#include <string>

#ifdef USE_CUDA
#include <ATen/cuda/CUDAEvent.h>
#include <ATen/cuda/CUDAGraph.h>
#endif

int main(int argc, char** argv) {
#ifndef USE_CUDA
  std::cerr << "This demo requires a CUDA-enabled LibTorch build.\n";
  return 1;
#else
  const std::string model_path =
      argc > 1 ? argv[1] : "chapter_10/tinynet.ts";

  if (!torch::cuda::is_available()) {
    std::cerr << "CUDA runtime not available.\n";
    return 1;
  }

  torch::NoGradGuard ng;

  auto module = torch::jit::load(model_path);
  module.eval();
  module.to(torch::kCUDA);

  auto static_in = torch::randn({8, 3, 224, 224}, torch::kCUDA);
  auto static_out = torch::empty({8, 10}, static_in.options());

  for (int i = 0; i < 5; ++i) {
    (void)module.forward({static_in});
  }

  at::cuda::CUDAGraph graph;
  {
    at::cuda::CUDAStreamCaptureModeGuard guard(
        at::cuda::CaptureMode::Relaxed);
    graph.capture_begin();
    auto y = module.forward({static_in}).toTensor();
    static_out.copy_(y);
    graph.capture_end();
  }

  for (int i = 0; i < 100; ++i) {
    graph.replay();
  }

  std::cout << "Captured/replayed CUDA graph.\n";
  return 0;
#endif
}
