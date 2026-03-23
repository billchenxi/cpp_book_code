#if __has_include(<onnxruntime/onnxruntime_cxx_api.h>)
#include <onnxruntime/onnxruntime_cxx_api.h>
#else
#include <onnxruntime_cxx_api.h>
#endif

#include <iostream>

int main() {
  Ort::SessionOptions options;
  options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
  options.SetIntraOpNumThreads(8);

  std::cout << "Configured ONNX Runtime session options:\n";
  std::cout << "  graph_optimization = ORT_ENABLE_ALL\n";
  std::cout << "  intra_op_threads = 8\n";
  std::cout << "  execution_mode = sequential (default; switch to ORT_PARALLEL if needed)\n";
  std::cout << "  provider = CPU by default; append CUDA/TensorRT explicitly when available\n";
  return 0;
}
