#if __has_include(<onnxruntime/onnxruntime_cxx_api.h>)
#include <onnxruntime/onnxruntime_cxx_api.h>
#else
#include <onnxruntime_cxx_api.h>
#endif

#include <array>
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char** argv) {
  const std::string model_path =
      argc > 1 ? argv[1] : "chapter_10/model.int8.onnx";

  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "int8");
  Ort::SessionOptions options;
  options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
  options.SetIntraOpNumThreads(8);

  Ort::Session session(env, model_path.c_str(), options);

  std::array<int64_t, 4> shape{16, 3, 224, 224};
  std::vector<float> data(16 * 3 * 224 * 224, 0.5f);
  auto memory =
      Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  auto input = Ort::Value::CreateTensor<float>(
      memory, data.data(), data.size(), shape.data(), shape.size());

  Ort::AllocatorWithDefaultOptions allocator;
  auto input_name = session.GetInputNameAllocated(0, allocator);
  auto output_name = session.GetOutputNameAllocated(0, allocator);
  const char* input_name_c = input_name.get();
  const char* output_name_c = output_name.get();

  for (int i = 0; i < 5; ++i) {
    (void)session.Run(Ort::RunOptions{},
                      &input_name_c,
                      &input,
                      1,
                      &output_name_c,
                      1);
  }

  auto outputs = session.Run(Ort::RunOptions{},
                             &input_name_c,
                             &input,
                             1,
                             &output_name_c,
                             1);
  auto& y = outputs.front();
  std::cout << "ran INT8 graph; first="
            << y.GetTensorMutableData<float>()[0] << "\n";
  return 0;
}
