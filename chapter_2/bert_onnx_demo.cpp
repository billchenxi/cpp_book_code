// bert_onnx_demo.cpp
// Runnable ONNX Runtime text-model demo: loads a transformer-style .onnx model,
// feeds token ids + mask + optional token_type ids, and prints the first few
// floats from the first output tensor.
//
// Usage:
//   c++ -std=c++17 -O2 chapter_2/bert_onnx_demo.cpp -o chapter_2/bert_demo \
//     -I"$(brew --prefix onnxruntime)/include" \
//     -L"$(brew --prefix onnxruntime)/lib" \
//     -lonnxruntime \
//     -Wl,-rpath,"$(brew --prefix onnxruntime)/lib"
//
//   DYLD_LIBRARY_PATH="$(brew --prefix onnxruntime)/lib" \
//     ./chapter_2/bert_demo bert_onnx/onnx/model.onnx

#include <onnxruntime/onnxruntime_cxx_api.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

static std::vector<int64_t> parse_ids(const std::string& s) {
  std::vector<int64_t> ids;
  std::istringstream ss(s);
  long long v = 0;
  while (ss >> v) {
    ids.push_back(static_cast<int64_t>(v));
  }
  return ids;
}

static void print_shape(std::ostream& os, const std::vector<int64_t>& shape) {
  os << "[";
  for (size_t i = 0; i < shape.size(); ++i) {
    os << shape[i] << (i + 1 < shape.size() ? ", " : "");
  }
  os << "]";
}

template <typename T>
static void fill_tensor_data(Ort::Value& tensor, const std::vector<int64_t>& src) {
  T* buf = tensor.GetTensorMutableData<T>();
  for (size_t i = 0; i < src.size(); ++i) {
    buf[i] = static_cast<T>(src[i]);
  }
}

static Ort::Value make_tensor_from_i64(
    OrtAllocator* allocator, ONNXTensorElementDataType etype,
    const std::vector<int64_t>& vals, const std::array<int64_t, 2>& shape) {
  Ort::Value tensor =
      Ort::Value::CreateTensor(allocator, shape.data(), shape.size(), etype);
  switch (etype) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
      fill_tensor_data<int64_t>(tensor, vals);
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
      fill_tensor_data<int32_t>(tensor, vals);
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
      fill_tensor_data<int16_t>(tensor, vals);
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
      fill_tensor_data<int8_t>(tensor, vals);
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
      fill_tensor_data<uint32_t>(tensor, vals);
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
      fill_tensor_data<uint16_t>(tensor, vals);
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
      fill_tensor_data<uint8_t>(tensor, vals);
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL: {
      bool* buf = tensor.GetTensorMutableData<bool>();
      for (size_t i = 0; i < vals.size(); ++i) {
        buf[i] = vals[i] != 0;
      }
      break;
    }
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
      fill_tensor_data<float>(tensor, vals);
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
      fill_tensor_data<double>(tensor, vals);
      break;
    default:
      throw std::runtime_error("Unsupported transformer input dtype");
  }
  return tensor;
}

int main(int argc, char** argv) {
  try {
    if (argc < 2) {
      std::cerr << "Usage: " << argv[0]
                << " /path/to/model.onnx [optional: space-separated token ids]\n";
      return 1;
    }

    const std::string model_path = argv[1];
    const std::vector<int64_t> input_ids =
        (argc >= 3) ? parse_ids(argv[2])
                    : std::vector<int64_t>{101, 2023, 2003, 1037, 3231, 102};
    const std::vector<int64_t> attention_mask(input_ids.size(), 1);
    const std::vector<int64_t> token_type_ids(input_ids.size(), 0);
    const std::array<int64_t, 2> shape{
        1, static_cast<int64_t>(input_ids.size())};

    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "bert_demo");
    Ort::SessionOptions so;
    so.SetIntraOpNumThreads(1);
    so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL);
    Ort::Session session(env, model_path.c_str(), so);
    Ort::AllocatorWithDefaultOptions allocator;

    const size_t n_inputs = session.GetInputCount();
    std::vector<std::string> input_names;
    input_names.reserve(n_inputs);

    std::cerr << "Model inputs (" << n_inputs << "):\n";
    for (size_t i = 0; i < n_inputs; ++i) {
      auto name_alloc = session.GetInputNameAllocated(i, allocator);
      input_names.emplace_back(name_alloc.get());
      auto type_info = session.GetInputTypeInfo(i);
      auto info = type_info.GetTensorTypeAndShapeInfo();
      std::cerr << "  - " << input_names.back()
                << " rank=" << info.GetShape().size()
                << " elem_type=" << info.GetElementType() << "\n";
    }

    std::vector<std::string> feed_names_keep;
    std::vector<const char*> feed_names;
    std::vector<Ort::Value> feeds;
    feed_names_keep.reserve(n_inputs);
    feeds.reserve(n_inputs);

    auto push_named = [&](const std::string& name,
                          const std::vector<int64_t>& values,
                          ONNXTensorElementDataType etype) {
      feed_names_keep.push_back(name);
      feeds.emplace_back(make_tensor_from_i64(allocator, etype, values, shape));
    };

    for (size_t i = 0; i < n_inputs; ++i) {
      const auto& name = input_names[i];
      auto type_info = session.GetInputTypeInfo(i);
      auto info = type_info.GetTensorTypeAndShapeInfo();
      const auto etype = info.GetElementType();

      if (name.find("input_ids") != std::string::npos) {
        push_named(name, input_ids, etype);
      } else if (name.find("attention") != std::string::npos) {
        push_named(name, attention_mask, etype);
      } else if (name.find("token_type") != std::string::npos ||
                 name.find("segment") != std::string::npos) {
        push_named(name, token_type_ids, etype);
      }
    }

    if (feed_names_keep.empty() && !input_names.empty()) {
      auto type_info0 = session.GetInputTypeInfo(0);
      auto info0 = type_info0.GetTensorTypeAndShapeInfo();
      push_named(input_names[0], input_ids, info0.GetElementType());
      if (input_names.size() > 1) {
        auto type_info1 = session.GetInputTypeInfo(1);
        auto info1 = type_info1.GetTensorTypeAndShapeInfo();
        push_named(input_names[1], attention_mask, info1.GetElementType());
      }
      if (input_names.size() > 2) {
        auto type_info2 = session.GetInputTypeInfo(2);
        auto info2 = type_info2.GetTensorTypeAndShapeInfo();
        push_named(input_names[2], token_type_ids, info2.GetElementType());
      }
    }

    feed_names.reserve(feed_names_keep.size());
    for (auto& name : feed_names_keep) {
      feed_names.push_back(name.c_str());
    }

    auto output_name_alloc = session.GetOutputNameAllocated(0, allocator);
    std::string output_name = output_name_alloc.get();
    const char* output_name_c = output_name.c_str();

    auto outputs = session.Run(Ort::RunOptions{}, feed_names.data(),
                               feeds.data(), feeds.size(), &output_name_c, 1);

    if (outputs.empty() || !outputs[0].IsTensor()) {
      throw std::runtime_error("Model returned no tensor outputs");
    }

    auto out_info = outputs[0].GetTensorTypeAndShapeInfo();
    std::cout << "\nFirst output shape=";
    print_shape(std::cout, out_info.GetShape());
    std::cout << " elem_type=" << out_info.GetElementType() << "\n";

    if (out_info.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
      const float* p = outputs[0].GetTensorData<float>();
      const size_t count = out_info.GetElementCount();
      const size_t to_print = std::min<size_t>(8, count);
      std::cout << "values: ";
      for (size_t i = 0; i < to_print; ++i) {
        std::cout << p[i] << (i + 1 < to_print ? ", " : "\n");
      }
    }

    return 0;
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 2;
  }
}
