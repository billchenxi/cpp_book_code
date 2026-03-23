#include <torch/cuda.h>
#include <torch/torch.h>

#include <opencv2/opencv.hpp>

#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct SmallCNN : torch::nn::Module {
  explicit SmallCNN(int num_classes = 2)
      : conv1(torch::nn::Conv2dOptions(3, 16, 3).padding(1)),
        conv2(torch::nn::Conv2dOptions(16, 32, 3).padding(1)),
        fc(32, num_classes) {
    register_module("conv1", conv1);
    register_module("conv2", conv2);
    register_module("fc", fc);
  }

  torch::Tensor forward(torch::Tensor x) {
    x = torch::relu(conv1->forward(x));
    x = torch::relu(conv2->forward(x));
    last_conv_activation = x;
    last_conv_activation.retain_grad();
    x = torch::adaptive_avg_pool2d(x, {1, 1});
    x = x.view({x.size(0), -1});
    return fc->forward(x);
  }

  torch::Tensor last_conv_activation;
  torch::nn::Conv2d conv1{nullptr};
  torch::nn::Conv2d conv2{nullptr};
  torch::nn::Linear fc{nullptr};
};

torch::Tensor load_image_tensor(const std::string& path, int height = 224,
                                int width = 224) {
  cv::Mat bgr = cv::imread(path, cv::IMREAD_COLOR);
  if (bgr.empty()) {
    throw std::runtime_error("Failed to read image: " + path);
  }

  cv::Mat rgb;
  cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
  cv::Mat resized;
  cv::resize(rgb, resized, cv::Size(width, height));
  resized.convertTo(resized, CV_32F, 1.0 / 255.0);

  auto t = torch::from_blob(resized.data, {height, width, 3}, torch::kFloat32)
               .clone();
  return t.permute({2, 0, 1}).unsqueeze(0);
}

cv::Mat overlay_heatmap(const torch::Tensor& heat01, const cv::Mat& base_bgr,
                        double alpha = 0.5) {
  torch::Tensor h =
      heat01.clamp(0, 1).mul(255).to(torch::kU8).contiguous().cpu();
  cv::Mat heat(static_cast<int>(h.size(0)), static_cast<int>(h.size(1)), CV_8U,
               h.data_ptr<uint8_t>());
  cv::Mat heat_color;
  cv::applyColorMap(heat, heat_color, cv::COLORMAP_JET);

  cv::Mat base_u8;
  base_bgr.convertTo(base_u8, CV_8UC3, 255.0);

  cv::Mat overlay;
  cv::addWeighted(heat_color, alpha, base_u8, 1.0 - alpha, 0.0, overlay);
  return overlay;
}

torch::Tensor gradcam_map(SmallCNN& model, torch::Tensor input,
                          int target_class, torch::Device device) {
  model.to(device);
  model.eval();
  input = input.to(device);

  model.zero_grad();
  auto logits = model.forward(input);
  const int64_t num_classes = logits.size(1);
  if (target_class < 0 || target_class >= num_classes) {
    throw std::out_of_range("target_class out of range for model logits");
  }

  auto score = logits.index({0, target_class});
  score.backward();

  auto activations = model.last_conv_activation.detach();
  auto grads = model.last_conv_activation.grad().detach();
  auto weights = grads.mean(std::vector<int64_t>{2, 3}, false);
  auto cam =
      torch::relu((activations * weights.unsqueeze(-1).unsqueeze(-1)).sum(1));
  cam = cam.squeeze(0);

  const double cam_min = cam.min().item<double>();
  const double cam_max = cam.max().item<double>();
  cam = (cam - cam_min) / (cam_max - cam_min + 1e-8);

  const int64_t H = input.size(2);
  const int64_t W = input.size(3);
  cam = torch::nn::functional::interpolate(
            cam.unsqueeze(0).unsqueeze(0),
            torch::nn::functional::InterpolateFuncOptions()
                .size(std::vector<int64_t>{H, W})
                .mode(torch::kBilinear)
                .align_corners(false))
            .squeeze();
  return cam.cpu();
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc < 3) {
      std::cerr << "Usage: " << argv[0] << " <image_path> <target_class>\n";
      return 1;
    }

    const std::string image_path = argv[1];
    const int target_class = std::stoi(argv[2]);
    torch::Device device(torch::cuda::is_available() ? torch::kCUDA
                                                     : torch::kCPU);

    cv::Mat bgr = cv::imread(image_path, cv::IMREAD_COLOR);
    if (bgr.empty()) {
      throw std::runtime_error("Failed to read: " + image_path);
    }
    cv::Mat bgr_resized;
    cv::resize(bgr, bgr_resized, cv::Size(224, 224));
    cv::Mat bgr_float;
    bgr_resized.convertTo(bgr_float, CV_32F, 1.0 / 255.0);

    auto input = load_image_tensor(image_path, 224, 224);

    SmallCNN net(5);
    net.to(device);

    auto heat = gradcam_map(net, input, target_class, device);
    cv::Mat overlay = overlay_heatmap(heat, bgr_float, 0.5);

    const std::string output_path = "chapter_13/gradcam_overlay.png";
    cv::imwrite(output_path, overlay);
    std::cout << "Saved: " << output_path << "\n";
  } catch (const c10::Error& e) {
    std::cerr << "LibTorch error: " << e.what() << "\n";
    return 2;
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 2;
  }

  return 0;
}
