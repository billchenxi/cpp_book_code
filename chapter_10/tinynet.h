#pragma once
#include <torch/torch.h>

// A tiny CNN: Conv-ReLU-Conv-ReLU -> GlobalAvgPool -> Linear
struct TinyNetImpl : torch::nn::Module {
  torch::nn::Conv2d c1{nullptr}, c2{nullptr};
  torch::nn::Linear fc{nullptr};

  TinyNetImpl(int in_ch=3, int num_classes=10)
  : c1(torch::nn::Conv2dOptions(in_ch, 8, 3).padding(1)),
    c2(torch::nn::Conv2dOptions(8, 16, 3).padding(1)),
    fc(16, num_classes) {
    register_module("c1", c1);
    register_module("c2", c2);
    register_module("fc", fc);
  }

  torch::Tensor forward(torch::Tensor x) {
    x = torch::relu(c1->forward(x));
    x = torch::relu(c2->forward(x));
    x = torch::adaptive_avg_pool2d(x, {1,1});   // [N,16,1,1]
    x = x.view({x.size(0), 16});                // [N,16]
    return fc->forward(x);                      // [N,C]
  }
};
TORCH_MODULE(TinyNet);
