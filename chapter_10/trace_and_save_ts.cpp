#include <torch/script.h>

#include <iostream>
#include <string>

#include "tinynet.h"

int main(int argc, char** argv) {
  const std::string out_path =
      argc > 1 ? argv[1] : "chapter_10/tinynet.ts";

  torch::NoGradGuard ng;
  torch::manual_seed(0);

  TinyNet native_model(/*in_ch=*/3, /*num_classes=*/10);
  native_model->eval();

  torch::jit::Module traced("TinyNet");
  traced.register_parameter("c1_weight", native_model->c1->weight, false);
  traced.register_parameter("c1_bias", native_model->c1->bias, false);
  traced.register_parameter("c2_weight", native_model->c2->weight, false);
  traced.register_parameter("c2_bias", native_model->c2->bias, false);
  traced.register_parameter("fc_weight", native_model->fc->weight, false);
  traced.register_parameter("fc_bias", native_model->fc->bias, false);
  traced.define(R"JIT(
    def forward(self, x):
      x = torch.conv2d(x, self.c1_weight, self.c1_bias, [1, 1], [1, 1], [1, 1], 1)
      x = torch.relu(x)
      x = torch.conv2d(x, self.c2_weight, self.c2_bias, [1, 1], [1, 1], [1, 1], 1)
      x = torch.relu(x)
      x = torch.adaptive_avg_pool2d(x, [1, 1])
      x = torch.flatten(x, 1)
      x = torch.matmul(x, self.fc_weight.t()) + self.fc_bias
      return x
  )JIT");

  traced.save(out_path);

  std::cout << "Saved TorchScript artifact: " << out_path << "\n";
  return 0;
}
