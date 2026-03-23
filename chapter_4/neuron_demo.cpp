/*
Build from the repo root:
g++ -std=c++17 chapter_4/neuron_demo.cpp -o chapter_4/neuron_demo \
  -I ./libtorch/include \
  -I ./libtorch/include/torch/csrc/api/include \
  -L ./libtorch/lib \
  -Wl,-rpath,./libtorch/lib \
  -ltorch -ltorch_cpu -lc10

Run:
DYLD_LIBRARY_PATH=./libtorch/lib ./chapter_4/neuron_demo
*/

#include <torch/torch.h>
#include <iostream>

int main()
{
	torch::manual_seed(42);

	// Pick device (CUDA if available)
	torch::Device device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
	std::cout << "Using device: " << (device.is_cuda() ? "CUDA" : "CPU") << "\n";

	const int64_t batch = 4, in_f = 5, out_f = 3;

	auto x = torch::randn({batch, in_f}).to(device);

	torch::nn::Linear fc(in_f, out_f);
	fc->to(device);

	torch::nn::init::kaiming_uniform_(fc->weight, /*a=*/std::sqrt(5.0));
	torch::nn::init::zeros_(fc->bias);

	auto z = fc->forward(x); // [batch, out_f]
	auto y = torch::relu(z); // activation

	std::cout << "Input shape : " << x.sizes() << "\n";
	std::cout << "Logits shape: " << z.sizes() << "\n";
	std::cout << "Output shape: " << y.sizes() << "\n";
	std::cout << "Output sample:\n"
			  << y.slice(/*dim=*/0, 0, 2) << "\n";
	return 0;
}
