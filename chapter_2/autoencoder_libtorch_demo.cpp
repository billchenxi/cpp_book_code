// autoencoder_libtorch_demo.cpp
// g++ -std=c++17 autoencoder_libtorch_demo.cpp -o ae_demo \
//   -I libtorch/include -I libtorch/include/torch/csrc/api/include \
//   -L libtorch/lib -Wl,-rpath,'libtorch/lib' \ 
//   -ltorch -ltorch_cpu -lc10
// ./ae_demo

#include <torch/torch.h>
#include <iostream>
#include <iomanip>

// ---- Your autoencoder (with minor Adam tweak in autoencode) ----
struct Autoencoder : torch::nn::Module {
    torch::nn::Linear encoder{nullptr}, decoder{nullptr};
    Autoencoder(int inputDim, int hiddenDim) {
        encoder = register_module("encoder", torch::nn::Linear(inputDim, hiddenDim));
        decoder = register_module("decoder", torch::nn::Linear(hiddenDim, inputDim));
    }
    torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(encoder->forward(x));
        return decoder->forward(x);
    }
};

torch::Tensor autoencode(torch::Tensor data, int hiddenDim) {
    const int inputDim = data.size(1);
    Autoencoder model(inputDim, hiddenDim);
    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(0.01));

    model.train();
    for (int epoch = 0; epoch < 100; ++epoch) {
        optimizer.zero_grad();
        auto output = model.forward(data);
        auto loss = torch::mse_loss(output, data);
        loss.backward();
        optimizer.step();
    }
    model.eval();
    torch::NoGradGuard ng;
    return model.encoder->forward(data); // latent codes
}

int main() {
    torch::manual_seed(42);

    // Make a simple low-rank dataset: X = Z * W + small noise
    const int N = 200, Din = 6, Dz = 2;
    auto Z  = torch::randn({N, Dz});
    auto W  = torch::randn({Dz, Din});
    auto X  = Z.matmul(W) + 0.05 * torch::randn({N, Din});
    X = X.to(torch::kFloat32);

    auto Z_hat = autoencode(X, /*hiddenDim=*/2);

    std::cout << "Input  shape: " << X.sizes() << "\n";
    std::cout << "Latent shape: " << Z_hat.sizes() << "\n";
    std::cout << "First row of latent: [" 
              << Z_hat[0][0].item<float>() << ", " 
              << Z_hat[0][1].item<float>() << "]\n";
    return 0;
}
