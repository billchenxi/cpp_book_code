// autoencoder_demo.cpp
// g++ -std=c++17 autoencoder_demo.cpp -o autoencoder_demo \
//   -I libtorch/include \
//   -I libtorch/include/torch/csrc/api/include \
//   -L libtorch/lib \
//   -Wl,-rpath,'$ORIGIN/libtorch/lib' \
//   -ltorch -ltorch_cpu -lc10

// DYLD_LIBRARY_PATH=libtorch/lib ./autoencoder_demo


#include <torch/torch.h>
#include <iostream>
#include <iomanip>

// --- Your code (with a minor tweak to Adam options for compilation) ---
struct Autoencoder : torch::nn::Module {
    torch::nn::Linear encoder{nullptr}, decoder{nullptr};
    Autoencoder(int inputDim, int hiddenDim) {
        encoder = register_module("encoder", torch::nn::Linear(inputDim, hiddenDim));
        decoder = register_module("decoder", torch::nn::Linear(hiddenDim, inputDim));
    }
    torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(encoder->forward(x));
        x = decoder->forward(x);
        return x;
    }
};

// Train the autoencoder and return the encoded representation
torch::Tensor reduceDimensionsWithAutoencoder(torch::Tensor data, int hiddenDim) {
    const int inputDim = data.size(1);
    Autoencoder model(inputDim, hiddenDim);
    model.to(torch::kFloat32);

    torch::optim::Adam optimizer(
        model.parameters(),
        torch::optim::AdamOptions(0.01)
    );

    model.train();
    for (int epoch = 1; epoch <= 150; ++epoch) {
        optimizer.zero_grad();
        auto output = model.forward(data);
        auto loss = torch::mse_loss(output, data);
        loss.backward();
        optimizer.step();

        if (epoch % 25 == 0) {
            std::cout << "Epoch " << std::setw(3) << epoch
                      << " | MSE: " << std::fixed << std::setprecision(6)
                      << loss.item<float>() << "\n";
        }
    }

    model.eval();
    torch::NoGradGuard ng;
    return model.encoder->forward(data); // latent codes (pre-activation)
}

int main() {
    torch::manual_seed(42);

    // --- Build a synthetic low-rank dataset: X = Z * W + noise ---
    const int N = 240;   // samples
    const int Din = 6;   // input dim
    const int Dz = 2;    // true latent dim
    auto Z  = torch::randn({N, Dz});            // latent factors
    auto W  = torch::randn({Dz, Din});          // projection to input space
    auto X  = Z.matmul(W) + 0.05 * torch::randn({N, Din}); // add small noise
    X = X.to(torch::kFloat32);

    // --- Train autoencoder and reduce to 2D ---
    auto Z_hat = reduceDimensionsWithAutoencoder(X, /*hiddenDim=*/2);

    std::cout << "\nShapes -> X: " << X.sizes() << ", Z_hat: " << Z_hat.sizes() << "\n";
    std::cout << "First 5 reduced vectors:\n";
    for (int i = 0; i < 5; ++i) {
        auto row = Z_hat[i];
        std::cout << "  [" << row[0].item<float>() << ", " << row[1].item<float>() << "]\n";
    }
    return 0;
}
