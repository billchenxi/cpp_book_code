#include <Eigen/Dense>

#include <iomanip>
#include <iostream>
#include <vector>

#include "neural_network.h"

int main() {
    using Eigen::VectorXd;

    // Tiny 2-feature toy dataset: XOR.
    std::vector<VectorXd> inputs;
    std::vector<VectorXd> targets;
    inputs.reserve(4);
    targets.reserve(4);

    auto make_vec = [](double a, double b) {
        VectorXd v(2);
        v << a, b;
        return v;
    };

    auto make_target = [](double y) {
        VectorXd v(1);
        v << y;
        return v;
    };

    inputs.push_back(make_vec(0.0, 0.0));
    targets.push_back(make_target(0.0));
    inputs.push_back(make_vec(0.0, 1.0));
    targets.push_back(make_target(1.0));
    inputs.push_back(make_vec(1.0, 0.0));
    targets.push_back(make_target(1.0));
    inputs.push_back(make_vec(1.0, 1.0));
    targets.push_back(make_target(0.0));

    NeuralNetwork model(/*input_size=*/2, /*hidden_size=*/3, /*output_size=*/1, /*lr=*/0.5);

    const int epochs = 5000;
    for (int epoch = 1; epoch <= epochs; ++epoch) {
        double mse = 0.0;
        for (size_t i = 0; i < inputs.size(); ++i) {
            model.train(inputs[i], targets[i]);
            const VectorXd pred = model.feedforward(inputs[i]);
            mse += 0.5 * (pred - targets[i]).squaredNorm();
        }

        if (epoch % 500 == 0) {
            std::cout << "Epoch " << std::setw(4) << epoch
                      << " | MSE=" << std::fixed << std::setprecision(6)
                      << (mse / inputs.size()) << '\n';
        }
    }

    std::cout << "\nPredictions after training:\n";
    for (size_t i = 0; i < inputs.size(); ++i) {
        const VectorXd pred = model.feedforward(inputs[i]);
        const int label = pred[0] >= 0.5 ? 1 : 0;
        std::cout << "[" << inputs[i][0] << ", " << inputs[i][1] << "]"
                  << " -> " << pred[0]
                  << " (class " << label << ", target " << targets[i][0] << ")\n";
    }

    return 0;
}

/*
Build from the repo root:
g++ -std=c++17 -O2 chapter_4/mlp_eigen_demo.cpp chapter_4/neural_network.cpp \
  -o chapter_4/mlp_eigen_demo \
  -I"$(brew --prefix eigen)/include/eigen3"

Run:
./chapter_4/mlp_eigen_demo
*/
