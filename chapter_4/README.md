# Chapter 4 Code

This folder groups the code used in chapter 4 of `B22398_04_edited 03112026.docx`.

Files in this chapter:

- `linear_regression_gd.cpp`: linear regression with per-sample gradient descent on synthetic 1-D data.
- `logistic_regression_gd.cpp`: logistic regression on two Gaussian blobs with binary cross-entropy.
- `neuron_demo.cpp`: one LibTorch affine layer plus ReLU.
- `neural_network.h`: from-scratch 2-layer MLP declaration using Eigen.
- `neural_network.cpp`: from-scratch forward pass and backpropagation implementation.
- `mlp_eigen_demo.cpp`: runnable training demo for the Eigen MLP on a tiny 2-feature toy dataset.

## Start

All commands below assume you run them from the repository root.

Quick checks:

```bash
clang++ --version
ls ./libtorch/include/torch/torch.h
brew --prefix eigen
```

## Linear regression

```bash
g++ -std=c++17 -O2 chapter_4/linear_regression_gd.cpp -o chapter_4/linear_reg
./chapter_4/linear_reg
```

## Logistic regression

```bash
g++ -std=c++17 -O2 chapter_4/logistic_regression_gd.cpp -o chapter_4/logistic_reg
./chapter_4/logistic_reg
```

## LibTorch neuron demo

```bash
g++ -std=c++17 chapter_4/neuron_demo.cpp -o chapter_4/neuron_demo \
  -I ./libtorch/include \
  -I ./libtorch/include/torch/csrc/api/include \
  -L ./libtorch/lib \
  -Wl,-rpath,./libtorch/lib \
  -ltorch -ltorch_cpu -lc10

DYLD_LIBRARY_PATH=./libtorch/lib ./chapter_4/neuron_demo
```

This prints the input, logits, and output shapes and shows a sample ReLU output.

## From-scratch Eigen MLP

```bash
g++ -std=c++17 -O2 chapter_4/mlp_eigen_demo.cpp chapter_4/neural_network.cpp \
  -o chapter_4/mlp_eigen_demo \
  -I"$(brew --prefix eigen)/include/eigen3"

./chapter_4/mlp_eigen_demo
```

This trains a small `2 -> 3 -> 1` network and prints the loss every 500 epochs plus final predictions.

## Notes

- The chapter text describes the `NeuralNetwork` header first but does not name the runnable training file explicitly, so `mlp_eigen_demo.cpp` was generated to make the Eigen implementation executable.
- `neuron_demo.cpp` automatically uses CUDA if LibTorch was built with CUDA support and a CUDA device is available; otherwise it runs on CPU.
