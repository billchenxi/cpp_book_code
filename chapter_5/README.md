# Chapter 5 Code

This folder groups the code referenced by chapter 5 of `B22398_05.docx`.

Files in this chapter:

- `eigen_mlp_xor.cpp`: official Packt Eigen MLP implementation for the XOR problem.
- `cuda_mlp.cu`: official Packt CUDA MLP walkthrough using CUDA, cuBLAS, and cuRAND.
- `libtorch_mlp.cpp`: official Packt LibTorch MLP implementation for the XOR problem.
- `optimizers.cpp`: official Packt optimizer showcase covering gradient descent, momentum, RMSprop, Adam, AdaGrad, and AdaDelta.

## Start

All commands below assume you run them from the repository root.

Quick checks:

```bash
clang++ --version
ls ./libtorch/include/torch/torch.h
brew --prefix eigen
```

If you want to build the CUDA example, also check:

```bash
nvcc --version
nvidia-smi
```

## Eigen MLP XOR

```bash
g++ -std=c++17 -O2 chapter_5/eigen_mlp_xor.cpp -o chapter_5/eigen_mlp_xor \
  -I"$(brew --prefix eigen)/include/eigen3"

./chapter_5/eigen_mlp_xor
```

## LibTorch MLP XOR

```bash
g++ -std=c++17 chapter_5/libtorch_mlp.cpp -o chapter_5/libtorch_mlp \
  -I ./libtorch/include \
  -I ./libtorch/include/torch/csrc/api/include \
  -L ./libtorch/lib \
  -Wl,-rpath,./libtorch/lib \
  -ltorch -ltorch_cpu -lc10

DYLD_LIBRARY_PATH=./libtorch/lib ./chapter_5/libtorch_mlp
```

## Optimizer showcase

```bash
g++ -std=c++17 -O2 chapter_5/optimizers.cpp -o chapter_5/optimizers \
  -I"$(brew --prefix eigen)/include/eigen3"

./chapter_5/optimizers
```

## CUDA MLP

```bash
nvcc -o chapter_5/cuda_mlp chapter_5/cuda_mlp.cu -lcublas -lcurand
./chapter_5/cuda_mlp
```

This file comes from the official chapter bundle and is best treated as the CUDA code walk-through for the chapter. It was not runnable in this environment because CUDA tooling is not installed here.

## Notes

- The chapter text links to Packt’s chapter 5 code bundle. The four files here were imported from that official repository path.
- The prose section on optimizers uses `torch::Tensor` snippets, but the official bundled source for this chapter’s optimizer code is `optimizers.cpp`, which demonstrates the same optimizer families with Eigen-based matrix updates.
