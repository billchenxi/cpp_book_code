# Chapter 1 Code

This folder collects the code examples used directly in chapter 1 of the book PDF:

- `example.cpp`: basic LibTorch tensor creation, arithmetic, and ReLU.
- `mnist_fc.cpp`: a small fully connected MNIST-style network that runs one forward pass.
- `welcome_from_cuda.cu`: generated from the chapter's CUDA/Colab snippet because it was not present in the repo.
- `simd_demo.cpp`: SIMD vector-add benchmark using AVX2 or NEON with a scalar fallback.

All commands below assume your current working directory is the repository root.

## Quick checks

```bash
clang++ --version
cmake --version
ls ./libtorch/include/torch/torch.h
```

If you want to run the CUDA example, also check:

```bash
nvcc --version
nvidia-smi
```

## Run the LibTorch tensor demo

```bash
clang++ chapter_1/example.cpp -o chapter_1/example \
  -I ./libtorch/include \
  -I ./libtorch/include/torch/csrc/api/include \
  -L ./libtorch/lib \
  -ltorch -ltorch_cpu -lc10 \
  -Wl,-rpath,./libtorch/lib \
  -std=c++17

./chapter_1/example
```

Expected output: three printed tensors named `X`, `Y`, and `Z`.

## Run the LibTorch MNIST-style network demo

```bash
clang++ chapter_1/mnist_fc.cpp -o chapter_1/mnist_fc \
  -I ./libtorch/include \
  -I ./libtorch/include/torch/csrc/api/include \
  -L ./libtorch/lib \
  -ltorch -ltorch_cpu -lc10 \
  -Wl,-rpath,./libtorch/lib \
  -std=c++17

./chapter_1/mnist_fc
```

Expected output: `Output size: [64, 10]`.

## Run the CUDA hello-kernel example

```bash
nvcc -std=c++14 -arch=sm_75 chapter_1/welcome_from_cuda.cu \
  -o chapter_1/welcome_from_cuda

./chapter_1/welcome_from_cuda
```

Expected output: nine lines, one for each block/thread pair in a `3 x 3` launch.

## Run the SIMD vector-add benchmark

```bash
g++ -std=c++17 -O3 chapter_1/simd_demo.cpp -o chapter_1/simd_demo
./chapter_1/simd_demo
```

Expected output: the active path (`AVX2`, `NEON`, or `Scalar fallback`), a zero max error check, and a timing summary.

## Platform notes

- The vendored `libtorch/` directory in this repo is a macOS arm64 CPU build, so `example.cpp` and `mnist_fc.cpp` can be built locally on Apple Silicon without CUDA.
- `welcome_from_cuda.cu` needs NVIDIA CUDA support. It will not run on Apple Silicon locally; use Google Colab or a Linux/Windows machine with an NVIDIA GPU instead.
- `simd_demo.cpp` works on Apple Silicon and x86_64. It auto-selects NEON or AVX2 when the compiler target enables those instructions, then falls back to scalar code for any remaining elements.
- If you prefer to `cd chapter_1` before building, change `./libtorch/...` to `../libtorch/...` in the commands above.
