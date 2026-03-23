# Chapter 3 Code

This folder groups the code examples referenced by chapter 3 of `B22398_03.docx`.

Notes:

- The draft uses both `some-CUDA.cu` and `add.cu` for the first CUDA example. In this repo, `add.cu` is the canonical file and `some-CUDA.cu` is a thin alias so either draft command still works.
- The debugging section in the draft does not name a file, so `cuda_error_check_demo.cu` was generated from that snippet.
- These CUDA examples require `nvcc` and an NVIDIA GPU runtime. They will not run on Apple Silicon locally.

## Start

From the repo root, verify the toolchain:

```bash
nvcc --version
nvidia-smi
```

If you do not have a local NVIDIA GPU, use Google Colab or another CUDA-capable Linux or Windows machine.

## CPU baseline

```bash
g++ -std=c++17 -O2 chapter_3/vector_add_cpu.cpp -o chapter_3/vector_add_cpu
./chapter_3/vector_add_cpu
```

Expected output: `Max error: 0`.

## First CUDA baseline

This is the single-thread CUDA version used as the first GPU port of the CPU baseline:

```bash
nvcc chapter_3/add.cu -o chapter_3/add_cuda
./chapter_3/add_cuda
```

If you want to match the draft's placeholder filename, this also works:

```bash
nvcc -arch=sm_70 chapter_3/some-CUDA.cu -o chapter_3/out
./chapter_3/out
```

Adjust `sm_70` to match your GPU if needed.

## Single-block CUDA version

```bash
nvcc chapter_3/add_block.cu -o chapter_3/add_block
./chapter_3/add_block
```

This launches `1` block with `256` threads and uses a grid-stride loop over the array.

## Multi-block CUDA version

```bash
nvcc chapter_3/add_grid.cu -o chapter_3/add_grid
./chapter_3/add_grid
```

This computes a flat global thread index and launches enough blocks to cover the input.

## CUDA error-check demo

```bash
nvcc chapter_3/cuda_error_check_demo.cu -o chapter_3/cuda_error_check_demo
./chapter_3/cuda_error_check_demo
./chapter_3/cuda_error_check_demo --bad-launch
```

The first run shows a successful kernel plus synchronization check. The second intentionally requests an invalid launch configuration so you can see the launch error path.

## Profiling

The draft uses `nvprof`:

```bash
nvprof ./chapter_3/add_cuda
nvprof ./chapter_3/add_block
nvprof ./chapter_3/add_grid
```

If your CUDA toolkit no longer ships `nvprof`, use Nsight Systems or Nsight Compute instead.
