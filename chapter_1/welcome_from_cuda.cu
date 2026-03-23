#include <cstdio>
#include <cuda_runtime.h>

__global__ void welcome() {
    printf("Welcome to DL with C++ - block %d, thread %d\n", blockIdx.x, threadIdx.x);
}

int main() {
    welcome<<<3, 3>>>();

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    return 0;
}

/*
Build on a CUDA-enabled machine or in Colab:
nvcc -std=c++14 -arch=sm_75 chapter_1/welcome_from_cuda.cu \
  -o chapter_1/welcome_from_cuda

Run:
./chapter_1/welcome_from_cuda
*/
