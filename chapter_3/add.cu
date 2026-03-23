#include <cmath>
#include <iostream>

#include <cuda_runtime.h>

__global__ void add(int n, const float* x, float* y) {
    int index = threadIdx.x;
    int stride = blockDim.x;

    for (int i = index; i < n; i += stride) {
        y[i] = x[i] + y[i];
    }
}

int main() {
    const int n = 1 << 20;
    float* x = nullptr;
    float* y = nullptr;

    cudaMallocManaged(&x, n * sizeof(float));
    cudaMallocManaged(&y, n * sizeof(float));

    for (int i = 0; i < n; ++i) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    // Single-thread kernel launch for the first CUDA baseline.
    add<<<1, 1>>>(n, x, y);
    cudaDeviceSynchronize();

    float maxError = 0.0f;
    for (int i = 0; i < n; ++i) {
        maxError = fmax(maxError, fabs(y[i] - 3.0f));
    }

    std::cout << "Max error: " << maxError << '\n';

    cudaFree(x);
    cudaFree(y);
    return 0;
}

/*
Build from the repo root:
nvcc chapter_3/add.cu -o chapter_3/add_cuda

Run:
./chapter_3/add_cuda
*/
