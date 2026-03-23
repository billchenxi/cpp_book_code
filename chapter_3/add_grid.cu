#include <cmath>
#include <iostream>

#include <cuda_runtime.h>

__global__ void add(int n, const float* x, float* y) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < n; i += stride) {
        y[i] = x[i] + y[i];
    }
}

int main() {
    const int n = 1 << 20;
    const int blockSize = 256;
    const int numBlocks = (n + blockSize - 1) / blockSize;

    float* x = nullptr;
    float* y = nullptr;

    cudaMallocManaged(&x, n * sizeof(float));
    cudaMallocManaged(&y, n * sizeof(float));

    for (int i = 0; i < n; ++i) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    add<<<numBlocks, blockSize>>>(n, x, y);
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
nvcc chapter_3/add_grid.cu -o chapter_3/add_grid

Run:
./chapter_3/add_grid
*/
