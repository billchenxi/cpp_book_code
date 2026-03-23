#include <cassert>
#include <iostream>

#include <cuda_runtime.h>

#define CUDA_CHECK(call) do {                                      \
    cudaError_t err__ = (call);                                    \
    if (err__ != cudaSuccess) {                                    \
        std::cerr << "CUDA error "                                 \
                  << cudaGetErrorString(err__)                     \
                  << " @ " << __FILE__ << ":" << __LINE__ << "\n"; \
        assert(false);                                             \
    }                                                              \
} while (0)

__global__ void add_one(float* data, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    data[idx] += 1.0f;
  }
}

int main() {
  constexpr int n = 8;
  float* data = nullptr;

  CUDA_CHECK(cudaMallocManaged(&data, n * sizeof(float)));
  for (int i = 0; i < n; ++i) {
    data[i] = static_cast<float>(i);
  }

  add_one<<<1, 32>>>(data, n);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  for (int i = 0; i < n; ++i) {
    std::cout << data[i] << (i + 1 < n ? " " : "\n");
  }

  CUDA_CHECK(cudaFree(data));
  return 0;
}
