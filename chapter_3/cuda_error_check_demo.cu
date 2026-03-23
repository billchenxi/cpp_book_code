#include <cassert>
#include <cstdio>
#include <iostream>
#include <string>

#include <cuda_runtime.h>

inline cudaError_t checkCuda(cudaError_t result) {
    if (result != cudaSuccess) {
        std::fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
        assert(result == cudaSuccess);
    }
    return result;
}

__global__ void touch(float* out) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        out[0] = 42.0f;
    }
}

int main(int argc, char** argv) {
    bool badLaunch = argc > 1 && std::string(argv[1]) == "--bad-launch";

    float* value = nullptr;
    checkCuda(cudaMallocManaged(&value, sizeof(float)));
    value[0] = 0.0f;

    if (badLaunch) {
        int maxThreadsPerBlock = 0;
        checkCuda(cudaDeviceGetAttribute(
            &maxThreadsPerBlock,
            cudaDevAttrMaxThreadsPerBlock,
            0));

        touch<<<1, maxThreadsPerBlock + 1>>>(value);
        cudaError_t launchErr = cudaGetLastError();
        if (launchErr != cudaSuccess) {
            std::cout << "Launch error: " << cudaGetErrorString(launchErr) << '\n';
        }
    } else {
        touch<<<1, 1>>>(value);
        checkCuda(cudaGetLastError());
        checkCuda(cudaDeviceSynchronize());
        std::cout << "Value after kernel: " << value[0] << '\n';
    }

    checkCuda(cudaFree(value));
    return 0;
}

/*
Build from the repo root:
nvcc chapter_3/cuda_error_check_demo.cu -o chapter_3/cuda_error_check_demo

Run:
./chapter_3/cuda_error_check_demo
./chapter_3/cuda_error_check_demo --bad-launch
*/
