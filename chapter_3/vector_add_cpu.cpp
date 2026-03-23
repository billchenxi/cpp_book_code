#include <algorithm>
#include <cmath>
#include <iostream>

// Perform element-wise accumulation of two vectors.
void vectorAddCPU(int length, const float* inputA, float* inputB) {
    for (int idx = 0; idx < length; ++idx) {
        inputB[idx] += inputA[idx];
    }
}

int main() {
    const int size = 1 << 20;  // 1,048,576 elements

    float* vecA = new float[size];
    float* vecB = new float[size];

    for (int i = 0; i < size; ++i) {
        vecA[i] = 0.5f;
        vecB[i] = 2.5f;
    }

    vectorAddCPU(size, vecA, vecB);

    float maxError = 0.0f;
    for (int i = 0; i < size; ++i) {
        maxError = std::max(maxError, std::fabs(vecB[i] - 3.0f));
    }

    std::cout << "Max error: " << maxError << '\n';

    delete[] vecA;
    delete[] vecB;
    return 0;
}

/*
Build from the repo root:
g++ -std=c++17 -O2 chapter_3/vector_add_cpu.cpp -o chapter_3/vector_add_cpu

Run:
./chapter_3/vector_add_cpu
*/
