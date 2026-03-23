#include <cstddef>
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cmath>

// chapter_1/simd_demo.cpp
// g++ -std=c++17 -O3 chapter_1/simd_demo.cpp -o chapter_1/simd_demo
// ./chapter_1/simd_demo

#if defined(__AVX2__)
  #include <immintrin.h>
#elif defined(__ARM_NEON)
  #include <arm_neon.h>
#endif

// SIMD + scalar tail
void vectorAddSIMD(const float* a, const float* b, float* c, std::size_t n) {
    std::size_t i = 0;

#if defined(__AVX2__)
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 vc = _mm256_add_ps(va, vb);
        _mm256_storeu_ps(c + i, vc);
    }
#elif defined(__ARM_NEON)
    for (; i + 4 <= n; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        float32x4_t vc = vaddq_f32(va, vb);
        vst1q_f32(c + i, vc);
    }
#endif
    for (; i < n; ++i) c[i] = a[i] + b[i];
}

// Reference (scalar) for correctness check
void vectorAddRef(const float* a, const float* b, float* c, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) c[i] = a[i] + b[i];
}

int main(int argc, char** argv) {
    // Problem size (can override via CLI): ./simd_demo 10000000 50
    std::size_t N = (argc > 1) ? static_cast<std::size_t>(std::stoll(argv[1])) : (1 << 22); // ~4M
    int iters = (argc > 2) ? std::stoi(argv[2]) : 20;

    std::cout
#if defined(__AVX2__)
        << "Path: AVX2 (x86_64)\n";
#elif defined(__ARM_NEON)
        << "Path: NEON (arm64)\n";
#else
        << "Path: Scalar fallback\n";
#endif

    std::cout << "N = " << N << ", iters = " << iters << "\n";

    std::vector<float> a(N), b(N), c(N), cref(N);
    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (std::size_t i = 0; i < N; ++i) { a[i] = dist(rng); b[i] = dist(rng); }

    // Correctness
    vectorAddSIMD(a.data(), b.data(), c.data(), N);
    vectorAddRef(a.data(), b.data(), cref.data(), N);
    double max_abs_err = 0.0;
    for (std::size_t i = 0; i < N; ++i) {
        max_abs_err = std::max(max_abs_err, std::abs(static_cast<double>(c[i]) - cref[i]));
    }
    std::cout << "Max abs error vs. reference: " << max_abs_err << "\n";

    // Warmup
    for (int w = 0; w < 3; ++w) vectorAddSIMD(a.data(), b.data(), c.data(), N);

    // Benchmark
    double total_ms = 0.0;
    for (int t = 0; t < iters; ++t) {
        auto t0 = std::chrono::high_resolution_clock::now();
        vectorAddSIMD(a.data(), b.data(), c.data(), N);
        auto t1 = std::chrono::high_resolution_clock::now();
        total_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
    double avg_ms = total_ms / iters;

    // Effective bandwidth ~ 3 arrays * N * 4 bytes (read a, read b, write c)
    double bytes = 3.0 * static_cast<double>(N) * sizeof(float);
    double gbps = (bytes / (avg_ms / 1000.0)) / 1e9;

    std::cout << "Avg time: " << avg_ms << " ms per call\n";
    std::cout << "Effective bandwidth: " << gbps << " GB/s\n";
    std::cout << "Samples c[0..4]: ";
    for (int i = 0; i < 5 && i < static_cast<int>(N); ++i) {
        std::cout << c[i] << (i < 4 ? ", " : "\n");
    }
    return 0;
}
