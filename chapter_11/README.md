# Chapter 11 Code

This folder groups the code referenced by chapter 11 of `B22398_11_edited_mar_15_26 (1).docx`.

Files in this chapter:

- `fixed_bin_histogram_psi.cpp`: fixed-bin histogram plus Population Stability Index drift demo.
- `observability_utils.h`: shared logging and trace-span helpers for the chapter 11 observability demos.
- `structured_logging_demo.cpp`: structured JSON-line logging helper and example usage.
- `metrics_tracing_demo.cpp`: in-process metrics plus scoped tracing spans.
- `repro_pack_and_sentinels_demo.cpp`: repro-pack metadata, seeded RNG replay, NaN or Inf sentinels, and stable log-sum-exp.
- `micro_batcher_debug_demo.cpp`: standard-library micro-batcher showing queue time vs compute time.
- `cuda_check_macro_demo.cu`: CUDA error-check macro example turned into a small runnable CUDA program.
- `acceptance.yaml`: example offline-plus-systems acceptance gate for promotion decisions.

## Start

All commands below assume you run them from the repository root.

Standard-library demos:

```bash
g++ -std=c++17 -O2 -Wall -Wextra -pedantic chapter_11/fixed_bin_histogram_psi.cpp -o chapter_11/fixed_bin_histogram_psi
./chapter_11/fixed_bin_histogram_psi
```

```bash
g++ -std=c++17 -O2 chapter_11/structured_logging_demo.cpp -o chapter_11/structured_logging_demo
./chapter_11/structured_logging_demo
```

```bash
g++ -std=c++17 -O2 chapter_11/metrics_tracing_demo.cpp -o chapter_11/metrics_tracing_demo
./chapter_11/metrics_tracing_demo
```

```bash
g++ -std=c++17 -O2 chapter_11/repro_pack_and_sentinels_demo.cpp -o chapter_11/repro_pack_and_sentinels_demo
./chapter_11/repro_pack_and_sentinels_demo
```

```bash
g++ -std=c++17 -O2 -pthread chapter_11/micro_batcher_debug_demo.cpp -o chapter_11/micro_batcher_debug_demo
./chapter_11/micro_batcher_debug_demo
```

The acceptance gate is configuration, not code:

```bash
sed -n '1,120p' chapter_11/acceptance.yaml
```

## CUDA debugging demo

This machine needs a working NVIDIA CUDA toolchain to build the CUDA-specific example.

```bash
nvcc --version
nvidia-smi
```

```bash
nvcc -O2 -o chapter_11/cuda_check_macro_demo chapter_11/cuda_check_macro_demo.cu
./chapter_11/cuda_check_macro_demo
```

## Notes

- `fixed_bin_histogram_psi.cpp` was moved from the old `ch11/` folder into `chapter_11/`.
- `structured_logging_demo.cpp` and `metrics_tracing_demo.cpp` share `observability_utils.h` so the logging helper and trace spans stay consistent.
- `micro_batcher_debug_demo.cpp` uses a small `std::vector<float>` tensor placeholder so the queueing example is runnable without LibTorch.
- `cuda_check_macro_demo.cu` is generated from the chapter’s CUDA macro snippet; it was not verifiable here without `nvcc`.
