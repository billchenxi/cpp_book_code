# C++ ML and Data Processing Demos

This repository is a collection of small, mostly self-contained C++ programs for machine learning, data preprocessing, model inference, and signal or image processing. It is a learning repo, not a single application: most source files are designed to be compiled and run independently.

The tree currently contains:

- dozens of source or header files (`*.cpp`, `*.cu`, `*.h`)
- standalone demos grouped mostly in chapter folders
- two small CMake subprojects in `chapter_2/data_pipeline/` and `data_pipeline_caffe2/`
- sample inputs `input.jpg` and `input.wav`
- optional large local downloads expected in `libtorch/`, `bert_onnx/`, `ner_onnx/`, and `sent_onnx/` after following the setup instructions below

Most source files already include an exact build or run comment near the top. Start there for the most specific command for a given demo.

Contributor expectations live in [CONTRIBUTING.md](CONTRIBUTING.md).

## What The Repo Covers

- tabular preprocessing: scaling, normalization, imputation, encoding, feature engineering
- time-series preprocessing: rolling mean, differencing, smoothing, FFT-related examples
- linear and logistic regression from scratch
- dimensionality reduction: PCA, t-SNE, autoencoders
- NLP preprocessing: tokenization, stemming, padding, categorical encodings, GloVe-style embeddings
- deep learning with LibTorch
- ONNX Runtime inference for BERT, NER, and sentiment models
- monitoring, observability, and online quality tracking for deployed inference services
- explainability with LIME, KernelSHAP, and Grad-CAM
- image and audio preprocessing with OpenCV, FFTW, and libsndfile
- systems-oriented examples such as memory-mapped IO and SIMD vectorization

See [docs/demo-index.md](docs/demo-index.md) for the source-by-source index.
Chapter 1 examples are grouped in [chapter_1/README.md](chapter_1/README.md).
Chapter 2 examples are grouped in [chapter_2/README.md](chapter_2/README.md).
Chapter 3 examples are grouped in [chapter_3/README.md](chapter_3/README.md).
Chapter 4 examples are grouped in [chapter_4/README.md](chapter_4/README.md).
Chapter 5 examples are grouped in [chapter_5/README.md](chapter_5/README.md).
Chapter 10 examples are grouped in [chapter_10/README.md](chapter_10/README.md).
Chapter 11 examples are grouped in [chapter_11/README.md](chapter_11/README.md).
Chapter 12 examples are grouped in [chapter_12/README.md](chapter_12/README.md).
Chapter 13 examples are grouped in [chapter_13/README.md](chapter_13/README.md).

## Repository Layout

- `chapter_*/`: standalone demos grouped by book chapter
- `chapter_1/`: chapter 1 examples and run notes collected from the book PDF
- `chapter_2/`: chapter 2 examples and run notes collected from the chapter DOCX
- `chapter_3/`: CUDA vector-add and error-checking examples collected from the chapter DOCX
- `chapter_4/`: regression, neuron, and from-scratch MLP examples collected from the chapter DOCX
- `chapter_5/`: Eigen, LibTorch, CUDA, and optimizer examples collected from the chapter DOCX and Packt bundle
- `chapter_10/`: deployment, inference, micro-batching, TorchScript, and ONNX Runtime examples collected from the chapter DOCX
- `chapter_11/`: drift monitoring, observability, debugging, and retraining examples collected from the chapter DOCX
- `chapter_12/`: SLIs, SLOs, metrics export, traces, logs, and online-quality monitoring examples collected from the chapter PDF
- `chapter_13/`: LIME, KernelSHAP, and Grad-CAM explainability examples collected from the chapter DOCX
- `chapter_2/data_pipeline/`: LibTorch dataset and dataloader example with a small CMake project
- `data_pipeline_caffe2/`: Caffe2 normalization example with a small CMake project
- `bert_onnx/`: local download target for the BERT tokenizer files and ONNX variants
- `ner_onnx/`: local export target for the token-classification ONNX model assets
- `sent_onnx/`: local export target for the sentiment-classification ONNX model assets
- `libtorch/`: local download target for a LibTorch distribution that matches your platform

## Dependency Overview

The repo mixes standard-library-only programs with examples that depend on external libraries.

Base toolchain:

- C++17 compiler (`clang++`, `g++`, or `c++`)
- CMake for the two subprojects
- `pkg-config` for many of the third-party examples

Optional libraries used by subsets of the repo:

- LibTorch
- ONNX Runtime
- OpenCV
- FFTW
- libsndfile
- Armadillo
- mlpack
- Eigen
- Snowball / `libstemmer`
- cereal headers for one mlpack example

On macOS with Homebrew, a typical setup looks like:

```bash
brew install cmake pkg-config opencv fftw libsndfile onnxruntime armadillo mlpack eigen snowball cereal
```

Notes:

- This GitHub repo intentionally excludes large downloaded assets such as LibTorch and the ONNX model bundles.
- The expected local paths still remain `libtorch/`, `bert_onnx/`, `ner_onnx/`, and `sent_onnx/`, so the source comments and build commands stay simple.
- Several compile comments assume Homebrew-style paths such as `$(brew --prefix onnxruntime)`.
- Some demos require `DYLD_LIBRARY_PATH` on macOS so the runtime can find shared libraries.

## Downloaded Assets

This repository is pushed as source code plus small sample inputs. Large third-party downloads are not committed to GitHub because they exceed practical clone size and some exceed GitHub's regular file-size limits.

### LibTorch

Most LibTorch demos assume a local `libtorch/` directory at the repo root. For macOS arm64 CPU with LibTorch `2.8.0`, one working setup is:

```bash
curl -L -o libtorch-macos-arm64-2.8.0.zip \
  https://download.pytorch.org/libtorch/cpu/libtorch-macos-arm64-2.8.0.zip
unzip libtorch-macos-arm64-2.8.0.zip
```

If you are on another platform, download the matching LibTorch package from the official PyTorch LibTorch page and unpack it to `./libtorch`.

### BERT ONNX bundle

`chapter_2/bert_onnx_demo.cpp` expects a local `bert_onnx/` folder. A convenient way to populate it is:

```bash
python3 -m pip install -U huggingface_hub
huggingface-cli download onnx-community/bert-base-uncased --local-dir bert_onnx
```

This provides the tokenizer files and the `bert_onnx/onnx/model.onnx` bundle used by the demo.

### NER and sentiment ONNX exports

`chapter_2/nlp_advanced_demo.cpp` expects `ner_onnx/model.onnx` and `sent_onnx/model.onnx`. The chapter code already documents the export flow; the same commands are repeated here:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U "optimum[exporters,onnxruntime]" onnx

optimum-cli export onnx \
  --task token-classification \
  --opset 14 \
  --model dbmdz/bert-base-cased-finetuned-conll03-english \
  ner_onnx/

optimum-cli export onnx \
  --task text-classification \
  --opset 14 \
  --model distilbert-base-uncased-finetuned-sst-2-english \
  sent_onnx/
```

If these assets are missing, the model-dependent demos and the full verification script will skip those cases instead of failing.

## Quick Start

### 1. Standard-library-only demo

```bash
g++ -std=c++17 -O2 chapter_2/rolling_mean_demo.cpp -o chapter_2/roll
./chapter_2/roll
```

### 2. OpenCV image preprocessing demo

```bash
g++ -std=c++17 -O2 chapter_2/image_preproc_demo.cpp -o chapter_2/image_preproc_demo \
  $(pkg-config --cflags --libs opencv4)
./chapter_2/image_preproc_demo input.jpg
```

This writes `out_resized.jpg`, `out_cropped.jpg`, `out_flipped.jpg`, `out_rotated.jpg`, and other `out_*.jpg` files to the working directory.

### 3. Audio spectrogram demo

```bash
g++ -std=c++17 chapter_2/spectrogram_demo.cpp -o chapter_2/spec_demo \
  $(pkg-config --cflags --libs fftw3 sndfile opencv4)
./chapter_2/spec_demo input.wav
```

This writes `spectrogram.png`.

### 4. LibTorch demo using a local `libtorch/` download

```bash
g++ -std=c++17 chapter_2/autoencoder_demo.cpp -o chapter_2/autoencoder_demo \
  -I libtorch/include \
  -I libtorch/include/torch/csrc/api/include \
  -L libtorch/lib \
  -ltorch -ltorch_cpu -lc10

DYLD_LIBRARY_PATH=libtorch/lib ./chapter_2/autoencoder_demo
```

### 5. ONNX Runtime demo

```bash
g++ -std=c++17 -O2 chapter_2/bert_onnx_demo.cpp -o chapter_2/bert_demo \
  -I"$(brew --prefix onnxruntime)/include" \
  -L"$(brew --prefix onnxruntime)/lib" \
  -lonnxruntime \
  -Wl,-rpath,"$(brew --prefix onnxruntime)/lib"

./chapter_2/bert_demo bert_onnx/onnx/model.onnx
```

## Common Build Patterns

### Standalone file with only the standard library

```bash
g++ -std=c++17 -O2 some_demo.cpp -o some_demo
./some_demo
```

### Standalone file using `pkg-config`

```bash
g++ -std=c++17 -O2 some_demo.cpp -o some_demo \
  $(pkg-config --cflags --libs opencv4 armadillo mlpack)
```

Use only the packages that the source file actually needs.

### Standalone file using LibTorch

```bash
LIBTORCH=./libtorch
g++ -std=c++17 some_libtorch_demo.cpp -o some_libtorch_demo \
  -I"$LIBTORCH/include" \
  -I"$LIBTORCH/include/torch/csrc/api/include" \
  -L"$LIBTORCH/lib" \
  -Wl,-rpath,"$LIBTORCH/lib" \
  -ltorch -ltorch_cpu -lc10
```

### CMake subproject using LibTorch or Caffe2

```bash
cmake -S chapter_2/data_pipeline -B chapter_2/data_pipeline/build_chapter_2
cmake --build chapter_2/data_pipeline/build_chapter_2 -j

cmake -S data_pipeline_caffe2 -B data_pipeline_caffe2/build \
  -DTorch_DIR=./libtorch/share/cmake/Torch
cmake --build data_pipeline_caffe2/build -j
```

## Suggested Reading Order

If you are new to the repo, a reasonable path is:

1. Start with the standard-library preprocessing demos such as `chapter_2/rolling_mean_demo.cpp`, `chapter_2/zscore_normalize_demo.cpp`, `chapter_2/forward_fill_example.cpp`, and `chapter_2/polynomial_features_demo.cpp`.
2. Move to matrix-library examples such as `chapter_2/armadillo_scaling_demo.cpp`, `chapter_2/pca_demo.cpp`, and `chapter_2/pca_mlpack4_demo.cpp`.
3. Then look at `chapter_2/image_preproc_demo.cpp`, `chapter_2/spectrogram_demo.cpp`, and `chapter_2/multimodal_preproc_demo.cpp` for non-tabular data.
4. Finish with `chapter_1/example.cpp`, `chapter_1/mnist_fc.cpp`, `chapter_2/autoencoder_demo.cpp`, `chapter_2/bert_onnx_demo.cpp`, and `chapter_2/nlp_advanced_demo.cpp`.
5. Then move to `chapter_3/add.cu`, `chapter_3/add_block.cu`, and `chapter_3/add_grid.cu` for the low-level CUDA progression.
6. After that, study `chapter_4/linear_regression_gd.cpp`, `chapter_4/logistic_regression_gd.cpp`, `chapter_4/neuron_demo.cpp`, and `chapter_4/mlp_eigen_demo.cpp`.
7. Then continue with `chapter_5/eigen_mlp_xor.cpp`, `chapter_5/libtorch_mlp.cpp`, `chapter_5/optimizers.cpp`, and `chapter_5/cuda_mlp.cu`.
8. Then move to `chapter_10/trace_and_save_ts.cpp`, `chapter_10/infer_torchscript.cpp`, `chapter_10/parity_check.cpp`, and `chapter_10/micro_batcher_demo.cpp` for deployment-oriented inference patterns.
9. After that, study `chapter_11/fixed_bin_histogram_psi.cpp`, `chapter_11/structured_logging_demo.cpp`, `chapter_11/metrics_tracing_demo.cpp`, and `chapter_11/micro_batcher_debug_demo.cpp` for drift monitoring and production debugging patterns.
10. Then continue with `chapter_12/metrics.hpp`, `chapter_12/metrics_server.cpp`, `chapter_12/trace.hpp`, `chapter_12/observability_demo.cpp`, and `chapter_12/quality_monitor_demo.cpp` for monitoring and online-quality instrumentation patterns.
11. Then finish with `chapter_13/explainability.hpp`, `chapter_13/lime_shap_demo.cpp`, and `chapter_13/gradcam_demo.cpp` for model explainability patterns in tabular and vision settings.

## Large Downloaded Assets

These assets are intentionally not committed to GitHub. At the time of writing, the main optional downloads are roughly:

- `libtorch/` is about `325 MB`
- `bert_onnx/` is about `1.3 GB`
- `ner_onnx/` is about `412 MB`
- `sent_onnx/` is about `256 MB`
- `sample.bin` is about `64 MB`

If clone size matters, these are the main reason. `sample.bin` is not downloaded; it is generated locally by `chapter_2/bigdata_demo.cpp`.

## Notes And Caveats

- Many demos write outputs into the current working directory.
- Generated demo outputs and downloaded dependencies are ignored by git.
- `data_pipeline_caffe2/` remains a standalone support subproject at the repo root because it has its own CMake-based build flow.
- Build commands in source comments are the most precise instructions for that file when they differ from this README.
