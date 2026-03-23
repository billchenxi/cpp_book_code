# Chapter 2 Code

This folder groups the code referenced by chapter 2 of `B22398_02.docx`.

Notes:

- `time_feature_demo.cpp` was generated from the chapter snippet because it was missing from the repo.
- The draft mentions `scaling.robust_scale_demo.cpp`; the actual file here is `robust_scale_demo.cpp`.
- Small sample inputs remain in the repo root: `../input.jpg` and `../input.wav`.
- Large optional assets such as `../libtorch/`, `../bert_onnx/`, `../ner_onnx/`, and `../sent_onnx/` are not committed to GitHub; download or export them by following the root `README.md` before running the dependent demos.
- Most source files already contain their most precise build command in a comment near the top. If a file comment differs from this README, use the file comment.

## Start

Run the chapter from inside this folder so outputs land here:

```bash
cd chapter_2
clang++ --version
cmake --version
ls ../libtorch/include/torch/torch.h
```

If `../libtorch/include/torch/torch.h` is missing, fetch LibTorch first using the root `README.md` instructions.

## Standard-library demos

For the pure standard-library examples, this pattern is usually enough:

```bash
g++ -std=c++17 -O2 FILE.cpp -o FILE_NOEXT
./FILE_NOEXT
```

This covers:

- `mean_substitution_example.cpp`
- `forward_fill_example.cpp`
- `backward_fill_example.cpp`
- `knn_imputation_example.cpp`
- `knn_imputation_multidim_example.cpp`
- `regression_imputation_example.cpp`
- `one_hot_example.cpp`
- `frequency_encode_example.cpp`
- `ordinal_encode_example.cpp`
- `binary_encoding_demo.cpp`
- `embedding_encode_demo.cpp`
- `minmax_scale_demo.cpp`
- `zscore_normalize_demo.cpp`
- `robust_scale_demo.cpp`
- `log_transform_demo.cpp`
- `power_transform_demo.cpp`
- `tsne_demo.cpp`
- `rolling_mean_demo.cpp`
- `exponential_smoothing_demo.cpp`
- `differencing_demo.cpp`
- `fft_demo.cpp`
- `time_feature_demo.cpp`
- `polynomial_features_demo.cpp`
- `interaction_terms_demo.cpp`
- `pad_truncate_demo.cpp`
- `token_dropout_demo.cpp`

Example:

```bash
g++ -std=c++17 -O2 rolling_mean_demo.cpp -o rolling_mean_demo
./rolling_mean_demo
```

## Large-data demo

`bigdata_demo.cpp` has its own CLI:

```bash
g++ -std=c++17 -O2 bigdata_demo.cpp -o bigdata_demo

./bigdata_demo make-sample sample.bin 67108864
./bigdata_demo mmap sample.bin
./bigdata_demo batch sample.bin 1048576
./bigdata_demo range sample.bin 2097152
```

## Eigen, Armadillo, and mlpack demos

These require extra libraries. Representative commands:

```bash
g++ -std=c++17 -O2 pca_demo.cpp -o pca_demo \
  -I"$(brew --prefix eigen)/include/eigen3"
./pca_demo
```

```bash
g++ -std=c++17 -O2 armadillo_scaling_demo.cpp -o armadillo_scaling_demo \
  $(pkg-config --cflags --libs armadillo)
./armadillo_scaling_demo
```

```bash
g++ -std=c++17 mlpack_impute_demo.cpp -o mlpack_impute_demo \
  $(pkg-config --cflags --libs mlpack armadillo) \
  -I"$(brew --prefix cereal)/include" \
  -Wno-deprecated-declarations
./mlpack_impute_demo
```

Other files in this group:

- `glove_eigen_demo.cpp`
- `mlpack_encoding_demo_min.cpp`
- `pca_mlpack4_demo.cpp`
- `poly_interact_armadillo_demo.cpp`
- `ts_preproc_armadillo_opencv_demo.cpp`

## Image, audio, and multimodal demos

OpenCV image preprocessing:

```bash
g++ -std=c++17 -O2 image_preproc_demo.cpp -o image_preproc_demo \
  $(pkg-config --cflags --libs opencv4)
./image_preproc_demo ../input.jpg
```

Multimodal preprocessing:

```bash
g++ -std=c++17 -O2 multimodal_preproc_demo.cpp -o multimodal_preproc_demo \
  $(pkg-config --cflags --libs opencv4)
./multimodal_preproc_demo ../input.jpg
```

Spectrogram generation:

```bash
g++ -std=c++17 spectrogram_demo.cpp -o spectrogram_demo \
  $(pkg-config --cflags --libs fftw3 sndfile opencv4)
./spectrogram_demo ../input.wav
```

## Text and ONNX demos

Classic tokenization and stemming needs Snowball:

```bash
g++ -std=c++17 -O2 tokenize_stem_demo.cpp -o tokenize_stem_demo \
  -I"$(brew --prefix)/include" \
  -L"$(brew --prefix)/lib" \
  -lstemmer
./tokenize_stem_demo
```

Token-dropout augmentation without external libraries:

```bash
g++ -std=c++17 -O2 token_dropout_demo.cpp -o token_dropout_demo
./token_dropout_demo
```

BERT ONNX Runtime demo:

```bash
g++ -std=c++17 -O2 bert_onnx_demo.cpp -o bert_demo \
  -I"$(brew --prefix onnxruntime)/include" \
  -L"$(brew --prefix onnxruntime)/lib" \
  -lonnxruntime \
  -Wl,-rpath,"$(brew --prefix onnxruntime)/lib"
./bert_demo ../bert_onnx/onnx/model.onnx
```

Download `../bert_onnx/` first if that path does not exist.

Advanced NER and sentiment ONNX Runtime demo:

```bash
g++ -std=c++17 -O2 nlp_advanced_demo.cpp -o nlp_advanced_demo \
  -I"$(brew --prefix onnxruntime)/include" \
  -L"$(brew --prefix onnxruntime)/lib" \
  -lonnxruntime \
  -Wl,-rpath,"$(brew --prefix onnxruntime)/lib"

./nlp_advanced_demo ner ../ner_onnx/model.onnx
./nlp_advanced_demo sentiment ../sent_onnx/model.onnx
```

Export `../ner_onnx/` and `../sent_onnx/` first if those paths do not exist.

## LibTorch demos

Autoencoder examples:

```bash
g++ -std=c++17 autoencoder_demo.cpp -o autoencoder_demo \
  -I ../libtorch/include \
  -I ../libtorch/include/torch/csrc/api/include \
  -L ../libtorch/lib \
  -Wl,-rpath,../libtorch/lib \
  -ltorch -ltorch_cpu -lc10
DYLD_LIBRARY_PATH=../libtorch/lib ./autoencoder_demo
```

```bash
g++ -std=c++17 autoencoder_libtorch_demo.cpp -o autoencoder_libtorch_demo \
  -I ../libtorch/include \
  -I ../libtorch/include/torch/csrc/api/include \
  -L ../libtorch/lib \
  -Wl,-rpath,../libtorch/lib \
  -ltorch -ltorch_cpu -lc10
DYLD_LIBRARY_PATH=../libtorch/lib ./autoencoder_libtorch_demo
```

These LibTorch demos require a local `../libtorch/` download.

LibTorch dataset subproject:

```bash
cmake -S data_pipeline -B data_pipeline/build_chapter_2
cmake --build data_pipeline/build_chapter_2 -j
./data_pipeline/build_chapter_2/custom_dataset_demo
```

## Outputs

Many demos write generated files into the current working directory. If you run them from `chapter_2`, outputs such as `processed_data.csv`, `normalized.png`, `noisy.png`, `spectrogram.png`, and `sample.bin` will appear here.
