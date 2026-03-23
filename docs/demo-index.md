# Demo Index

This index groups the examples by topic and dependency. For exact compiler flags, check the comment block at the top of each source file when present.
Chapter 2 examples are grouped in `chapter_2/README.md`.
Chapter 3 examples are grouped in `chapter_3/README.md`.
Chapter 4 examples are grouped in `chapter_4/README.md`.
Chapter 5 examples are grouped in `chapter_5/README.md`.
Chapter 10 examples are grouped in `chapter_10/README.md`.
Chapter 11 examples are grouped in `chapter_11/README.md`.
Chapter 12 examples are grouped in `chapter_12/README.md`.
Chapter 13 examples are grouped in `chapter_13/README.md`.

## Standard Library: Preprocessing, Models, And Utilities

- `chapter_2/backward_fill_example.cpp`: backward-fill missing values in a 1D sequence.
- `chapter_2/bigdata_demo.cpp`: compare sample-file generation, memory mapping, streaming batch reads, and ranged reads.
- `chapter_2/binary_encoding_demo.cpp`: encode categories into binary feature vectors.
- `chapter_2/differencing_demo.cpp`: first-order differencing for time-series preprocessing.
- `chapter_2/embedding_encode_demo.cpp`: simple embedding-style categorical encoding with synthetic vectors.
- `chapter_2/exponential_smoothing_demo.cpp`: exponential smoothing over a numeric series.
- `chapter_2/fft_demo.cpp`: naive discrete Fourier transform for a toy signal.
- `chapter_2/forward_fill_example.cpp`: forward-fill missing values in a 1D sequence.
- `chapter_2/frequency_encode_example.cpp`: frequency encoding for repeated categories.
- `chapter_2/interaction_terms_demo.cpp`: build pairwise interaction features.
- `chapter_2/knn_imputation_example.cpp`: 1D k-nearest-neighbor imputation.
- `chapter_2/knn_imputation_multidim_example.cpp`: multidimensional k-nearest-neighbor imputation.
- `chapter_4/linear_regression_gd.cpp`: linear regression trained with gradient descent.
- `chapter_2/log_transform_demo.cpp`: log transform demo for skewed features.
- `chapter_4/logistic_regression_gd.cpp`: binary logistic regression trained with gradient descent.
- `chapter_2/mean_substitution_example.cpp`: replace missing values with the column mean.
- `chapter_2/minmax_scale_demo.cpp`: min-max scaling to a fixed range.
- `chapter_2/one_hot_example.cpp`: one-hot encoding for categorical values.
- `chapter_2/ordinal_encode_example.cpp`: ordinal encoding with a label-to-rank mapping.
- `chapter_2/pad_truncate_demo.cpp`: pad or truncate sequences to a fixed length.
- `chapter_2/polynomial_features_demo.cpp`: add polynomial powers of input features.
- `chapter_2/power_transform_demo.cpp`: power transforms for numeric stabilization.
- `chapter_2/regression_imputation_example.cpp`: fill missing targets with a linear regression equation.
- `chapter_2/robust_scale_demo.cpp`: robust scaling using median and IQR-style statistics.
- `chapter_2/rolling_mean_demo.cpp`: rolling-window mean for smoothing.
- `chapter_1/simd_demo.cpp`: AVX2 or NEON vector add with correctness and throughput reporting.
- `chapter_2/time_feature_demo.cpp`: derive basic calendar and cyclical hour features from timestamps.
- `chapter_2/tsne_demo.cpp`: simple t-SNE style dimensionality-reduction walkthrough.
- `chapter_2/zscore_normalize_demo.cpp`: z-score normalization.

## Armadillo, Eigen, And mlpack

- `chapter_2/armadillo_scaling_demo.cpp`: scaling, robust statistics, and log or power transforms using Armadillo matrices.
- `chapter_2/glove_eigen_demo.cpp`: load tiny GloVe-style embeddings, embed tokens, and pool a sentence vector with Eigen.
- `chapter_2/mlpack_encoding_demo_min.cpp`: one-hot and frequency encoding with Armadillo-backed mlpack types.
- `chapter_2/mlpack_impute_demo.cpp`: simple mean imputation and forward fill, then save processed data to CSV.
- `chapter_2/pca_demo.cpp`: PCA with Eigen covariance and eigendecomposition.
- `chapter_2/pca_mlpack4_demo.cpp`: PCA with mlpack 4 and Armadillo.
- `chapter_2/poly_interact_armadillo_demo.cpp`: polynomial and interaction terms with Armadillo.
- `chapter_2/ts_preproc_armadillo_opencv_demo.cpp`: time-series rolling mean, smoothing, differencing, and FFT magnitude using Armadillo plus OpenCV.

## NLP And Text Preprocessing

- `chapter_2/tokenize_stem_demo.cpp`: tokenize text, remove stop words, stem with Snowball, and apply a toy lemmatizer.
- `chapter_2/token_dropout_demo.cpp`: small runnable token-dropout augmentation example.

The following files are also relevant to text or categorical preprocessing even though they do not depend on NLP-specific libraries:

- `chapter_2/binary_encoding_demo.cpp`
- `chapter_2/embedding_encode_demo.cpp`
- `chapter_2/frequency_encode_example.cpp`
- `chapter_2/glove_eigen_demo.cpp`
- `chapter_2/one_hot_example.cpp`
- `chapter_2/ordinal_encode_example.cpp`
- `chapter_2/pad_truncate_demo.cpp`

## Image, Audio, And Multimodal Data

- `chapter_2/image_preproc_demo.cpp`: resize, crop, flip, rotate, perspective warp, equalize, adjust contrast, and detect edges with OpenCV.
- `chapter_2/multimodal_preproc_demo.cpp`: image normalization and noise injection, token dropout, and numeric z-score or jitter in one walkthrough.
- `chapter_2/spectrogram_demo.cpp`: load audio with libsndfile, compute an STFT with FFTW, and write a colorized spectrogram with OpenCV.

Useful sample assets already in the repo:

- `input.jpg`: sample input for image demos.
- `input.wav`: sample input for the spectrogram demo.
- `out_*.jpg`, `normalized.png`, `noisy.png`, `spectrogram.png`: example outputs generated by the demos.

## Deep Learning With LibTorch And CUDA

- `chapter_1/example.cpp`: basic tensor creation, arithmetic, and activation functions.
- `chapter_3/vector_add_cpu.cpp`: CPU baseline for the chapter 3 array-add walkthrough.
- `chapter_3/add.cu`: first CUDA port of vector addition using Unified Memory and a single-thread launch.
- `chapter_3/add_block.cu`: single-block, multi-thread CUDA vector addition.
- `chapter_3/add_grid.cu`: multi-block grid-stride CUDA vector addition.
- `chapter_3/cuda_error_check_demo.cu`: CUDA runtime, launch, and synchronization error-checking demo.
- `chapter_4/neuron_demo.cpp`: single linear layer plus ReLU on CPU or CUDA if available.
- `chapter_4/mlp_eigen_demo.cpp`: from-scratch `2 -> 3 -> 1` Eigen MLP with sigmoid activations and SGD training.
- `chapter_4/neural_network.h`: declaration for the from-scratch Eigen MLP.
- `chapter_4/neural_network.cpp`: forward and backpropagation implementation for the from-scratch Eigen MLP.
- `chapter_5/eigen_mlp_xor.cpp`: chapter 5 Eigen MLP for XOR imported from the official Packt code bundle.
- `chapter_5/libtorch_mlp.cpp`: chapter 5 LibTorch MLP for XOR imported from the official Packt code bundle.
- `chapter_5/optimizers.cpp`: chapter 5 optimizer comparison covering SGD, momentum, RMSprop, Adam, AdaGrad, and AdaDelta.
- `chapter_5/cuda_mlp.cu`: chapter 5 CUDA MLP walkthrough using CUDA, cuBLAS, and cuRAND.
- `chapter_10/tinynet.h`: TinyNet header used across the chapter 10 deployment and inference examples.
- `chapter_10/trace_and_save_ts.cpp`: export TinyNet to a TorchScript artifact directly from C++.
- `chapter_10/infer_torchscript.cpp`: load a TorchScript model and run warm-up plus inference.
- `chapter_10/parity_check.cpp`: compare native TinyNet output against the traced TorchScript artifact.
- `chapter_10/micro_batcher_demo.cpp`: bounded micro-batching scheduler using LibTorch futures, promises, and one inference worker.
- `chapter_10/fp16_torchscript_demo.cpp`: FP16 TorchScript inference path for CUDA deployments.
- `chapter_10/pruned_torchscript_demo.cpp`: load a structurally pruned TorchScript artifact and run inference.
- `chapter_10/benchmark_student_vs_baseline.cpp`: throughput harness for teacher-vs-student TorchScript comparisons.
- `chapter_10/cuda_graph_capture_demo.cpp`: CUDA Graph capture and replay for steady fixed-shape inference.
- `chapter_1/mnist_fc.cpp`: minimal two-layer fully connected network with MNIST-like tensor shapes.
- `chapter_1/welcome_from_cuda.cu`: CUDA hello-kernel used in chapter 1 to verify GPU execution.
- `chapter_2/autoencoder_demo.cpp`: train an autoencoder on synthetic low-rank data and print the latent representation.
- `chapter_2/autoencoder_libtorch_demo.cpp`: shorter LibTorch autoencoder training example.
- `chapter_2/data_pipeline/custom_dataset_demo.cpp`: custom dataset, dataloader, batching, and a tiny supervised training loop.

Repo support files for this group:

- `libtorch/`: local LibTorch download target; see the root `README.md` setup section.
- `libtorch-macos-arm64-2.8.0.zip`: archived LibTorch download used by the repo.

## ONNX Runtime And Model Assets

- `chapter_2/bert_onnx_demo.cpp`: inspect model inputs and outputs and run a BERT ONNX model on token IDs.
- `chapter_2/nlp_advanced_demo.cpp`: run NER or sentiment ONNX models with dynamic input handling.
- `chapter_10/onnx_loader_demo.cpp`: chapter 10 ONNX Runtime loader for a deployment-format TinyNet model.
- `chapter_10/onnx_int8_demo.cpp`: chapter 10 quantized ONNX Runtime loader for INT8 graphs.
- `chapter_10/onnx_session_options_demo.cpp`: small runnable snippet showing common ONNX Runtime session options.

## Monitoring, Observability, And Online Quality

- `chapter_11/fixed_bin_histogram_psi.cpp`: fixed-bin histogram plus Population Stability Index drift calculation.
- `chapter_11/observability_utils.h`: shared logging and tracing helpers for the chapter 11 debugging demos.
- `chapter_11/structured_logging_demo.cpp`: chapter 11 structured JSON-line logging example.
- `chapter_11/metrics_tracing_demo.cpp`: chapter 11 in-process metrics and scoped tracing example.
- `chapter_11/repro_pack_and_sentinels_demo.cpp`: chapter 11 repro pack, seeded replay, NaN or Inf sentinel, and stable log-sum-exp demo.
- `chapter_11/micro_batcher_debug_demo.cpp`: chapter 11 queue-time vs compute-time batching demo using standard-library placeholders.
- `chapter_11/cuda_check_macro_demo.cu`: chapter 11 CUDA error-check macro example wrapped in a small runnable CUDA program.
- `chapter_12/p95_demo.cpp`: chapter 12 percentile helper for rolling latency windows.
- `chapter_12/metrics.hpp`: chapter 12 counters, gauges, histograms, labels, and registry primitives.
- `chapter_12/metrics_server.cpp`: chapter 12 tiny `/metrics` exporter for Prometheus-style scraping.
- `chapter_12/log.hpp`: chapter 12 structured JSON logging helper.
- `chapter_12/trace.hpp`: chapter 12 RAII span helper that logs and records span latency histograms.
- `chapter_12/quality.hpp`: chapter 12 delayed-label joiner, calibration summaries, disagreement counter, and confidence proxies.
- `chapter_12/cohort_quality.hpp`: chapter 12 cohort-level moving summaries for leading indicators.
- `chapter_12/observability_demo.cpp`: runnable observability demo that ties together logs, spans, and metrics.
- `chapter_12/quality_monitor_demo.cpp`: runnable delayed-label and leading-indicator monitoring demo.

## Explainability And Responsible AI

- `chapter_13/explainability.hpp`: reusable LIME and KernelSHAP support code for model-agnostic explanations.
- `chapter_13/lime_shap_demo.cpp`: runnable tabular explainability demo covering local surrogates and additive feature attributions.
- `chapter_13/gradcam_demo.cpp`: LibTorch plus OpenCV Grad-CAM demo that captures the last convolutional activation and renders an overlay.

Model directories:

- `bert_onnx/`: local download target for the BERT tokenizer files and ONNX variants.
- `ner_onnx/`: local export target for the token-classification model assets.
- `sent_onnx/`: local export target for the sentiment-classification model assets.

## CMake Subprojects

- `chapter_2/data_pipeline/CMakeLists.txt`: build `custom_dataset_demo.cpp` against LibTorch with CMake.
- `data_pipeline_caffe2/CMakeLists.txt`: build `caffe2_normalize_demo.cpp` against Torch or Caffe2 with CMake.
- `data_pipeline_caffe2/caffe2_normalize_demo.cpp`: normalization example using Caffe2 operators and broadcasting.

## Suggested Starting Points

- If you want standard-library-only code, start with `chapter_2/rolling_mean_demo.cpp`, `chapter_2/zscore_normalize_demo.cpp`, `chapter_2/fft_demo.cpp`, and `chapter_2/bigdata_demo.cpp`.
- If you want matrix libraries, start with `chapter_2/armadillo_scaling_demo.cpp`, `chapter_2/pca_demo.cpp`, and `chapter_2/pca_mlpack4_demo.cpp`.
- If you want CV or signal processing, start with `chapter_2/image_preproc_demo.cpp` and `chapter_2/spectrogram_demo.cpp`.
- If you want neural nets, start with `chapter_1/example.cpp`, `chapter_4/neuron_demo.cpp`, `chapter_4/mlp_eigen_demo.cpp`, and `chapter_2/autoencoder_demo.cpp`.
- If you want inference deployment patterns, start with `chapter_10/trace_and_save_ts.cpp`, `chapter_10/infer_torchscript.cpp`, `chapter_10/parity_check.cpp`, and `chapter_10/micro_batcher_demo.cpp`.
- If you want drift monitoring and debugging patterns, start with `chapter_11/fixed_bin_histogram_psi.cpp`, `chapter_11/structured_logging_demo.cpp`, `chapter_11/metrics_tracing_demo.cpp`, and `chapter_11/micro_batcher_debug_demo.cpp`.
- If you want service monitoring and online-quality instrumentation, start with `chapter_12/p95_demo.cpp`, `chapter_12/observability_demo.cpp`, `chapter_12/metrics_server.cpp`, and `chapter_12/quality_monitor_demo.cpp`.
- If you want explainability, start with `chapter_13/lime_shap_demo.cpp`, then move to `chapter_13/gradcam_demo.cpp`.
- If you want CUDA kernels, start with `chapter_3/vector_add_cpu.cpp`, then `chapter_3/add.cu`, then `chapter_3/add_grid.cu`.
- If you want model inference, start with `chapter_2/bert_onnx_demo.cpp` and `chapter_2/nlp_advanced_demo.cpp`.
