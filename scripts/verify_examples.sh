#!/usr/bin/env bash

set -u
set -o pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_ROOT="${BUILD_ROOT:-$(mktemp -d /tmp/cpp_book_verify.XXXXXX)}"
BIN_DIR="$BUILD_ROOT/bin"
LOG_DIR="$BUILD_ROOT/logs"
RUN_DIR="$BUILD_ROOT/run"

mkdir -p "$BIN_DIR" "$LOG_DIR" "$RUN_DIR"

PASS=0
FAIL=0
SKIP=0
TOTAL=0

ONNX_PREFIX="$(brew --prefix onnxruntime 2>/dev/null || true)"
PROTOBUF_PREFIX="$(brew --prefix protobuf 2>/dev/null || true)"
EIGEN_PREFIX="$(brew --prefix eigen 2>/dev/null || true)"
CEREAL_PREFIX="$(brew --prefix cereal 2>/dev/null || true)"
BREW_PREFIX="$(brew --prefix 2>/dev/null || true)"
HAS_LIBTORCH=0
HAS_BERT_ONNX=0
HAS_NER_ONNX=0
HAS_SENT_ONNX=0

if [[ -f "$REPO_ROOT/libtorch/include/torch/torch.h" && -d "$REPO_ROOT/libtorch/lib" ]]; then
  HAS_LIBTORCH=1
fi
if [[ -f "$REPO_ROOT/bert_onnx/onnx/model.onnx" ]]; then
  HAS_BERT_ONNX=1
fi
if [[ -f "$REPO_ROOT/ner_onnx/model.onnx" ]]; then
  HAS_NER_ONNX=1
fi
if [[ -f "$REPO_ROOT/sent_onnx/model.onnx" ]]; then
  HAS_SENT_ONNX=1
fi

DYLD_PATHS=""
if [[ $HAS_LIBTORCH -eq 1 ]]; then
  DYLD_PATHS="$REPO_ROOT/libtorch/lib"
fi
if [[ -n "$ONNX_PREFIX" ]]; then
  DYLD_PATHS="${DYLD_PATHS:+$ONNX_PREFIX/lib:}$DYLD_PATHS"
  if [[ -z "$DYLD_PATHS" ]]; then
    DYLD_PATHS="$ONNX_PREFIX/lib"
  fi
fi
if [[ -n "$PROTOBUF_PREFIX" ]]; then
  DYLD_PATHS="${DYLD_PATHS:+$PROTOBUF_PREFIX/lib:}$DYLD_PATHS"
  if [[ -z "$DYLD_PATHS" ]]; then
    DYLD_PATHS="$PROTOBUF_PREFIX/lib"
  fi
fi
if [[ -n "$BREW_PREFIX" ]]; then
  DYLD_PATHS="${DYLD_PATHS:+$BREW_PREFIX/lib:}$DYLD_PATHS"
  if [[ -z "$DYLD_PATHS" ]]; then
    DYLD_PATHS="$BREW_PREFIX/lib"
  fi
fi

SHIM_DIR="$BUILD_ROOT/dylib_shims"
mkdir -p "$SHIM_DIR"
if [[ -n "$PROTOBUF_PREFIX" ]]; then
  PROTOBUF_REAL="$(ls "$PROTOBUF_PREFIX/lib"/libprotobuf.32.*.dylib 2>/dev/null | sort | tail -1 || true)"
  PROTOBUF_LITE_REAL="$(ls "$PROTOBUF_PREFIX/lib"/libprotobuf-lite.32.*.dylib 2>/dev/null | sort | tail -1 || true)"
  if [[ -n "$PROTOBUF_REAL" ]]; then
    ln -sf "$PROTOBUF_REAL" "$SHIM_DIR/libprotobuf.32.0.0.dylib"
  fi
  if [[ -n "$PROTOBUF_LITE_REAL" ]]; then
    ln -sf "$PROTOBUF_LITE_REAL" "$SHIM_DIR/libprotobuf-lite.32.0.0.dylib"
  fi
fi
DYLD_PATHS="${DYLD_PATHS:+$SHIM_DIR:}$DYLD_PATHS"
if [[ -z "$DYLD_PATHS" ]]; then
  DYLD_PATHS="$SHIM_DIR"
fi

RUN_ENV="env DYLD_LIBRARY_PATH=\"$DYLD_PATHS${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}\" DYLD_FALLBACK_LIBRARY_PATH=\"$DYLD_PATHS${DYLD_FALLBACK_LIBRARY_PATH:+:$DYLD_FALLBACK_LIBRARY_PATH}\""
LIBTORCH_FLAGS="-I\"$REPO_ROOT/libtorch/include\" -I\"$REPO_ROOT/libtorch/include/torch/csrc/api/include\" -L\"$REPO_ROOT/libtorch/lib\" -Wl,-rpath,\"$REPO_ROOT/libtorch/lib\" -ltorch -ltorch_cpu -lc10"
ONNX_FLAGS="-I\"$ONNX_PREFIX/include\" -L\"$ONNX_PREFIX/lib\" -Wl,-rpath,\"$ONNX_PREFIX/lib\" -lonnxruntime"
EIGEN_FLAGS="-I\"$EIGEN_PREFIX/include/eigen3\""
CEREAL_FLAGS="-I\"$CEREAL_PREFIX/include\""
STEMMER_FLAGS="-I\"$BREW_PREFIX/include\" -L\"$BREW_PREFIX/lib\" -lstemmer"

note() {
  printf '%s\n' "$*"
}

run_case() {
  local name="$1"
  local build_cmd="$2"
  local run_cmd="$3"
  local log="$LOG_DIR/${name//\//_}.log"

  TOTAL=$((TOTAL + 1))
  note "CASE  $name"

  {
    printf 'build: %s\n' "$build_cmd"
    eval "$build_cmd"
    printf 'run: %s\n' "$run_cmd"
    (
      cd "$RUN_DIR"
      eval "$run_cmd"
    )
  } >"$log" 2>&1
  local status=$?

  if [[ $status -eq 0 ]]; then
    PASS=$((PASS + 1))
    note "PASS  $name"
  else
    FAIL=$((FAIL + 1))
    note "FAIL  $name"
    tail -n 20 "$log" || true
  fi
}

build_only_case() {
  local name="$1"
  local build_cmd="$2"
  local reason="$3"
  local log="$LOG_DIR/${name//\//_}.log"

  TOTAL=$((TOTAL + 1))
  note "CASE  $name"

  {
    printf 'build: %s\n' "$build_cmd"
    eval "$build_cmd"
  } >"$log" 2>&1
  local status=$?

  if [[ $status -eq 0 ]]; then
    SKIP=$((SKIP + 1))
    note "SKIP  $name ($reason)"
  else
    FAIL=$((FAIL + 1))
    note "FAIL  $name"
    tail -n 20 "$log" || true
  fi
}

skip_case() {
  local name="$1"
  local reason="$2"
  TOTAL=$((TOTAL + 1))
  SKIP=$((SKIP + 1))
  note "SKIP  $name ($reason)"
}

run_metrics_server_case() {
  local name="chapter_12/metrics_server.cpp"
  local bin="$BIN_DIR/metrics_server"
  local log="$LOG_DIR/${name//\//_}.log"
  local port

  port="$(python3 - <<'PY'
import socket
s = socket.socket()
s.bind(("127.0.0.1", 0))
print(s.getsockname()[1])
s.close()
PY
)"

  TOTAL=$((TOTAL + 1))
  note "CASE  $name"

  {
    printf 'build: %s\n' "c++ -std=c++17 -O2 -pthread \"$REPO_ROOT/chapter_12/metrics_server.cpp\" -o \"$bin\""
    c++ -std=c++17 -O2 -pthread "$REPO_ROOT/chapter_12/metrics_server.cpp" -o "$bin"
    printf 'run: %s\n' "$RUN_ENV \"$bin\" $port 3"
    (
      cd "$RUN_DIR"
      eval "$RUN_ENV \"$bin\" $port 3" >metrics_server.out 2>&1 &
      server_pid=$!
      sleep 1
      curl -fsS "http://127.0.0.1:$port/metrics" >metrics_scrape.txt
      wait "$server_pid"
      grep -q "inference_requests_total" metrics_scrape.txt
    )
  } >"$log" 2>&1
  local status=$?

  if [[ $status -eq 0 ]]; then
    PASS=$((PASS + 1))
    note "PASS  $name"
  else
    FAIL=$((FAIL + 1))
    note "FAIL  $name"
    tail -n 20 "$log" || true
  fi
}

note "Build root: $BUILD_ROOT"

if [[ $HAS_LIBTORCH -eq 1 ]]; then
  run_case "chapter_1/example.cpp" \
    "c++ -std=c++17 \"$REPO_ROOT/chapter_1/example.cpp\" -o \"$BIN_DIR/ch1_example\" $LIBTORCH_FLAGS" \
    "$RUN_ENV \"$BIN_DIR/ch1_example\""

  run_case "chapter_1/mnist_fc.cpp" \
    "c++ -std=c++17 \"$REPO_ROOT/chapter_1/mnist_fc.cpp\" -o \"$BIN_DIR/ch1_mnist_fc\" $LIBTORCH_FLAGS" \
    "$RUN_ENV \"$BIN_DIR/ch1_mnist_fc\""
else
  skip_case "chapter_1/example.cpp" "libtorch/ is not present; see README.md"
  skip_case "chapter_1/mnist_fc.cpp" "libtorch/ is not present; see README.md"
fi

run_case "chapter_1/simd_demo.cpp" \
  "c++ -std=c++17 -O3 \"$REPO_ROOT/chapter_1/simd_demo.cpp\" -o \"$BIN_DIR/ch1_simd\"" \
  "\"$BIN_DIR/ch1_simd\" 1048576 5"

skip_case "chapter_1/welcome_from_cuda.cu" "nvcc is not installed in this environment"

chapter2_std=(
  backward_fill_example
  binary_encoding_demo
  differencing_demo
  embedding_encode_demo
  exponential_smoothing_demo
  fft_demo
  forward_fill_example
  frequency_encode_example
  interaction_terms_demo
  knn_imputation_example
  knn_imputation_multidim_example
  log_transform_demo
  mean_substitution_example
  minmax_scale_demo
  one_hot_example
  ordinal_encode_example
  pad_truncate_demo
  polynomial_features_demo
  power_transform_demo
  regression_imputation_example
  robust_scale_demo
  rolling_mean_demo
  time_feature_demo
  token_dropout_demo
  tsne_demo
  zscore_normalize_demo
)

for base in "${chapter2_std[@]}"; do
  run_case "chapter_2/${base}.cpp" \
    "c++ -std=c++17 -O2 \"$REPO_ROOT/chapter_2/${base}.cpp\" -o \"$BIN_DIR/${base}\"" \
    "\"$BIN_DIR/${base}\""
done

run_case "chapter_2/bigdata_demo.cpp" \
  "c++ -std=c++17 -O2 \"$REPO_ROOT/chapter_2/bigdata_demo.cpp\" -o \"$BIN_DIR/bigdata_demo\"" \
  "\"$BIN_DIR/bigdata_demo\" make-sample sample.bin 4194304 && \"$BIN_DIR/bigdata_demo\" mmap sample.bin && \"$BIN_DIR/bigdata_demo\" batch sample.bin 262144 && \"$BIN_DIR/bigdata_demo\" range sample.bin 524288"

run_case "chapter_2/pca_demo.cpp" \
  "c++ -std=c++17 -O2 \"$REPO_ROOT/chapter_2/pca_demo.cpp\" -o \"$BIN_DIR/pca_demo\" $EIGEN_FLAGS" \
  "\"$BIN_DIR/pca_demo\""

run_case "chapter_2/glove_eigen_demo.cpp" \
  "c++ -std=c++17 -O2 \"$REPO_ROOT/chapter_2/glove_eigen_demo.cpp\" -o \"$BIN_DIR/glove_eigen_demo\" $EIGEN_FLAGS" \
  "\"$BIN_DIR/glove_eigen_demo\""

for base in armadillo_scaling_demo poly_interact_armadillo_demo; do
  run_case "chapter_2/${base}.cpp" \
    "c++ -std=c++17 -O2 \"$REPO_ROOT/chapter_2/${base}.cpp\" -o \"$BIN_DIR/${base}\" \$(pkg-config --cflags --libs armadillo)" \
    "\"$BIN_DIR/${base}\""
done

run_case "chapter_2/mlpack_encoding_demo_min.cpp" \
  "c++ -std=c++17 -O2 \"$REPO_ROOT/chapter_2/mlpack_encoding_demo_min.cpp\" -o \"$BIN_DIR/mlpack_encoding_demo_min\" \$(pkg-config --cflags --libs mlpack armadillo)" \
  "\"$BIN_DIR/mlpack_encoding_demo_min\""

run_case "chapter_2/mlpack_impute_demo.cpp" \
  "c++ -std=c++17 -O2 \"$REPO_ROOT/chapter_2/mlpack_impute_demo.cpp\" -o \"$BIN_DIR/mlpack_impute_demo\" \$(pkg-config --cflags --libs mlpack armadillo) $CEREAL_FLAGS -Wno-deprecated-declarations" \
  "\"$BIN_DIR/mlpack_impute_demo\""

run_case "chapter_2/pca_mlpack4_demo.cpp" \
  "c++ -std=c++17 -O2 \"$REPO_ROOT/chapter_2/pca_mlpack4_demo.cpp\" -o \"$BIN_DIR/pca_mlpack4_demo\" \$(pkg-config --cflags --libs mlpack armadillo)" \
  "\"$BIN_DIR/pca_mlpack4_demo\""

run_case "chapter_2/ts_preproc_armadillo_opencv_demo.cpp" \
  "c++ -std=c++17 -O2 \"$REPO_ROOT/chapter_2/ts_preproc_armadillo_opencv_demo.cpp\" -o \"$BIN_DIR/ts_preproc_armadillo_opencv_demo\" \$(pkg-config --cflags --libs armadillo opencv4)" \
  "\"$BIN_DIR/ts_preproc_armadillo_opencv_demo\""

run_case "chapter_2/image_preproc_demo.cpp" \
  "c++ -std=c++17 -O2 \"$REPO_ROOT/chapter_2/image_preproc_demo.cpp\" -o \"$BIN_DIR/image_preproc_demo\" \$(pkg-config --cflags --libs opencv4)" \
  "\"$BIN_DIR/image_preproc_demo\" \"$REPO_ROOT/input.jpg\""

run_case "chapter_2/multimodal_preproc_demo.cpp" \
  "c++ -std=c++17 -O2 \"$REPO_ROOT/chapter_2/multimodal_preproc_demo.cpp\" -o \"$BIN_DIR/multimodal_preproc_demo\" \$(pkg-config --cflags --libs opencv4)" \
  "\"$BIN_DIR/multimodal_preproc_demo\" \"$REPO_ROOT/input.jpg\""

run_case "chapter_2/spectrogram_demo.cpp" \
  "c++ -std=c++17 -O2 \"$REPO_ROOT/chapter_2/spectrogram_demo.cpp\" -o \"$BIN_DIR/spectrogram_demo\" \$(pkg-config --cflags --libs fftw3 sndfile opencv4)" \
  "\"$BIN_DIR/spectrogram_demo\" \"$REPO_ROOT/input.wav\""

run_case "chapter_2/tokenize_stem_demo.cpp" \
  "c++ -std=c++17 -O2 \"$REPO_ROOT/chapter_2/tokenize_stem_demo.cpp\" -o \"$BIN_DIR/tokenize_stem_demo\" $STEMMER_FLAGS" \
  "\"$BIN_DIR/tokenize_stem_demo\""

if [[ -n "$ONNX_PREFIX" ]]; then
  if [[ $HAS_BERT_ONNX -eq 1 ]]; then
    run_case "chapter_2/bert_onnx_demo.cpp" \
      "c++ -std=c++17 -O2 \"$REPO_ROOT/chapter_2/bert_onnx_demo.cpp\" -o \"$BIN_DIR/bert_onnx_demo\" $ONNX_FLAGS" \
      "$RUN_ENV \"$BIN_DIR/bert_onnx_demo\" \"$REPO_ROOT/bert_onnx/onnx/model.onnx\""
  else
    skip_case "chapter_2/bert_onnx_demo.cpp" "bert_onnx/onnx/model.onnx is not present; see README.md"
  fi

  if [[ $HAS_NER_ONNX -eq 1 && $HAS_SENT_ONNX -eq 1 ]]; then
    run_case "chapter_2/nlp_advanced_demo.cpp" \
      "c++ -std=c++17 -O2 \"$REPO_ROOT/chapter_2/nlp_advanced_demo.cpp\" -o \"$BIN_DIR/nlp_advanced_demo\" $ONNX_FLAGS" \
      "$RUN_ENV \"$BIN_DIR/nlp_advanced_demo\" ner \"$REPO_ROOT/ner_onnx/model.onnx\" && $RUN_ENV \"$BIN_DIR/nlp_advanced_demo\" sentiment \"$REPO_ROOT/sent_onnx/model.onnx\""
  else
    skip_case "chapter_2/nlp_advanced_demo.cpp" "ner_onnx/ or sent_onnx/ assets are not present; see README.md"
  fi
else
  skip_case "chapter_2/bert_onnx_demo.cpp" "onnxruntime is not installed"
  skip_case "chapter_2/nlp_advanced_demo.cpp" "onnxruntime is not installed"
fi

if [[ $HAS_LIBTORCH -eq 1 ]]; then
  for base in autoencoder_demo autoencoder_libtorch_demo; do
    run_case "chapter_2/${base}.cpp" \
      "c++ -std=c++17 -O2 \"$REPO_ROOT/chapter_2/${base}.cpp\" -o \"$BIN_DIR/${base}\" $LIBTORCH_FLAGS" \
      "$RUN_ENV \"$BIN_DIR/${base}\""
  done

  run_case "chapter_2/data_pipeline/custom_dataset_demo.cpp" \
    "cmake -S \"$REPO_ROOT/chapter_2/data_pipeline\" -B \"$BUILD_ROOT/ch2_data_pipeline\" -DCMAKE_PREFIX_PATH=\"$REPO_ROOT/libtorch\" >/dev/null && cmake --build \"$BUILD_ROOT/ch2_data_pipeline\" -j2 >/dev/null" \
    "$RUN_ENV \"$BUILD_ROOT/ch2_data_pipeline/custom_dataset_demo\""

  run_case "data_pipeline_caffe2/caffe2_normalize_demo.cpp" \
    "cmake -S \"$REPO_ROOT/data_pipeline_caffe2\" -B \"$BUILD_ROOT/data_pipeline_caffe2\" -DTorch_DIR=\"$REPO_ROOT/libtorch/share/cmake/Torch\" >/dev/null && cmake --build \"$BUILD_ROOT/data_pipeline_caffe2\" -j2 >/dev/null" \
    "$RUN_ENV \"$BUILD_ROOT/data_pipeline_caffe2/caffe2_normalize_demo\""
else
  skip_case "chapter_2/autoencoder_demo.cpp" "libtorch/ is not present; see README.md"
  skip_case "chapter_2/autoencoder_libtorch_demo.cpp" "libtorch/ is not present; see README.md"
  skip_case "chapter_2/data_pipeline/custom_dataset_demo.cpp" "libtorch/ is not present; see README.md"
  skip_case "data_pipeline_caffe2/caffe2_normalize_demo.cpp" "libtorch/ is not present; see README.md"
fi

run_case "chapter_3/vector_add_cpu.cpp" \
  "c++ -std=c++17 -O2 \"$REPO_ROOT/chapter_3/vector_add_cpu.cpp\" -o \"$BIN_DIR/vector_add_cpu\"" \
  "\"$BIN_DIR/vector_add_cpu\""

skip_case "chapter_3/add.cu" "nvcc is not installed in this environment"
skip_case "chapter_3/add_block.cu" "nvcc is not installed in this environment"
skip_case "chapter_3/add_grid.cu" "nvcc is not installed in this environment"
skip_case "chapter_3/cuda_error_check_demo.cu" "nvcc is not installed in this environment"
skip_case "chapter_3/some-CUDA.cu" "nvcc is not installed in this environment"

for base in linear_regression_gd logistic_regression_gd; do
  run_case "chapter_4/${base}.cpp" \
    "c++ -std=c++17 -O2 \"$REPO_ROOT/chapter_4/${base}.cpp\" -o \"$BIN_DIR/${base}\"" \
    "\"$BIN_DIR/${base}\""
done

run_case "chapter_4/mlp_eigen_demo.cpp" \
  "c++ -std=c++17 -O2 \"$REPO_ROOT/chapter_4/mlp_eigen_demo.cpp\" \"$REPO_ROOT/chapter_4/neural_network.cpp\" -o \"$BIN_DIR/mlp_eigen_demo\" $EIGEN_FLAGS" \
  "\"$BIN_DIR/mlp_eigen_demo\""

if [[ $HAS_LIBTORCH -eq 1 ]]; then
  run_case "chapter_4/neuron_demo.cpp" \
    "c++ -std=c++17 -O2 \"$REPO_ROOT/chapter_4/neuron_demo.cpp\" -o \"$BIN_DIR/neuron_demo\" $LIBTORCH_FLAGS" \
    "$RUN_ENV \"$BIN_DIR/neuron_demo\""
else
  skip_case "chapter_4/neuron_demo.cpp" "libtorch/ is not present; see README.md"
fi

run_case "chapter_5/eigen_mlp_xor.cpp" \
  "c++ -std=c++17 -O2 \"$REPO_ROOT/chapter_5/eigen_mlp_xor.cpp\" -o \"$BIN_DIR/eigen_mlp_xor\" $EIGEN_FLAGS" \
  "\"$BIN_DIR/eigen_mlp_xor\""

if [[ $HAS_LIBTORCH -eq 1 ]]; then
  run_case "chapter_5/libtorch_mlp.cpp" \
    "c++ -std=c++17 -O2 \"$REPO_ROOT/chapter_5/libtorch_mlp.cpp\" -o \"$BIN_DIR/libtorch_mlp\" $LIBTORCH_FLAGS" \
    "$RUN_ENV \"$BIN_DIR/libtorch_mlp\""

  run_case "chapter_5/optimizers.cpp" \
    "c++ -std=c++17 -O2 \"$REPO_ROOT/chapter_5/optimizers.cpp\" -o \"$BIN_DIR/optimizers\" $EIGEN_FLAGS $LIBTORCH_FLAGS" \
    "$RUN_ENV \"$BIN_DIR/optimizers\""
else
  skip_case "chapter_5/libtorch_mlp.cpp" "libtorch/ is not present; see README.md"
  skip_case "chapter_5/optimizers.cpp" "libtorch/ is not present; see README.md"
fi

skip_case "chapter_5/cuda_mlp.cu" "nvcc is not installed in this environment"

TINYSCRIPT_PATH="$BUILD_ROOT/tinynet.ts"
TINYSCRIPT_SLIM_PATH="$BUILD_ROOT/tinynet_slim.ts"

if [[ $HAS_LIBTORCH -eq 1 ]]; then
  run_case "chapter_10/trace_and_save_ts.cpp" \
    "c++ -std=c++17 -O2 \"$REPO_ROOT/chapter_10/trace_and_save_ts.cpp\" -o \"$BIN_DIR/trace_and_save_ts\" $LIBTORCH_FLAGS" \
    "$RUN_ENV \"$BIN_DIR/trace_and_save_ts\" \"$TINYSCRIPT_PATH\""

  if [[ -f "$TINYSCRIPT_PATH" ]]; then
    cp "$TINYSCRIPT_PATH" "$TINYSCRIPT_SLIM_PATH"

    run_case "chapter_10/infer_torchscript.cpp" \
      "c++ -std=c++17 -O2 \"$REPO_ROOT/chapter_10/infer_torchscript.cpp\" -o \"$BIN_DIR/infer_torchscript\" $LIBTORCH_FLAGS" \
      "$RUN_ENV \"$BIN_DIR/infer_torchscript\" \"$TINYSCRIPT_PATH\""

    run_case "chapter_10/parity_check.cpp" \
      "c++ -std=c++17 -O2 \"$REPO_ROOT/chapter_10/parity_check.cpp\" -o \"$BIN_DIR/parity_check\" $LIBTORCH_FLAGS" \
      "$RUN_ENV \"$BIN_DIR/parity_check\" \"$TINYSCRIPT_PATH\""

    run_case "chapter_10/micro_batcher_demo.cpp" \
      "c++ -std=c++17 -O2 \"$REPO_ROOT/chapter_10/micro_batcher_demo.cpp\" -o \"$BIN_DIR/micro_batcher_demo\" $LIBTORCH_FLAGS" \
      "$RUN_ENV \"$BIN_DIR/micro_batcher_demo\" \"$TINYSCRIPT_PATH\""

    run_case "chapter_10/pruned_torchscript_demo.cpp" \
      "c++ -std=c++17 -O2 \"$REPO_ROOT/chapter_10/pruned_torchscript_demo.cpp\" -o \"$BIN_DIR/pruned_torchscript_demo\" $LIBTORCH_FLAGS" \
      "$RUN_ENV \"$BIN_DIR/pruned_torchscript_demo\" \"$TINYSCRIPT_SLIM_PATH\""

    run_case "chapter_10/benchmark_student_vs_baseline.cpp" \
      "c++ -std=c++17 -O2 \"$REPO_ROOT/chapter_10/benchmark_student_vs_baseline.cpp\" -o \"$BIN_DIR/benchmark_student_vs_baseline\" $LIBTORCH_FLAGS" \
      "$RUN_ENV \"$BIN_DIR/benchmark_student_vs_baseline\" \"$TINYSCRIPT_PATH\" \"$TINYSCRIPT_PATH\" 4 2"
  else
    skip_case "chapter_10/infer_torchscript.cpp" "TorchScript artifact generation failed"
    skip_case "chapter_10/parity_check.cpp" "TorchScript artifact generation failed"
    skip_case "chapter_10/micro_batcher_demo.cpp" "TorchScript artifact generation failed"
    skip_case "chapter_10/pruned_torchscript_demo.cpp" "TorchScript artifact generation failed"
    skip_case "chapter_10/benchmark_student_vs_baseline.cpp" "TorchScript artifact generation failed"
  fi
else
  skip_case "chapter_10/trace_and_save_ts.cpp" "libtorch/ is not present; see README.md"
  skip_case "chapter_10/infer_torchscript.cpp" "libtorch/ is not present; see README.md"
  skip_case "chapter_10/parity_check.cpp" "libtorch/ is not present; see README.md"
  skip_case "chapter_10/micro_batcher_demo.cpp" "libtorch/ is not present; see README.md"
  skip_case "chapter_10/pruned_torchscript_demo.cpp" "libtorch/ is not present; see README.md"
  skip_case "chapter_10/benchmark_student_vs_baseline.cpp" "libtorch/ is not present; see README.md"
fi

if [[ -n "$ONNX_PREFIX" ]]; then
  run_case "chapter_10/onnx_session_options_demo.cpp" \
    "c++ -std=c++17 -O2 \"$REPO_ROOT/chapter_10/onnx_session_options_demo.cpp\" -o \"$BIN_DIR/onnx_session_options_demo\" $ONNX_FLAGS" \
    "$RUN_ENV \"$BIN_DIR/onnx_session_options_demo\""

  build_only_case "chapter_10/onnx_loader_demo.cpp" \
    "c++ -std=c++17 -O2 \"$REPO_ROOT/chapter_10/onnx_loader_demo.cpp\" -o \"$BIN_DIR/onnx_loader_demo\" $ONNX_FLAGS" \
    "no compatible chapter_10 ONNX artifact is checked into the repo"

  build_only_case "chapter_10/onnx_int8_demo.cpp" \
    "c++ -std=c++17 -O2 \"$REPO_ROOT/chapter_10/onnx_int8_demo.cpp\" -o \"$BIN_DIR/onnx_int8_demo\" $ONNX_FLAGS" \
    "no compatible chapter_10 INT8 ONNX artifact is checked into the repo"
else
  skip_case "chapter_10/onnx_session_options_demo.cpp" "onnxruntime is not installed"
  skip_case "chapter_10/onnx_loader_demo.cpp" "onnxruntime is not installed"
  skip_case "chapter_10/onnx_int8_demo.cpp" "onnxruntime is not installed"
fi

if [[ $HAS_LIBTORCH -eq 1 ]]; then
  build_only_case "chapter_10/fp16_torchscript_demo.cpp" \
    "c++ -std=c++17 -O2 \"$REPO_ROOT/chapter_10/fp16_torchscript_demo.cpp\" -o \"$BIN_DIR/fp16_torchscript_demo\" $LIBTORCH_FLAGS" \
    "GPU execution is required for this demo"

  build_only_case "chapter_10/cuda_graph_capture_demo.cpp" \
    "c++ -std=c++17 -O2 \"$REPO_ROOT/chapter_10/cuda_graph_capture_demo.cpp\" -o \"$BIN_DIR/cuda_graph_capture_demo\" $LIBTORCH_FLAGS" \
    "requires a CUDA-enabled LibTorch build and CUDA runtime"
else
  skip_case "chapter_10/fp16_torchscript_demo.cpp" "libtorch/ is not present; see README.md"
  skip_case "chapter_10/cuda_graph_capture_demo.cpp" "libtorch/ is not present; see README.md"
fi

for base in fixed_bin_histogram_psi structured_logging_demo metrics_tracing_demo repro_pack_and_sentinels_demo micro_batcher_debug_demo; do
  run_case "chapter_11/${base}.cpp" \
    "c++ -std=c++17 -O2 \"$REPO_ROOT/chapter_11/${base}.cpp\" -o \"$BIN_DIR/${base}\"" \
    "\"$BIN_DIR/${base}\""
done

skip_case "chapter_11/cuda_check_macro_demo.cu" "nvcc is not installed in this environment"

for base in p95_demo observability_demo quality_monitor_demo; do
  run_case "chapter_12/${base}.cpp" \
    "c++ -std=c++17 -O2 \"$REPO_ROOT/chapter_12/${base}.cpp\" -o \"$BIN_DIR/${base}\"" \
    "\"$BIN_DIR/${base}\""
done

run_metrics_server_case

run_case "chapter_13/lime_shap_demo.cpp" \
  "c++ -std=c++17 -O2 \"$REPO_ROOT/chapter_13/lime_shap_demo.cpp\" -o \"$BIN_DIR/lime_shap_demo\" $EIGEN_FLAGS" \
  "\"$BIN_DIR/lime_shap_demo\""

if [[ $HAS_LIBTORCH -eq 1 ]]; then
  run_case "chapter_13/gradcam_demo.cpp" \
    "c++ -std=c++17 -O2 \"$REPO_ROOT/chapter_13/gradcam_demo.cpp\" -o \"$BIN_DIR/gradcam_demo\" $LIBTORCH_FLAGS \$(pkg-config --cflags --libs opencv4)" \
    "$RUN_ENV \"$BIN_DIR/gradcam_demo\" \"$REPO_ROOT/input.jpg\" 3"
else
  skip_case "chapter_13/gradcam_demo.cpp" "libtorch/ is not present; see README.md"
fi

note
note "Summary"
note "  total: $TOTAL"
note "  pass : $PASS"
note "  skip : $SKIP"
note "  fail : $FAIL"
note "  logs : $LOG_DIR"

if [[ $FAIL -ne 0 ]]; then
  exit 1
fi
