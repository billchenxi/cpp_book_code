# C++ ML Book Code

This repo contains the code examples grouped by book chapter. Each chapter folder has its own `README.md` with the exact build and run commands.

## Start

- Use a C++17 compiler.
- Install the libraries needed by the chapter you want to run.
- Open the chapter README and run the commands from there.

Common macOS packages:

```bash
brew install cmake pkg-config opencv fftw libsndfile onnxruntime armadillo mlpack eigen snowball cereal
```

## Downloaded Assets

Large downloaded dependencies and model bundles are not committed to GitHub. Some chapters expect these local paths at the repo root:

- `libtorch/`
- `bert_onnx/`
- `ner_onnx/`
- `sent_onnx/`

LibTorch example for macOS arm64 CPU:

```bash
curl -L -o libtorch-macos-arm64-2.8.0.zip \
  https://download.pytorch.org/libtorch/cpu/libtorch-macos-arm64-2.8.0.zip
unzip libtorch-macos-arm64-2.8.0.zip
```

BERT ONNX bundle:

```bash
python3 -m pip install -U huggingface_hub
huggingface-cli download onnx-community/bert-base-uncased --local-dir bert_onnx
```

NER and sentiment ONNX exports:

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

## Chapters

- [chapter_1/README.md](chapter_1/README.md): LibTorch basics, MNIST-style network, CUDA hello example, SIMD demo.
- [chapter_2/README.md](chapter_2/README.md): preprocessing, feature engineering, image and audio prep, ONNX, LibTorch datasets.
- [chapter_3/README.md](chapter_3/README.md): CUDA vector-add progression and CUDA error checking.
- [chapter_4/README.md](chapter_4/README.md): linear regression, logistic regression, neuron demo, Eigen MLP.
- [chapter_5/README.md](chapter_5/README.md): XOR MLPs, optimizers, CUDA MLP.
- [chapter_10/README.md](chapter_10/README.md): TorchScript, ONNX Runtime, micro-batching, benchmarking, deployment demos.
- [chapter_11/README.md](chapter_11/README.md): drift monitoring, observability, debugging, acceptance gates.
- [chapter_12/README.md](chapter_12/README.md): metrics, traces, logs, metrics server, online quality monitoring.
- [chapter_13/README.md](chapter_13/README.md): LIME, KernelSHAP, Grad-CAM.

## Verification

To run the automated verification script:

```bash
./scripts/verify_examples.sh
```

The script skips demos whose large local assets are not present.
