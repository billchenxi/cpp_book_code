# Caffe2 Normalize Demo

This folder is a standalone support subproject rather than a chapter folder. It keeps the older Caffe2 normalization example together with its own CMake build.

Build from the repository root:

```bash
cmake -S data_pipeline_caffe2 -B data_pipeline_caffe2/build \
  -DTorch_DIR=./libtorch/share/cmake/Torch
cmake --build data_pipeline_caffe2/build -j
```

Run:

```bash
DYLD_LIBRARY_PATH=./libtorch/lib ./data_pipeline_caffe2/build/caffe2_normalize_demo
```
