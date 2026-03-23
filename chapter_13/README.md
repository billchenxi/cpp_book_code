# Chapter 13 Code

This folder groups the code referenced by chapter 13 of `B22398_13_032026_xc_edited.docx`.

Files in this chapter:

- `explainability.hpp`: reusable LIME and KernelSHAP helper types and functions based on the chapter snippets.
- `lime_shap_demo.cpp`: runnable model-agnostic tabular demo that uses the header to generate a LIME explanation and KernelSHAP attributions.
- `gradcam_demo.cpp`: LibTorch plus OpenCV Grad-CAM walkthrough for a small CNN, following the chapter's vision explainability section.

## Start

All commands below assume you run them from the repository root.

LIME and KernelSHAP demo:

```bash
g++ -std=c++17 -O2 chapter_13/lime_shap_demo.cpp -o chapter_13/lime_shap_demo \
  -I"$(brew --prefix eigen)/include/eigen3"
./chapter_13/lime_shap_demo
```

Grad-CAM demo:

```bash
g++ -std=c++17 -O2 chapter_13/gradcam_demo.cpp -o chapter_13/gradcam_demo \
  -I./libtorch/include \
  -I./libtorch/include/torch/csrc/api/include \
  -L./libtorch/lib \
  -Wl,-rpath,./libtorch/lib \
  $(pkg-config --cflags --libs opencv4) \
  -ltorch -ltorch_cpu -lc10

DYLD_LIBRARY_PATH=./libtorch/lib ./chapter_13/gradcam_demo input.jpg 3
```

The Grad-CAM demo writes `chapter_13/gradcam_overlay.png`.

## Notes

- `lime_shap_demo.cpp` uses a small synthetic regression-style model so the explainability pipeline is runnable without extra model files.
- `gradcam_demo.cpp` intentionally uses random model weights because the manuscript focuses on the capture and visualization pattern, not pretrained accuracy.
- The chapter's responsible-AI sections around uncertainty, abstention, audit trails, and model cards are prose in the manuscript, so this folder focuses on the runnable code examples.
