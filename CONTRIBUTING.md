# Contributing

This repo is a collection of small C++ demos, not a single product build. Contributions should keep that shape: focused examples, clear dependencies, and lightweight setup.

## What Fits Here

- small, self-contained examples for ML, preprocessing, inference, systems, or signal-processing topics
- short supporting headers or CMake subprojects when a standalone file is not enough
- checked-in assets only when they are genuinely needed for the example and are documented

## Expectations For New Demos

1. Use a descriptive filename such as `rolling_mean_demo.cpp` or `bert_onnx_demo.cpp`.
2. Prefer C++17 and keep the example as small as possible.
3. Add a short header comment with:
   - what the demo does
   - how to build it
   - how to run it
   - any required input files or libraries
4. Keep the program deterministic when practical by fixing seeds or using fixed toy inputs.
5. Write outputs to predictable filenames and mention those outputs in the source comment if they are not obvious.

## Dependency Rules

- Prefer the standard library when it is enough.
- If you need third-party libraries, use the lightest dependency that makes the example clear.
- Prefer `pkg-config` for simple one-file builds and CMake for subprojects.
- Reuse the vendored `libtorch/` directory for LibTorch examples when possible.
- If you add a new dependency, update [README.md](README.md) and [docs/demo-index.md](docs/demo-index.md).

## Large Files And Assets

- Do not add generated binaries, build directories, or output images.
- If you add a large checked-in asset, document why it belongs in the repo and update the "Large Checked-In Assets" section in [README.md](README.md).
- Model assets should live in a clearly named directory with enough tokenizer or config files to run the example.

## Documentation To Update

When you add or substantially change a demo, update:

- [README.md](README.md) if the change affects setup, repo layout, dependencies, or quick-start guidance
- [docs/demo-index.md](docs/demo-index.md) so the file is listed in the right category

The source comment at the top of each file is the canonical place for per-demo build and run commands.

## Verification

Before submitting changes, build and run at least the demos you touched. Record the exact command you used so another contributor can repeat it.

Good verification examples:

- `g++ -std=c++17 -O2 rolling_mean_demo.cpp -o /tmp/roll_check && /tmp/roll_check`
- `g++ -std=c++17 -O2 image_preproc_demo.cpp -o /tmp/image_preproc_demo $(pkg-config --cflags --libs opencv4) && /tmp/image_preproc_demo input.jpg`
- `cmake -S data_pipeline -B data_pipeline/build -DCMAKE_PREFIX_PATH=./libtorch && cmake --build data_pipeline/build -j`

## Style Notes

- Keep comments short and technical.
- Favor explicit code over heavy abstraction.
- Avoid hidden inputs, network fetches, or environment assumptions unless they are documented.
- Treat the source file as the source of truth; checked-in binaries are local artifacts, not primary deliverables.
