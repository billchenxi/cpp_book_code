# Chapter 12 Code

This folder groups the code referenced by chapter 12 of `Final Draft Submission_B22398_12.pdf`.

Files in this chapter:

- `p95_demo.cpp`: partial-selection percentile helper for rolling latency windows.
- `metrics.hpp`: counters, gauges, histograms, labels, and the in-process metrics registry.
- `metrics_server.cpp`: tiny `/metrics` exporter using a POSIX socket HTTP handler.
- `log.hpp`: ISO8601 timestamp and structured JSON logging helper.
- `trace.hpp`: RAII span helper that logs span durations and updates histogram metrics.
- `quality.hpp`: delayed-label joiner, calibration metrics, disagreement counter, and confidence proxies.
- `cohort_quality.hpp`: cohort-level moving summaries for entropy, margin, disagreement, and abstention.
- `observability_demo.cpp`: runnable logging, tracing, and metrics demo.
- `quality_monitor_demo.cpp`: runnable delayed-label and leading-indicator monitoring demo.

## Start

All commands below assume you run them from the repository root.

Percentile helper:

```bash
g++ -std=c++17 -O2 chapter_12/p95_demo.cpp -o chapter_12/p95_demo
./chapter_12/p95_demo
```

Observability demo:

```bash
g++ -std=c++17 -O2 chapter_12/observability_demo.cpp -o chapter_12/observability_demo
./chapter_12/observability_demo
```

Online quality demo:

```bash
g++ -std=c++17 -O2 chapter_12/quality_monitor_demo.cpp -o chapter_12/quality_monitor_demo
./chapter_12/quality_monitor_demo
```

Metrics server and scrape demo:

```bash
g++ -std=c++17 -O2 -pthread chapter_12/metrics_server.cpp -o chapter_12/metrics_server
./chapter_12/metrics_server 9100 15
```

While it is running, scrape the endpoint from another terminal:

```bash
curl -s http://127.0.0.1:9100/metrics
```

## Notes

- `metrics_server.cpp` is POSIX-specific because the chapter shows a tiny socket-based HTTP handler.
- `metrics.hpp`, `log.hpp`, `trace.hpp`, `quality.hpp`, and `cohort_quality.hpp` are reusable headers, while the three `*_demo.cpp` files are the runnable entry points.
- The chapter text shows `log.hpp` usage but omits the actual `log_json` implementation, so this repo includes a small compatible helper to make the logging and tracing examples runnable.
