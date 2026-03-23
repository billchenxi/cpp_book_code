#include <atomic>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <string>
#include <thread>

#include "observability_utils.h"

struct Metrics {
  std::atomic<uint64_t> req_total{0};
  std::atomic<uint64_t> err_total{0};
  std::atomic<uint64_t> q_time_us{0};
  std::atomic<uint64_t> infer_time_us{0};
  std::atomic<uint64_t> bytes_in{0};
  std::atomic<uint64_t> bytes_out{0};

  void observe_request(uint64_t q_us,
                       uint64_t i_us,
                       uint64_t in_b,
                       uint64_t out_b) {
    req_total++;
    q_time_us += q_us;
    infer_time_us += i_us;
    bytes_in += in_b;
    bytes_out += out_b;
  }

  void observe_error() { err_total++; }
};

inline void flush_metrics(const Metrics& m) {
  std::cout << "req_total " << m.req_total.load() << "\n"
            << "err_total " << m.err_total.load() << "\n"
            << "q_time_us " << m.q_time_us.load() << "\n"
            << "infer_time_us " << m.infer_time_us.load() << "\n"
            << "bytes_in " << m.bytes_in.load() << "\n"
            << "bytes_out " << m.bytes_out.load() << "\n";
}

struct Request {
  std::string id;
};

void handle_request(const Request& r, Metrics& metrics) {
  using namespace std::chrono_literals;

  Span s_all("req_total", r.id);

  auto queue_start = std::chrono::high_resolution_clock::now();
  { Span s_parse("parse", r.id); std::this_thread::sleep_for(2ms); }
  { Span s_pre("preprocess", r.id); std::this_thread::sleep_for(3ms); }
  { Span s_queue("queue", r.id); std::this_thread::sleep_for(4ms); }
  auto infer_start = std::chrono::high_resolution_clock::now();
  { Span s_infer("infer", r.id); std::this_thread::sleep_for(5ms); }
  auto infer_end = std::chrono::high_resolution_clock::now();
  { Span s_post("postprocess", r.id); std::this_thread::sleep_for(1ms); }
  auto request_end = std::chrono::high_resolution_clock::now();

  const auto infer_us =
      std::chrono::duration_cast<std::chrono::microseconds>(infer_end -
                                                            infer_start)
          .count();
  const auto total_us =
      std::chrono::duration_cast<std::chrono::microseconds>(request_end -
                                                            queue_start)
          .count();
  const auto queue_us = total_us > infer_us ? total_us - infer_us : 0;
  metrics.observe_request(queue_us, infer_us, 8 * 3 * 224 * 224 * 4, 8 * 1000);
}

int main() {
  Metrics metrics;
  handle_request({"req-1001"}, metrics);
  flush_metrics(metrics);
  return 0;
}
