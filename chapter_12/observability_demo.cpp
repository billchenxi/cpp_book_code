#include "log.hpp"
#include "metrics.hpp"
#include "trace.hpp"

#include <chrono>
#include <cstdint>
#include <iostream>
#include <string>
#include <thread>

struct Request {
  std::string id;
  std::string trace;
  std::string device;
  int batch;
};

void handle_request(const Request& r, Registry& R) {
  using namespace std::chrono_literals;

  auto& req_total =
      R.counter("inference_requests_total",
                {{{"model", "m1"}, {"version", "1.3"}}});
  auto& lat_histo =
      R.histo("inference_latency_us",
              {{{"model", "m1"}, {"version", "1.3"}}});

  const auto t0 = std::chrono::high_resolution_clock::now();
  req_total.inc();

  log_json("INFO",
           "recv",
           r.id,
           r.trace,
           "m1",
           "1.3",
           r.device,
           r.batch,
           "{\"shape\":[1,3,224,224]}");

  Span s_all("request_total", r.id, r.trace, "m1", "1.3", r.device, &R);
  { Span s("parse", r.id, r.trace, "m1", "1.3", r.device, &R); std::this_thread::sleep_for(2ms); }
  { Span s("preprocess", r.id, r.trace, "m1", "1.3", r.device, &R); std::this_thread::sleep_for(3ms); }
  { Span s("queue", r.id, r.trace, "m1", "1.3", r.device, &R); std::this_thread::sleep_for(4ms); }
  { Span s("infer", r.id, r.trace, "m1", "1.3", r.device, &R); std::this_thread::sleep_for(5ms); }
  { Span s("postprocess", r.id, r.trace, "m1", "1.3", r.device, &R); std::this_thread::sleep_for(2ms); }
  { Span s("serialize", r.id, r.trace, "m1", "1.3", r.device, &R); std::this_thread::sleep_for(1ms); }

  const auto dur_us = std::chrono::duration_cast<std::chrono::microseconds>(
                          std::chrono::high_resolution_clock::now() - t0)
                          .count();
  lat_histo.observe(static_cast<uint64_t>(dur_us));

  log_json("INFO",
           "done",
           r.id,
           r.trace,
           "m1",
           "1.3",
           r.device,
           r.batch,
           std::string("{\"lat_us\":") + std::to_string(dur_us) + "}");
}

int main() {
  Registry R;
  handle_request({"r-7f23", "t-9ab1", "cpu", 1}, R);
  std::cout << "\n--- metrics ---\n" << R.render_metrics_text();
  return 0;
}
