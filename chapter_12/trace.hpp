#pragma once

#include "log.hpp"
#include "metrics.hpp"

#include <chrono>
#include <string>
#include <utility>

struct Span {
  const char* name;
  std::string req_id;
  std::string trace_id;
  std::string model;
  std::string version;
  std::string device;
  Registry* reg;
  std::chrono::high_resolution_clock::time_point t0;

  Span(const char* n,
       std::string req,
       std::string trace,
       std::string m,
       std::string v,
       std::string dev,
       Registry* r)
      : name(n),
        req_id(std::move(req)),
        trace_id(std::move(trace)),
        model(std::move(m)),
        version(std::move(v)),
        device(std::move(dev)),
        reg(r),
        t0(std::chrono::high_resolution_clock::now()) {}

  ~Span() {
    using namespace std::chrono;
    const auto us =
        duration_cast<microseconds>(high_resolution_clock::now() - t0).count();

    log_json("TRACE",
             "span",
             req_id,
             trace_id,
             model,
             version,
             device,
             1,
             std::string("{\"span\":\"") + name + "\",\"dur_us\":" +
                 std::to_string(us) + "}");

    if (reg) {
      Labels l;
      l.kv = {{"model", model},
              {"version", version},
              {"span", name},
              {"device", device}};
      reg->histo("span_latency_us", l).observe(static_cast<uint64_t>(us));
    }
  }
};
