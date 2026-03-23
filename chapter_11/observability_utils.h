#pragma once

#include <chrono>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

inline std::string iso8601_now() {
  using namespace std::chrono;
  auto t = system_clock::now();
  auto s = time_point_cast<std::chrono::seconds>(t);
  auto subsecs = duration_cast<microseconds>(t - s).count();
  std::time_t tt = system_clock::to_time_t(t);
  std::tm tm = *std::gmtime(&tt);
  std::ostringstream os;
  os << std::put_time(&tm, "%FT%T") << "." << std::setw(6)
     << std::setfill('0') << subsecs << "Z";
  return os.str();
}

inline void log_structured(const std::string& level,
                           const std::string& msg,
                           const std::string& req_id,
                           const std::string& model,
                           const std::string& version,
                           const std::string& device,
                           int batch,
                           const std::string& extra_json = "{}") {
  std::cout << "{\"ts\":\"" << iso8601_now()
            << "\",\"level\":\"" << level
            << "\",\"msg\":\"" << msg
            << "\",\"req\":\"" << req_id
            << "\",\"model\":\"" << model
            << "\",\"ver\":\"" << version
            << "\",\"device\":\"" << device
            << "\",\"batch\":" << batch
            << ",\"extra\":" << extra_json << "}\n";
}

struct Span {
  std::string name;
  std::string req_id;
  std::chrono::high_resolution_clock::time_point t0;

  Span(std::string n, std::string r)
      : name(std::move(n)),
        req_id(std::move(r)),
        t0(std::chrono::high_resolution_clock::now()) {}

  ~Span() {
    auto us = std::chrono::duration_cast<std::chrono::microseconds>(
                  std::chrono::high_resolution_clock::now() - t0)
                  .count();
    log_structured("TRACE",
                   "span",
                   req_id,
                   "resnet50",
                   "1.12.3",
                   "cpu",
                   1,
                   std::string("{\"span\":\"") + name +
                       "\",\"dur_us\":" + std::to_string(us) + "}");
  }
};
