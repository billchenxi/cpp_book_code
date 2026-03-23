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
  auto sub = duration_cast<microseconds>(t - s).count();
  std::time_t tt = system_clock::to_time_t(t);
  std::tm tm = *std::gmtime(&tt);

  std::ostringstream os;
  os << std::put_time(&tm, "%FT%T") << "." << std::setw(6)
     << std::setfill('0') << sub << "Z";
  return os.str();
}

inline void log_json(const std::string& level,
                     const std::string& msg,
                     const std::string& req_id,
                     const std::string& trace_id,
                     const std::string& model,
                     const std::string& version,
                     const std::string& device,
                     int batch,
                     const std::string& extra_json = "{}") {
  std::cout << "{\"ts\":\"" << iso8601_now()
            << "\",\"level\":\"" << level
            << "\",\"msg\":\"" << msg
            << "\",\"req\":\"" << req_id
            << "\",\"trace\":\"" << trace_id
            << "\",\"model\":\"" << model
            << "\",\"ver\":\"" << version
            << "\",\"device\":\"" << device
            << "\",\"batch\":" << batch
            << ",\"extra\":" << extra_json << "}\n";
}
