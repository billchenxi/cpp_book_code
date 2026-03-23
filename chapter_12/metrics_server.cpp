#include "metrics.hpp"

#include <arpa/inet.h>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <netinet/in.h>
#include <sstream>
#include <string>
#include <sys/socket.h>
#include <thread>
#include <unistd.h>

void serve_metrics(Registry& reg, int port = 9100) {
  int fd = socket(AF_INET, SOCK_STREAM, 0);
  int yes = 1;
  setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));

  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = INADDR_ANY;
  addr.sin_port = htons(port);

  bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr));
  listen(fd, 16);

  std::thread([fd, &reg]() {
    for (;;) {
      int c = accept(fd, nullptr, nullptr);
      if (c < 0) {
        continue;
      }

      std::string body = reg.render_metrics_text();
      std::ostringstream resp;
      resp << "HTTP/1.1 200 OK\r\n"
           << "Content-Type: text/plain; version=0.0.4\r\n"
           << "Content-Length: " << body.size() << "\r\n\r\n"
           << body;

      auto s = resp.str();
      (void)write(c, s.data(), s.size());
      close(c);
    }
  }).detach();
}

int main(int argc, char** argv) {
  const int port = argc > 1 ? std::atoi(argv[1]) : 9100;
  const int duration_sec = argc > 2 ? std::atoi(argv[2]) : 15;

  Registry r;
  auto& req_total =
      r.counter("inference_requests_total",
                {{{"model", "m1"}, {"version", "1.3"}}});
  auto& err_total =
      r.counter("inference_errors_total",
                {{{"model", "m1"}, {"version", "1.3"}}});
  auto& lat_histo =
      r.histo("inference_latency_us",
              {{{"model", "m1"}, {"version", "1.3"}}});
  auto& gpu_util =
      r.gauge("gpu_util_percent", {{{"device", "cuda:0"}}});
  auto& vram_free =
      r.gauge("gpu_vram_free_mb", {{{"device", "cuda:0"}}});

  serve_metrics(r, port);
  std::cout << "Serving metrics on http://127.0.0.1:" << port
            << "/metrics for " << duration_sec << " seconds\n";

  for (int i = 0; i < duration_sec; ++i) {
    req_total.inc(2);
    if (i % 5 == 0) {
      err_total.inc();
    }
    lat_histo.observe(1000 + 500 * static_cast<uint64_t>(i));
    lat_histo.observe(2000 + 800 * static_cast<uint64_t>(i));
    gpu_util.set(35 + i);
    vram_free.set(4096 - 16 * i);
    std::this_thread::sleep_for(std::chrono::seconds(1));
  }
  return 0;
}
