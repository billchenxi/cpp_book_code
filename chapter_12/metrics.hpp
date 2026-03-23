#pragma once

#include <array>
#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

struct Counter {
  std::atomic<uint64_t> value{0};

  void inc(uint64_t n = 1) { value.fetch_add(n, std::memory_order_relaxed); }

  uint64_t get() const { return value.load(std::memory_order_relaxed); }
};

struct Gauge {
  std::atomic<long long> value{0};

  void set(long long v) { value.store(v, std::memory_order_relaxed); }

  long long get() const { return value.load(std::memory_order_relaxed); }
};

struct Histogram {
  // Fixed buckets in microseconds: 1,2,4,8,16,32,64,128,256, >=256 ms
  static constexpr int K = 10;
  static constexpr std::array<uint64_t, K - 1> bounds_us{
      1000, 2000, 4000, 8000, 16000, 32000, 64000, 128000, 256000};

  std::array<std::atomic<uint64_t>, K> buckets{};
  std::atomic<uint64_t> sum_us{0};
  std::atomic<uint64_t> count{0};

  void observe(uint64_t us) {
    size_t i = 0;
    while (i < K - 1 && us > bounds_us[i]) {
      ++i;
    }
    buckets[i].fetch_add(1, std::memory_order_relaxed);
    sum_us.fetch_add(us, std::memory_order_relaxed);
    count.fetch_add(1, std::memory_order_relaxed);
  }
};

struct Labels {
  std::vector<std::pair<std::string, std::string>> kv;

  std::string to_text() const {
    if (kv.empty()) {
      return "";
    }
    std::ostringstream os;
    os << "{";
    for (size_t i = 0; i < kv.size(); ++i) {
      if (i) {
        os << ",";
      }
      os << kv[i].first << "=\"" << kv[i].second << "\"";
    }
    os << "}";
    return os.str();
  }
};

class Registry {
 public:
  Counter& counter(const std::string& name, const Labels& lbl = {}) {
    return get(counters_, name, lbl);
  }

  Gauge& gauge(const std::string& name, const Labels& lbl = {}) {
    return get(gauges_, name, lbl);
  }

  Histogram& histo(const std::string& name, const Labels& lbl = {}) {
    return get(histos_, name, lbl);
  }

  std::string render_metrics_text() const {
    std::lock_guard<std::mutex> lk(mu_);
    std::ostringstream os;

    for (const auto& it : counters_) {
      os << "# TYPE " << it.first << " counter\n";
      for (const auto& row : it.second) {
        os << it.first << row.labels.to_text() << " " << row.metric->get()
           << "\n";
      }
    }

    for (const auto& it : gauges_) {
      os << "# TYPE " << it.first << " gauge\n";
      for (const auto& row : it.second) {
        os << it.first << row.labels.to_text() << " " << row.metric->get()
           << "\n";
      }
    }

    for (const auto& it : histos_) {
      os << "# TYPE " << it.first << " histogram\n";
      for (const auto& row : it.second) {
        const auto& lbl = row.labels;
        const auto& h = *row.metric;
        uint64_t cum = 0;

        for (size_t i = 0; i < Histogram::K - 1; ++i) {
          cum += h.buckets[i].load(std::memory_order_relaxed);
          Labels l = lbl;
          l.kv.emplace_back("le",
                            std::to_string(Histogram::bounds_us[i] / 1000.0));
          os << it.first << "_bucket" << l.to_text() << " " << cum << "\n";
        }

        cum += h.buckets[Histogram::K - 1].load(std::memory_order_relaxed);
        Labels linf = lbl;
        linf.kv.emplace_back("le", "+Inf");
        os << it.first << "_bucket" << linf.to_text() << " " << cum << "\n";
        os << it.first << "_sum" << lbl.to_text() << " "
           << (h.sum_us.load(std::memory_order_relaxed) / 1000.0) << "\n";
        os << it.first << "_count" << lbl.to_text() << " "
           << h.count.load(std::memory_order_relaxed) << "\n";
      }
    }

    return os.str();
  }

 private:
  template <typename T>
  struct Series {
    Labels labels;
    std::unique_ptr<T> metric;
  };

  template <typename T>
  using Table = std::unordered_map<std::string, std::vector<Series<T>>>;

  template <typename T>
  T& get(Table<T>& table, const std::string& name, const Labels& lbl) {
    std::lock_guard<std::mutex> lk(mu_);
    auto& vec = table[name];
    for (auto& row : vec) {
      if (row.labels.kv == lbl.kv) {
        return *row.metric;
      }
    }
    vec.push_back(Series<T>{lbl, std::make_unique<T>()});
    return *vec.back().metric;
  }

  mutable std::mutex mu_;
  Table<Counter> counters_;
  Table<Gauge> gauges_;
  Table<Histogram> histos_;
};
