#include <chrono>
#include <condition_variable>
#include <deque>
#include <future>
#include <iostream>
#include <mutex>
#include <numeric>
#include <string>
#include <thread>
#include <utility>
#include <vector>

using Tensor = std::vector<float>;

struct Task {
  std::string id;
  Tensor x;
  std::promise<Tensor> p;
  std::chrono::steady_clock::time_point submitted_at;
};

class MicroBatcher {
 public:
  MicroBatcher(size_t max_batch, int max_delay_ms)
      : max_batch_(max_batch),
        max_delay_ms_(max_delay_ms),
        stop_(false),
        worker_([this] { run(); }) {}

  ~MicroBatcher() {
    {
      std::lock_guard<std::mutex> lk(mu_);
      stop_ = true;
    }
    cv_.notify_all();
    if (worker_.joinable()) {
      worker_.join();
    }
  }

  std::future<Tensor> submit(std::string id, Tensor x) {
    std::promise<Tensor> p;
    auto f = p.get_future();
    {
      std::lock_guard<std::mutex> lk(mu_);
      q_.push_back({std::move(id), std::move(x), std::move(p),
                    std::chrono::steady_clock::now()});
    }
    cv_.notify_one();
    return f;
  }

 private:
  void run() {
    for (;;) {
      std::deque<Task> batch;
      {
        std::unique_lock<std::mutex> lk(mu_);
        if (stop_ && q_.empty()) {
          break;
        }
        if (q_.empty()) {
          cv_.wait(lk);
          continue;
        }

        auto deadline = std::chrono::steady_clock::now() +
                        std::chrono::milliseconds(max_delay_ms_);
        while (q_.size() < max_batch_) {
          if (cv_.wait_until(lk, deadline) == std::cv_status::timeout) {
            break;
          }
          if (stop_) {
            break;
          }
        }

        while (!q_.empty() && batch.size() < max_batch_) {
          batch.push_back(std::move(q_.front()));
          q_.pop_front();
        }
      }

      const size_t batch_size = batch.size();
      for (auto& t : batch) {
        const auto now = std::chrono::steady_clock::now();
        const auto queue_us =
            std::chrono::duration_cast<std::chrono::microseconds>(
                now - t.submitted_at)
                .count();
        const float sum =
            std::accumulate(t.x.begin(), t.x.end(), 0.0f) +
            static_cast<float>(batch_size);
        std::cout << "task=" << t.id << " queue_us=" << queue_us
                  << " batch_size=" << batch_size << "\n";
        t.p.set_value(Tensor{sum});
      }
    }
  }

  size_t max_batch_;
  int max_delay_ms_;
  std::mutex mu_;
  std::condition_variable cv_;
  std::deque<Task> q_;
  std::thread worker_;
  bool stop_;
};

int main() {
  using namespace std::chrono_literals;

  MicroBatcher batcher(/*max_batch=*/3, /*max_delay_ms=*/10);
  std::vector<std::future<Tensor>> futures;

  for (int i = 0; i < 6; ++i) {
    futures.push_back(
        batcher.submit("req-" + std::to_string(i), Tensor{1.0f, float(i)}));
    std::this_thread::sleep_for(3ms);
  }

  for (size_t i = 0; i < futures.size(); ++i) {
    auto out = futures[i].get();
    std::cout << "result[" << i << "]=" << out.front() << "\n";
  }
  return 0;
}
