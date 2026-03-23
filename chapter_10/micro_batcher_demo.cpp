#include <torch/script.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <future>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "device_utils.h"

using Clock = std::chrono::steady_clock;
using ms = std::chrono::milliseconds;

struct InferenceTask {
  torch::Tensor sample_chw;
  std::promise<torch::Tensor> promise;
};

class MicroBatcher {
 public:
  MicroBatcher(torch::jit::script::Module& module,
               int max_batch,
               int max_delay_ms,
               size_t queue_cap)
      : module_(module),
        max_batch_(max_batch),
        max_delay_ms_(max_delay_ms),
        queue_cap_(queue_cap),
        stop_(false),
        worker_(&MicroBatcher::run, this) {}

  ~MicroBatcher() { shutdown(); }

  std::future<torch::Tensor> submit(torch::Tensor chw) {
    std::promise<torch::Tensor> promise;
    auto future = promise.get_future();
    {
      std::unique_lock<std::mutex> lock(mu_);
      if (queue_.size() >= queue_cap_) {
        promise.set_exception(std::make_exception_ptr(
            std::runtime_error("server busy")));
        return future;
      }
      queue_.push_back(InferenceTask{std::move(chw), std::move(promise)});
    }
    cv_.notify_one();
    return future;
  }

  void shutdown() {
    bool expected = false;
    if (stop_.compare_exchange_strong(expected, true)) {
      cv_.notify_all();
      if (worker_.joinable()) {
        worker_.join();
      }
    }
  }

 private:
  void run() {
    torch::NoGradGuard ng;
    const bool use_cuda = chapter10::cuda_available();
    if (use_cuda) {
      module_.to(torch::kCUDA);
    }

    while (!stop_) {
      std::vector<InferenceTask> batch;
      batch.reserve(max_batch_);

      {
        std::unique_lock<std::mutex> lock(mu_);
        cv_.wait(lock, [&] { return stop_ || !queue_.empty(); });
        if (stop_) {
          break;
        }

        auto window_start = Clock::now();
        batch.push_back(std::move(queue_.front()));
        queue_.pop_front();

        while (batch.size() < static_cast<size_t>(max_batch_)) {
          auto elapsed =
              std::chrono::duration_cast<ms>(Clock::now() - window_start)
                  .count();
          if (elapsed >= max_delay_ms_) {
            break;
          }
          if (queue_.empty()) {
            cv_.wait_for(lock,
                         ms(max_delay_ms_ - static_cast<int>(elapsed)),
                         [&] { return stop_ || !queue_.empty(); });
            if (stop_ || queue_.empty()) {
              break;
            }
          }
          batch.push_back(std::move(queue_.front()));
          queue_.pop_front();
        }
      }

      if (batch.empty()) {
        continue;
      }

      std::vector<torch::Tensor> rows;
      rows.reserve(batch.size());
      for (auto& task : batch) {
        rows.push_back(task.sample_chw.unsqueeze(0));
      }

      auto x = torch::cat(rows, 0).contiguous();
      if (use_cuda) {
        x = x.to(torch::kCUDA).to(torch::MemoryFormat::ChannelsLast);
      }

      auto y = module_.forward({x}).toTensor().contiguous();
      for (size_t i = 0; i < batch.size(); ++i) {
        auto row = y.index({static_cast<int64_t>(i)}).cpu();
        batch[i].promise.set_value(row);
      }
    }
  }

  torch::jit::script::Module& module_;
  const int max_batch_;
  const int max_delay_ms_;
  const size_t queue_cap_;
  std::mutex mu_;
  std::condition_variable cv_;
  std::deque<InferenceTask> queue_;
  std::atomic<bool> stop_;
  std::thread worker_;
};

int main(int argc, char** argv) {
  const std::string model_path =
      argc > 1 ? argv[1] : "chapter_10/tinynet.ts";

  try {
    auto net = torch::jit::load(model_path);
    net.eval();

    MicroBatcher batcher(net, /*max_batch=*/16, /*max_delay_ms=*/8,
                         /*queue_cap=*/256);

    const int producers = 4;
    std::vector<std::thread> threads;
    threads.reserve(producers);
    for (int p = 0; p < producers; ++p) {
      threads.emplace_back([&batcher, p] {
        for (int i = 0; i < 8; ++i) {
          auto chw = torch::randn({3, 224, 224}, torch::kFloat);
          auto future = batcher.submit(chw);
          try {
            auto out = future.get();
            if (p == 0) {
              std::cout << "producer " << p
                        << " logit[0]=" << out[0].item<float>() << "\n";
            }
          } catch (const std::exception& ex) {
            std::cerr << "submit failed: " << ex.what() << "\n";
          }
          std::this_thread::sleep_for(ms(2));
        }
      });
    }

    for (auto& thread : threads) {
      thread.join();
    }
    batcher.shutdown();
  } catch (const std::exception& ex) {
    std::cerr << "fatal: " << ex.what() << "\n";
    return 1;
  }

  return 0;
}
