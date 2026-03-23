#include <torch/script.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <string>

#include "device_utils.h"

using Clock = std::chrono::steady_clock;

static double bench(torch::jit::script::Module& module,
                    int batch_size,
                    int iters) {
  torch::NoGradGuard ng;

  auto x = torch::randn({batch_size, 3, 224, 224});
  if (chapter10::cuda_available()) {
    module.to(torch::kCUDA);
    x = x.to(torch::kCUDA).to(torch::MemoryFormat::ChannelsLast);
  }

  for (int i = 0; i < 10; ++i) {
    (void)module.forward({x});
  }

  auto t0 = Clock::now();
  for (int i = 0; i < iters; ++i) {
    (void)module.forward({x});
  }
  auto elapsed_us =
      std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - t0)
          .count();

  elapsed_us = std::max<int64_t>(elapsed_us, 1);
  return 1000000.0 * iters * batch_size / static_cast<double>(elapsed_us);
}

int main(int argc, char** argv) {
  const std::string teacher_path =
      argc > 1 ? argv[1] : "chapter_10/teacher.ts";
  const std::string student_path =
      argc > 2 ? argv[2] : teacher_path;
  const int batch_size = argc > 3 ? std::stoi(argv[3]) : 16;
  const int iters = argc > 4 ? std::stoi(argv[4]) : 50;

  auto teacher = torch::jit::load(teacher_path);
  auto student = torch::jit::load(student_path);
  teacher.eval();
  student.eval();

  std::cout << "teacher img/s: " << bench(teacher, batch_size, iters) << "\n";
  std::cout << "student img/s: " << bench(student, batch_size, iters) << "\n";
  return 0;
}
