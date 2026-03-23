#pragma once

#if __has_include(<torch/cuda.h>)
#include <torch/cuda.h>
#endif

namespace chapter10 {

inline bool cuda_available() {
#if __has_include(<torch/cuda.h>)
  return torch::cuda::is_available();
#else
  return false;
#endif
}

}  // namespace chapter10
