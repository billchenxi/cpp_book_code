#include <string>

#include "observability_utils.h"

int main() {
  log_structured("INFO",
                 "recv",
                 "r-7f23",
                 "resnet50",
                 "1.12.3",
                 "cpu",
                 8,
                 "{\"shape\":[8,3,224,224],\"deadline_ms\":200}");

  log_structured("INFO",
                 "done",
                 "r-7f23",
                 "resnet50",
                 "1.12.3",
                 "cpu",
                 8,
                 "{\"p50_ms\":12.1,\"p99_ms\":27.4,\"gpu_mem_mb\":0,\"cache_hit\":true}");
  return 0;
}
