#ifndef MEMBENCH_PLATFORM_PLATFORM_H
#define MEMBENCH_PLATFORM_PLATFORM_H

#include "core/types.h"

#include <cstdint>
#include <string>
#include <vector>

namespace membench {

std::vector<unsigned int> parseCpuList(const std::string& text);
PlatformInfo detectPlatformInfo();
std::uint64_t monotonicNowNs();
void applyCurrentThreadPolicy(RunMode mode, bool use_qos);
void applyCurrentThreadAffinity(const std::vector<unsigned int>& cpu_affinity_order,
                                unsigned int worker_index);

}  // namespace membench

#endif  // MEMBENCH_PLATFORM_PLATFORM_H
